---
title: "FP4 量化的理想与现实：MR-GPTQ 如何弥合 NVFP4/MXFP4 的精度缺口"
date: 2026-03-07T07:30:00+08:00
tags: ["quantization", "LLM-inference", "Blackwell"]
description: "MR-GPTQ 通过 block-wise Hadamard 旋转和格式专用优化，在 Blackwell GPU 上实现 FP4 量化的高精度和高性能推理"
showToc: true
---

## TL;DR

NVIDIA Blackwell 硬件原生支持 FP4 量化（NVFP4/MXFP4），但直接用现有 PTQ 方法套上去效果很差。IST Austria 和 Red Hat AI 的团队提出了 MR-GPTQ，通过 block-wise Hadamard 旋转 + 格式专用优化，在 RTX 5090 上实现了 6x layer-wise / 4x 端到端加速，同时把 MXFP4 精度拉到接近 NVFP4 水平。

---

## 背景：FP4 硬件已就绪，软件还没跟上

NVIDIA Blackwell 架构（B200、RTX 5090）的 Tensor Core 原生支持两种 FP4 格式：NVFP4（NVIDIA 自有，group size 16，scale 用 E4M3 FP8）和 MXFP4（行业标准 microscaling，group size 32，scale 用 E8M0 即纯 power-of-two）。理论上这意味着 LLM 推理可以比 INT4 更快、模型更小。

与此同时，llama.cpp 社区也在跟进——三天前一个 PR（[#19769](https://github.com/ggml-org/llama.cpp/pull/19769)）给 ggml 加入了 NVFP4 的 block struct 和 CPU 端 quantize/dequantize 支持，CUDA kernel 还在路上。一旦 GPU 支持落地，GGUF 格式的模型可以直接利用 Blackwell 的 FP4 硬件加速。

但问题来了：把现有的 PTQ 方法（SmoothQuant、QuaRot、SpinQuant）直接用在 FP4 上，效果远不如预期。这篇论文就是第一个系统性地回答「FP4 PTQ 到底行不行」的工作。

## 两种 FP4 格式的核心差异

FP4 每个元素只有 4 bit（1 sign + 2 exponent + 1 mantissa），能表示的正值只有 7 个：{0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}。精度全靠 group scale 来补偿。

NVFP4 用 16 个元素一组，scale 是 E4M3 FP8（8 bit，有 mantissa，表达力强），平均每个元素占 4.5 bit。MXFP4 用 32 个元素一组，scale 是 E8M0（纯 power-of-two，只有 exponent），平均 4.25 bit。

关键发现：MXFP4 的 power-of-two scale 导致量化误差显著增大（约 10% 相对精度下降）。更反直觉的是，传统的 outlier 缓解技术（如 Hadamard 旋转）在 NVFP4 上反而有害——因为 NVFP4 的 group size 只有 16，旋转后反而破坏了本来可以被小 group 很好捕捉的局部分布特征。论文用数学证明了这一点：旋转对 Laplace 分布（LLM 权重的典型分布）下的 NVFP4 量化误差是增加的。

## MR-GPTQ 的解法

既然整体旋转不行，那就做 block-wise 的微旋转。MR-GPTQ 的核心思路：

**1. Block-wise Hadamard Transform**

不在整个 hidden dimension 上做旋转，而是在每个 quantization group 内部做 Hadamard 变换。这样既能让组内分布更均匀（对 MXFP4 特别有效），又不会破坏 NVFP4 的小 group 优势。

**2. 权重端旋转融合**

Hadamard 矩阵是正交矩阵，可以预先乘进权重里，推理时零开销。激活端需要在线计算，但他们写了专门的 fused kernel，开销可忽略。

**3. 格式专用 scale search**

针对 MXFP4 的 power-of-two scale 限制，用搜索算法找最优 scale，而不是简单 round-to-nearest。

**4. 改进的 GPTQ activation reordering**

传统 GPTQ 按 Hessian 对角线排序列来决定量化顺序，MR-GPTQ 提出了更高效的变体。

## 实测性能

在 Llama-3 和 Qwen-3 系列模型上，MR-GPTQ 的表现：

- **NVFP4 + MR-GPTQ**：大模型可恢复 FP16 基线 98-99% 的精度
- **MXFP4 + MR-GPTQ**：精度显著提升，接近 NVFP4 水平（差距从 ~10% 缩小到 1-2%）
- **速度**：B200 上 layer-wise 3.6x / 端到端 2.2x；RTX 5090 上 layer-wise 6x / 端到端 4x（对比 FP16）
- 他们的 GPU kernel（叫 QuTLASS）基于 Blackwell 架构专门优化，MXFP4 kernel 的吞吐甚至超过了理论上的 NVFP4 matmul

## 对边缘部署的意义

这个工作目前聚焦在 Blackwell 桌面/数据中心 GPU，但思路对 Jetson 系列也有参考价值。Jetson Orin 的 Tensor Core 支持 INT8/INT4，如果未来的 Jetson 芯片跟进 FP4 支持，MR-GPTQ 的 block-wise rotation 方案可以直接迁移。即便是当前硬件，论文里对 FP4 vs INT4 量化误差的分析框架也值得借鉴——他们证明了 FP4 并不是 INT4 的自动升级，选择量化格式需要根据权重分布特征来决定。

## llama.cpp 生态的进展

配合这篇论文看 llama.cpp 的 NVFP4 PR 会更有意思：目前 PR 只加了 CPU 端支持和 GGUF 转换逻辑，CUDA kernel 还没写。一旦 GPU kernel 落地，Blackwell 用户就可以用 GGUF 格式直接跑 NVFP4 模型，预计速度提升 2-3x，模型体积减少 30-70%。这对 LocalLLaMA 社区是个大事。

## 面试关联知识点

**Q: 模型量化中 PTQ 和 QAT 的区别是什么？各自适用场景？**

PTQ（Post-Training Quantization）不需要重新训练，直接对训练好的权重做量化，速度快但精度损失较大，适合部署时快速压缩。QAT（Quantization-Aware Training）在训练过程中模拟量化误差（forward pass 用量化值，backward pass 用 STE），精度更好但需要训练资源。FP4 场景下，NVIDIA 还提出了 QAD（Quantization-Aware Distillation），用原始模型做 teacher、量化模型做 student，用 KL divergence loss 恢复精度。

**Q: GPTQ 的核心原理是什么？**

GPTQ 基于 Optimal Brain Quantization（OBQ）框架，逐列量化权重矩阵。每量化一列，用 Hessian 矩阵的逆来计算最优的误差补偿量，更新剩余未量化列的权重。关键优化是把逐行处理改成逐列处理（lazy batch updates），使得可以用矩阵乘法加速，把 175B 参数模型的量化时间从几天缩短到几小时。

**Q: Flash Attention 的核心改进是什么？**

Flash Attention 的核心是 tiling + online softmax。传统 attention 需要把完整的 N x N attention matrix 写入 HBM 再读回来做 softmax，IO 成本是 O(N^2)。Flash Attention 把 Q/K/V 分成小块，在 SRAM 中计算局部 attention，用 online softmax（边算边更新 running max 和 denominator）避免存储完整矩阵。最新的 Flash Attention 4 进一步用多项式近似替代 exp，减少共享内存开销，在 Blackwell 上接近 matmul 的理论吞吐。

---

原文链接：https://arxiv.org/abs/2509.23202

llama.cpp NVFP4 PR：https://github.com/ggml-org/llama.cpp/pull/19769
