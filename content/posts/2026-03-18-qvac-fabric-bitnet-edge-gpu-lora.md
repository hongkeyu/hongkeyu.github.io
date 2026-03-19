---
title: "在手机上 LoRA 微调 13B 模型：QVAC Fabric 让 BitNet 跑在边缘 GPU 上"
date: 2026-03-18T07:30:00+08:00
tags: ["edge-ai", "quantization", "lora"]
description: "QVAC Fabric 实现了 BitNet b1.58 模型在消费级 GPU（包括手机）上的推理和 LoRA 微调，13B BitNet 内存占用比 4B Q4 量化模型还少。"
showToc: true
---

## TL;DR

Tether 开源了 QVAC Fabric，世界上第一个支持在异构消费级 GPU（包括手机）上对 BitNet b1.58 模型进行 LoRA 微调的框架。Samsung S25 上 78 分钟微调 1B 模型，iPhone 16 上能跑 13B——比同规模 Q4 量化模型还省内存。

---

## 背景：为什么 BitNet 值得关注

大模型的核心矛盾一直是"参数量 vs 部署成本"。传统量化路线（INT8/INT4/NF4）在精度和效率之间做 trade-off，但 BitNet b1.58 走了一条更激进的路：权重只有三个值 {-1, 0, 1}，即 1.58-bit 表示。这不是 post-training quantization，而是从训练阶段就设计好的极端低比特架构。

Microsoft 在 2024 年发布了 BitNet b1.58 论文，随后开源了 bitnet.cpp 实现 CPU 上的无损推理。但问题是：CPU 推理太慢，而 GPU 端一直没有完整的 BitNet 支持——直到这周 QVAC Fabric 的发布。

## 核心贡献：GPU 上的 BitNet 推理 + 微调

QVAC Fabric 做了几件关键的事：

第一，实现了 BitNet 的 GPU 推理后端。他们在 llama.cpp 基础上扩展了 Vulkan 后端，编写了专门的 GPU shader 来处理三值权重的解码。这意味着 BitNet 模型不再只能跑在 CPU 上。实测结果：在旗舰手机的 GPU 上，推理速度比 CPU 快 2.1 到 11.3 倍。

第二，实现了世界首个 BitNet LoRA 微调的 GPU 支持。基座的 1.58-bit 权重保持冻结，LoRA adapter 的权重是 FP16。使用 AdamW 优化器，线性学习率衰减，512 token 上下文窗口。在 Samsung S25 上微调 1B 模型（297 条 PubMedQA 数据，约 18K tokens），GPU 耗时 78 分钟；125M 模型只需 10 分钟。

第三，真正的跨平台。通过 Vulkan 和 Metal 后端，框架支持 AMD、Intel、NVIDIA 桌面 GPU，以及 Adreno（高通）、Mali（ARM）、Apple Bionic GPU。不是"理论支持"，是他们实际在 Samsung S25、Pixel 9、iPhone 16 上都跑了 benchmark。

## 内存优势：13B BitNet < 4B Q4 量化模型

这是最值得记住的数字：BitNet-13B（TQ1_0 格式）的 VRAM 占用是 2789 MB，而 Qwen3-4B 的 Q4 量化版需要更多内存。也就是说，一个 13B 参数的 BitNet 模型，内存消耗比 4B 的 Q4 模型还少 29%。BitNet-1B 比 Gemma-3-1B（FP16）省 77.8% VRAM，比 Qwen3-0.6B（FP16）省 65.6%。

这个数字对边缘部署的意义是直接的：以前在手机或 Jetson 上只能勉强跑 4B Q4 模型，现在可以跑 13B BitNet。参数量翻了三倍多，内存反而更少。

## 技术细节：Vulkan shader 里的三值解码

传统量化格式（GGUF 的 Q4_K_M 等）在 GPU 上需要 dequantize 操作，把低比特权重还原成 FP16 再做矩阵乘法。BitNet 的三值表示更极端——权重只有 {-1, 0, 1}，矩阵乘法退化为加减法和 mask 操作，完全不需要浮点乘法。

QVAC 的 Vulkan shader 在 GPU 端实时解码 TQ1_0/TQ2_0 格式的打包权重，同时保持与 CPU 实现 bit-exact 等价。他们还实现了 dynamic tiling 来适配不同 GPU 架构的 workgroup size，这是跨平台性能一致性的关键。

## 对 Jetson 用户的启示

虽然文章没有直接测试 Jetson，但 Vulkan 后端天然支持 NVIDIA GPU。Jetson Orin Nano 的 1024-core NVIDIA GPU + 8GB 统一内存，理论上可以跑 BitNet-13B（2.8GB VRAM），甚至还有余量做 LoRA 微调。相比之下，用 Q4 量化跑同等参数量的模型在 8GB 机器上基本不可能。

这可能是边缘 LLM 部署格局的一个拐点：不是"如何把大模型压小"，而是"从头训练一个极低比特的大模型"。

## 延伸：GTC 2026 的 inference 叙事

巧合的是，Jensen Huang 在本周 GTC 2026 keynote 里宣布 AI 已经到达"inference 的 inflection point"。他的论点是：AI 现在能做生产力工作了（agentic AI），所以 inference 需求会爆发式增长。QVAC Fabric 这类框架正好踩在这个趋势上——不是所有 inference 都需要 H100 集群，边缘设备上的高效推理同样是巨大市场。

## 面试关联知识点

### 1. LoRA 原理（低秩分解，为什么低秩可行）

LoRA 的核心假设是微调时的权重更新矩阵 deltaW 具有低秩结构，可以分解为两个小矩阵 A(d x r) 和 B(r x d) 的乘积，其中 r 远小于 d。原因：微调通常只需要调整模型在某个子空间的表达能力，大部分参数方向不需要变化。在 BitNet 场景下，基座权重是 1.58-bit 冻结的，LoRA adapter 是 FP16，训练只更新这部分低秩参数——这也是为什么手机上能微调 13B 模型。

### 2. 模型量化：PTQ vs QAT vs Architecture-level Quantization

传统量化分两类：PTQ（Post-Training Quantization，训练后量化，如 GPTQ/AWQ/GGUF Q4_K_M）和 QAT（Quantization-Aware Training，训练中量化）。BitNet b1.58 走的是第三条路——architecture-level quantization，从模型架构层面就设计为低比特。BitLinear 层替换标准 Linear 层，权重在训练时就是三值的。面试时可以把这三者做对比：PTQ 最简单但精度损失最大；QAT 需要重训练但精度更好；architecture-level 精度最优但需要从头预训练。

### 3. KV Cache 原理及 KV Cache Quantization

BitNet 模型虽然权重极小，但推理时 KV Cache 仍然是 FP16（因为 attention score 需要高精度）。在 512 token 上下文、13B 模型下，KV Cache 可能占到总 VRAM 的 30-50%。这也是为什么 QVAC 的 benchmark 选择 512 token 上下文——更长的上下文会让 KV Cache 成为内存瓶颈。面试常问：KV Cache 存的是什么（每层的 K 和 V 矩阵），为什么能加速推理（避免重复计算已生成 token 的 attention），以及 GQA 如何减少 KV Cache 大小（多个 query head 共享 KV head）。

---

原文链接：<https://huggingface.co/blog/qvac/fabric-llm-finetune-bitnet>

GitHub：<https://github.com/tetherto/qvac-fabric-llm.cpp>
