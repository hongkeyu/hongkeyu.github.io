---
title: "Google TurboQuant: KV Cache 压缩到 3-bit，内存省 6 倍，速度快 8 倍"
date: 2026-03-29T07:30:00+08:00
tags: ["LLM-Inference", "Quantization", "KV-Cache"]
description: "Google Research 在 ICLR 2026 发表 TurboQuant，通过 PolarQuant + QJL 两阶段压缩将 KV Cache 从 FP16 压到 3-4 bit，内存减少 4-6 倍，attention 计算速度提升 8 倍，社区已跟进做 weight quantization 扩展。"
showToc: true
---

## TL;DR

Google Research 在 ICLR 2026 发表 TurboQuant，一种 training-free 的 KV Cache 量化算法，通过 PolarQuant + QJL 两阶段压缩，将 KV Cache 从 FP16 压到 3-4 bit，内存减少 4-6 倍，attention 计算速度提升 8 倍，且下游任务质量几乎无损。社区已经跟进做了 weight quantization 的扩展版本。

---

## 背景：KV Cache 为什么是瓶颈

Transformer 推理时，每生成一个 token 都要计算 attention，需要访问之前所有 token 的 key 和 value 向量。这些向量存在 KV Cache 里，避免重复计算。问题是 KV Cache 的大小随 context length 线性增长，且默认用 FP16 存储。一个 8B 模型在 32K context 下，KV Cache 单独就要吃掉约 4.6 GB 显存。多用户并发或更长 context 时，KV Cache 比模型权重本身更先撑爆显存。

现有方案比如 vLLM 的 FP8 cache、Ollama 的 q4_0/q8_0 cache type，要么压缩不够激进，要么质量损失不可预测。TurboQuant 想在两个维度上都做得更好。

## 核心方法：PolarQuant + QJL 两阶段压缩

TurboQuant 不需要任何训练数据、calibration set 或模型特定配置，纯数学驱动，对任何 Transformer 架构即插即用。

### 第一阶段：PolarQuant（b-1 bits）

传统量化直接对 KV 向量的 Cartesian 坐标做离散化，问题是不同维度的数值分布差异大，需要 per-block normalization constant，既浪费空间又引入额外误差。

PolarQuant 的做法：先对每个 KV 向量施加一个随机正交旋转（random orthogonal rotation）。这个旋转把向量的能量均匀分散到所有坐标上，使得旋转后的每个坐标都服从可预测的统计分布（近似 Beta 或 Gaussian，取决于 head dimension）。因为分布已知，可以提前用 Lloyd-Max 算法计算数学上最优的量化桶，不需要任何数据驱动的 calibration。

然后将坐标从 Cartesian 转换为极坐标（polar form）——只保留半径（数据强度）和方向（数据语义），省掉了传统量化器需要的 per-block normalization constant。Google 给了一个直觉类比：传统编码像说"向东走 3 格，向北走 4 格"，极坐标编码是"朝 37 度方向走 5 格"——信息等价但存储更紧凑。

### 第二阶段：QJL 残差修正（1 bit）

PolarQuant 压完后还有残差误差。QJL（Quantized Johnson-Lindenstrauss）取这个残差，通过一个随机 Gaussian 矩阵做 JL 变换，然后只保留结果的符号位（+1 或 -1）。这个 1-bit sketch 作为 bias correction，使得 attention score 的内积估计在数学上是 unbiased 的。每个坐标只多花 1 bit 的开销。

两个阶段合计：每个坐标 b bits，有可证明的 near-optimal distortion bound，且零 normalization 开销。

## 实测结果与社区扩展

Google 在 Gemma 和 Mistral 开源模型上跑了 long-context benchmark 套件，4-bit TurboQuant 下游任务结果与 FP16 完全一致，KV Cache 内存减少 6 倍。在 H100 上，4-bit TurboQuant 计算 attention logits 比 32-bit unquantized keys 快 8 倍。3-bit 也能用，但 8B 以下模型会出现明显质量下降。

社区反应很快。Reddit r/LocalLLaMA 上已经有人把 TurboQuant 从 KV Cache 量化扩展到了 weight quantization。用 Qwen3.5-4B 测试，4+4 residual（8-bit 总宽度）的 perplexity 与 BF16 baseline 几乎无差（PPL 10.70 vs 10.67），KLD 仅 0.0028。纯 4-bit 则 PPL 升到 11.28。6-bit 的 4+2 residual 方案也很有前景，PPL 甚至略低于 baseline。

实用层面，社区已有 `pip install turboquant` 的 Python 包，三行代码替换 HuggingFace 的 KV Cache。llama.cpp 也有 fork 支持 turbo3 cache type，Apple Silicon Metal kernel 已跑通。vLLM 有 feature request 在推进。Google 官方实现预计 Q2 2026 发布。

## 对边缘部署的意义

这对 Jetson 这类显存有限的设备特别重要。Orin Nano 8GB 跑 7B 模型时，KV Cache 是限制 context length 的主要瓶颈。TurboQuant 能在不换硬件的前提下把可用 context length 推高几倍，或者在相同 context 下腾出显存给更大的模型。移动端同理——不用上云就能跑更好的模型。

## 面试关联知识点

### 1. KV Cache 原理及量化

KV Cache 缓存已计算的 key/value 向量避免重复计算，大小 = layers x 2 x seq_len x hidden_dim x dtype_bytes。传统量化（如 Q4_0）直接对数值做 uniform/non-uniform quantization，需要 per-group scale factor。TurboQuant 的创新在于用随机正交旋转让分布均匀化后再量化，省掉 normalization constant，理论上达到 rate-distortion optimal。面试可以结合 Flash Attention（减少 KV Cache 的 IO 开销）和 GQA（减少 KV head 数量从而减少 Cache 大小）一起回答。

### 2. Johnson-Lindenstrauss 引理与降维

JL 引理：将 n 个高维点随机投影到 O(log n / epsilon^2) 维空间，任意两点间距离以 (1 +/- epsilon) 概率保持。TurboQuant 的 QJL 阶段本质是用 JL 变换做 1-bit 的残差 sketch，保证内积估计 unbiased。这个引理在向量检索（ANN）、推荐系统的 embedding 压缩中也是核心工具。

### 3. 模型量化的精度-效率 trade-off

量化位宽选择不是越低越好。4-bit 是目前的 sweet spot（3B+ 模型质量几乎无损），3-bit 在小模型上退化明显。面试常问：PTQ（Post-Training Quantization）vs QAT（Quantization-Aware Training）区别——TurboQuant 属于 PTQ，零训练开销；QAT 在训练中模拟量化噪声，精度更高但成本也更高。实际部署中通常 weight 用 GPTQ/AWQ（4-bit PTQ），KV Cache 用 FP8 或 TurboQuant，activation 保持 FP16。

---

**参考链接：**

- [论文](https://arxiv.org/abs/2504.19874)
- [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [社区 weight quant 实现](https://github.com/cksac/turboquant-model)
