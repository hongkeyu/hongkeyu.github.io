---
title: "Google TurboQuant: KV Cache 压缩到 3-bit，推理内存降 6 倍"
date: 2026-03-30T07:30:00+08:00
tags: [LLM-Inference, Quantization, KV-Cache]
description: "Google Research 发布 TurboQuant，一种无需重训练的 KV Cache 量化算法，将 KV Cache 压缩到 3-4 bit，内存减少 4-6 倍，精度几乎无损。"
showToc: true
---

**TL;DR:** Google Research 发布 TurboQuant，一种无需重训练的 KV Cache 量化算法，将 KV Cache 压缩到 3-4 bit/元素，内存减少 4-6 倍，精度几乎无损。论文已被 ICLR 2026 接收，4 月底正式报告。

---

## 背景：KV Cache 为什么是推理瓶颈

Transformer 生成文本时，每个 token 的 Key 和 Value 向量都要存到 KV Cache 里，避免重复计算。问题是 KV Cache 随 context length 线性增长，且默认 FP16 精度。一个 8B 模型在 32K context 下，KV Cache 单独就占约 4.6 GB VRAM。多用户并发或更长 context 时，KV Cache 比模型权重本身更先把显存吃光。

现有方案要么压缩不够激进（vLLM 的 FP8），要么精度代价不可预测（Ollama 的 q4_0/q8_0 cache type）。TurboQuant 的目标是两头都做好：压得更狠，质量还不掉。

## 核心方法：两阶段压缩流水线

### Stage 1 — PolarQuant（b-1 bits）

对每个 KV 向量施加随机正交旋转，将向量能量均匀分散到所有坐标上。旋转后每个坐标服从可预测的统计分布（近似 Beta/Gaussian），因此可以用 Lloyd-Max 算法预先计算最优量化 codebook，不需要任何校准数据。然后转换为极坐标表示（radius + angle），省掉传统量化器必需的 per-block normalization 常数。

### Stage 2 — QJL Residual Correction（1 bit）

取 Stage 1 的量化残差，通过随机 Gaussian 矩阵做 Johnson-Lindenstrauss 投影，只保留符号位（+1/-1）。这个 1-bit sketch 充当偏差修正，使得 attention score 的内积估计在数学上无偏。额外开销仅 1 bit/坐标。

两阶段合计 b bits/坐标，有可证明的近最优失真界，且零 normalization 内存开销。

## 关键实验结果

Google 在标准 long-context benchmark 上测试：
- 3-bit KV Cache 量化，无训练、无微调、无可测量精度损失
- Needle-in-a-haystack 全部满分
- H100 上 attention 计算相比 FP32 加速最高 8 倍
- 向量检索场景下 recall 优于使用更大 codebook + 数据集特定调优的 SOTA 方法

## 实践要点

- 4-bit 是甜区：3B+ 模型上与 FP16 几乎无法区分；3-bit 在 8B 以下模型开始有可感知退化
- Value 比 Key 敏感：2-bit value 的 cosine similarity 降到 ~0.94，4-bit 维持 0.997。如果做不均匀分配，给 value 更多 bit
- 短 context（<1K token）收益可忽略，甚至有负开销；4K+ 才开始显著
- 多数实现保留最近 128-256 token 为 FP16，只压缩历史 token（residual window）

## 对边缘设备的意义

一个实际例子：RTX 5090（32 GB）跑 120B 模型，KV Cache 限制导致 context 只能开到 16K。TurboQuant 6 倍压缩理论上可以推到 32K+。对 Jetson 这种更受限的设备，同样的道理——KV Cache 往往比模型权重更先撑爆内存，压缩 KV Cache 比继续量化权重的边际收益可能更大。

另外 TurboQuant 同样适用于向量检索 / ANN 索引压缩，对 RAG pipeline 的 embedding index 构建也有加速效果。

## 现有实现

论文是数学推导，Google 还没有官方代码（预计 Q2 2026），但社区已经跑通了多个独立实现：
- `pip install turboquant`：HuggingFace drop-in replacement，自带 OpenAI 兼容 server
- llama.cpp fork（turboquant_plus）：Apple Silicon + Metal 已跑通，`--cache-type-k turbo3 --cache-type-v turbo3`
- Triton kernel 实现：RTX 4090 上 Gemma 3 4B 模型，2-bit 精度输出与无压缩逐字符一致
- vLLM 有 open feature request 集成中

## 面试关联知识点

### 1. KV Cache 原理及 KV Cache Quantization

KV Cache 存储 decoder 每层每个 attention head 的 K/V 向量，避免 autoregressive 生成时重复计算前序 token 的 attention。大小 = 2 x num_layers x num_heads x head_dim x seq_len x precision_bytes。量化方式分 post-training（TurboQuant、FP8）和 training-aware。TurboQuant 的核心创新是用随机正交旋转使分布可预测，从而用固定 codebook 替代 calibration-based 量化。

### 2. GQA（Grouped Query Attention）与 KV Cache 的关系

GQA 让多个 query head 共享同一组 K/V head，直接从架构层面减少 KV Cache 大小（比如 Llama 3 用 8 个 KV head 对应 32 个 Q head，KV Cache 减少 4 倍）。TurboQuant 和 GQA 是正交优化，可叠加使用。

### 3. Speculative Decoding 与 KV Cache 的互动

投机解码用小模型 draft 多个 token，大模型一次 verify。verify 阶段需要一次性处理 draft 长度的 KV Cache 更新。KV Cache 压缩对 speculative decoding 有利——更小的 cache 意味着 verify 步骤的内存压力更低，可以用更长的 draft 窗口。

---

原文：
- Google Blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- Paper: https://arxiv.org/abs/2504.19874
- DEV.to 开发者指南: https://dev.to/arshtechpro/turboquant-what-developers-need-to-know-about-googles-kv-cache-compression-eeg
