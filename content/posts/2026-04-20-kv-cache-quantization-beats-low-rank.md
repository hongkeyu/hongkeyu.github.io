---
title: "KV Cache 压缩：量化全面碾压低秩分解，Softmax 几何结构是关键"
date: 2026-04-20T07:30:00-04:00
tags: [LLM推理, KV-Cache, 量化, 低秩分解, Transformer]
description: "INT4 量化 KV Cache 只损失 +0.18 PPL 就能压缩 75%，而同等存储的 rank-32 低秩分解直接崩盘。原因藏在 softmax 的离散路由几何里。"
showToc: true
---

## 背景

KV Cache 是 Transformer 推理的核心瓶颈。对于一个 L 层、H 头、维度 d、序列长度 T 的模型，KV Cache 占用 2LHdT 个元素。长上下文场景下（128K+ tokens），这个开销可以轻松吃掉几十 GB 显存，直接决定了你能跑多大的 batch size 和多长的上下文。

压缩 KV Cache 主要有两条路：

- **低秩分解（Rank Reduction）**：把 K/V 向量投影到更少的维度，比如从 128 维降到 32 维。代表工作包括 SVD-based pruning、KQ-SVD 等。
- **量化（Quantization）**：保留所有维度，但降低每个值的精度，比如从 FP16 降到 INT4。

此前几乎没有人在相同存储预算下做过严格的 head-to-head 对比。

## 核心发现

Samuel Salfati（fraQtl AI Research）在 [arXiv:2604.11501](https://arxiv.org/abs/2604.11501) 中，用 5 个模型（GPT-2 124M 到 Qwen 14B，涵盖 MHA 和 GQA 架构），在 WikiText-2 和 C4 两个数据集上做了系统对比。

**量化在所有设定下都赢了低秩分解，差距从 4 PPL 到 364 PPL 不等。**

| 模型 | 方法 | 压缩率 | PPL 变化 |
|------|------|--------|----------|
| Mistral 7B | Joint K+V INT4 量化 | 75% | +0.18 |
| Mistral 7B | Rank-32 低秩分解 | 75% | +34.77 |
| LAMBADA | INT4 量化 | 75% | +0.23 PPL，准确率几乎不掉 |
| LAMBADA | Rank-32 低秩分解 | 75% | 准确率崩到 0.4% |

即使把低秩分解和量化混合使用（先降维再量化），也打不过直接全维度量化。而且差距随着 GQA 的激进程度增大而增大——GQA 模型（Llama 3、Mistral、Qwen2 等主流架构）对维度删除更敏感。

## Softmax 的几何解释：为什么量化赢

论文最有价值的部分是给出了理论解释。

**核心论点：删除维度和添加噪声，在 softmax attention 下造成的损害是质的不同，不是量的不同。**

### 删维度 → 离散故障

Softmax 本质上是一个离散路由器——它决定"attend to 哪个 token"。当你删除 Key 向量的一个维度时，两个原本差距很小的 attention score 可能发生排序翻转（discrete failure），模型直接"看错了 token"。这是一个离散的、不可逆的错误。

### 量化 → 有界模糊

量化只是在每个值上加了有界噪声。对于 b-bit 量化，噪声被限制在一个已知范围内。在绝大多数情况下，这个噪声不足以翻转 attention score 的排序，模型虽然看得"模糊"了一点，但没看错方向。

### 定量差距

在 softmax Fisher metric 下，投影损害（删维度）比量化损害大 **3 × 2^(2b)** 倍。对于 INT4（b=4），删一个维度的损害约等于量化噪声的 **768 倍**。

一个关键的 basis ablation 实验排除了"PCA 基底选得不好"的质疑——在各种基底下，量化质量的波动不超过 0.4 PPL，说明优势来自"保留所有维度"这个结构性选择，而不是某个特定的坐标系。

## 工程意义

### KV Cache 压缩策略有了明确答案

如果你在做 LLM serving（vLLM、TensorRT-LLM、llama.cpp），优先考虑 KV Cache 量化。INT4 KV Cache 是一个几乎免费的 75% 压缩。

### GGUF 量化为什么效果好

GGUF 格式中的 Q4_K_M 等方案本质上就是保留所有维度、降低精度。这篇论文从理论上解释了为什么这条路是对的。

### GQA 架构的隐含代价

GQA 通过共享 K/V heads 来减少 KV Cache 大小，但让每个 KV head 承载更多信息，因此对维度删除更敏感。在 GQA 模型上，量化是比剪枝更安全的二次压缩手段。

### 理论框架可复用

论文提出的 softmax Fisher metric 分析框架可以用来分析其他涉及 attention score 扰动的优化方法，比如 attention head pruning、sparse attention 等。

## 延伸阅读

论文还提出了一个 downstream-optimal compression framework（Theorem 1），统一了 PCA、Fisher-SVD、GPTQ、AWQ 和 KQ-SVD 作为特殊情况，提供了理论上最优的低秩分解基底（在 rank-32 上恢复了 52% 的差距）。虽然低秩分解仍然整体不如量化，但如果必须做低秩分解（比如某些硬件约束下），这个框架给出了最优选择。

**原文链接**: [arXiv:2604.11501](https://arxiv.org/abs/2604.11501)

## 面试关联知识点

### KV Cache 原理及其量化

KV Cache 在 autoregressive decoding 时缓存历史 token 的 Key/Value 向量，避免重复计算。占用空间 = 2 × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes。KV Cache Quantization 将 FP16 值压缩为 INT4/INT8，在几乎不损失精度的前提下减少 50-75% 显存。关键点：量化噪声是有界的，不会翻转 softmax 路由。

### GGUF 格式与 Q4_K_M

GGUF 是 llama.cpp 使用的量化格式。Q4_K_M 表示 4-bit 量化 + K-means 分组 + Medium 质量。量化的核心思路是保留所有维度但降低精度——这篇论文从理论上证明了这比删维度更优。实际部署中，INT4 权重量化 + INT4 KV Cache 量化可以叠加使用，实现极致的显存压缩。

### GQA（Grouped Query Attention）

GQA 让多个 Query head 共享同一组 Key/Value head（比如 Llama 3 用 8 个 KV heads 对应 32 个 Q heads），本身就是一种 KV Cache 压缩（4x）。但副作用是让每个 KV head 承载更多信息，对维度删除更敏感。在 GQA 模型上做进一步 KV Cache 压缩时，量化优于剪枝的优势更加明显。
