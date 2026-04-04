---
title: "TurboQuant: Google 的 KV Cache 量化方案，把推理显存砍掉 4-6 倍"
date: 2026-04-03T07:30:00+08:00
tags: ["LLM推理优化", "量化", "KV-Cache"]
description: "TurboQuant 用随机正交旋转 + 预计算码本将 KV Cache 压缩到 3-4 bit，无需训练、无需校准，即插即用，社区已有 pip 包和 llama.cpp 实现。"
showToc: true
---

TL;DR: Google Research + NYU 提出的 TurboQuant 用随机正交旋转 + 预计算码本，将 KV cache 压缩到 3-4 bit，无需训练、无需校准，即插即用。ICLR 2026 论文，社区已有 pip 包和 llama.cpp 实现。

---

## 为什么 KV Cache 是瓶颈

Transformer 推理时，每生成一个 token 都要存储之前所有 token 的 Key 和 Value 向量（即 KV cache），避免重复计算。问题在于 KV cache 随 context length 线性增长，且默认以 FP16 存储。一个 8B 模型在 32K context 下，仅 KV cache 就吃掉约 4.6GB 显存。多用户并发或更长上下文时，KV cache 比模型权重本身更先把显存撑爆。

现有推理引擎对 KV cache 的量化支持很有限：vLLM 主分支只支持 FP8/INT8（8-bit），llama.cpp 有一些 block-based GGML 量化到 4-bit，但都不是专门为 KV cache 设计的方案。文献里有 100+ 种更好的方法，但集成到高度优化的推理引擎中工程难度大，所以主流框架一直没跟进。

## TurboQuant 核心原理

TurboQuant 分两个阶段：

### Stage 1: PolarQuant（主体压缩）

核心思路很巧妙：对每个 KV 向量先施加一个固定的随机正交旋转（random orthogonal rotation）。旋转本身不改变向量的几何关系（正交变换保范数、保内积），但旋转后向量的各维度分布变得非常均匀，近似独立的高斯分布。这意味着一个预计算的固定标量码本（Lloyd-Max 算法求最优量化桶）就能在所有 KV 向量上通用，不需要针对每个模型或数据集做校准。

量化流程：生成一个旋转矩阵 -> 预计算码本 -> 旋转向量 -> 存储码本索引和范数 -> 反量化时查表 + 逆旋转。极其简洁。

论文还用了 outlier-aware 混合精度：对少量 outlier channel 分配更多 bit。比如 head dim=128 时，32 个 outlier channel 给 3 bit，96 个普通 channel 给 2 bit，平均 2.5 bit/channel。

### Stage 2: QJL 残差修正（1 bit）

Stage 1 的量化误差会让 attention score（依赖内积）产生偏差。Stage 2 对残差做 Johnson-Lindenstrauss 随机投影，只保留符号位（+1/-1），形成 1-bit 的 sketch。这额外 1 bit 可以让内积估计变成无偏的。不过社区实验发现，直接把 QJL 修正加回重建的 cache 向量效果反而不好，单独用 MSE 阶段（不加 Stage 2）在 drop-in 场景下更稳定。QJL 更适合你能自己写 attention kernel、直接消费两部分表示的情况。

## 实际效果与工程落地

Google 报告 KV cache 内存至少降 6 倍，优化后的 4-bit TurboQuant 在 H100 上 attention logit 计算速度达到 8 倍加速（vs 未量化 FP32 key）。

实用建议（来自社区实验）：

- **4-bit 是甜区：** 3B+ 模型质量几乎无损，和 FP16 无法区分
- **3-bit 可以更激进地压缩，** 但 8B 以下模型开始出现质量下降
- **Value 比 Key 更敏感：** 2-bit value 的 cosine similarity 降到 0.94，4-bit 能保持 0.997。调 bit 分配时优先给 value 更多 bit
- **短 context（<1K token）收益不大，** 旋转 + 量化的开销可能抵消。4K+ token 后收益明显
- 实践中保留最近 128-256 个 token 为 FP16 全精度，只压缩更早的 token

### 现在就能用

`pip install turboquant`，三行代码替换 HuggingFace 的 KV cache。也有 llama.cpp 的 fork（turboquant_plus）在 Apple Silicon 上跑通了 Metal GPU kernel，prefill 吞吐量接近 q8_0 水平但 KV cache 压缩 4.6 倍。vLLM 也有 open feature request 在推进中。Google 官方实现预计 Q2 2026。

### 对边缘设备的意义

对 Jetson 这类边缘设备，KV cache 是跑长 context LLM 时最致命的显存瓶颈。TurboQuant 训练无关、架构无关的特性让它非常适合嵌入边缘推理 pipeline。

## 参考链接

- 论文: https://arxiv.org/pdf/2504.19874
- Google 博客: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- Kaitchup 分析: https://kaitchup.substack.com/p/turboquant-finally-fast-and-widely
