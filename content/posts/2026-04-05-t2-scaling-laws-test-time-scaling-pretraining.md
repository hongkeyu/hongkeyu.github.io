---
title: "T² Scaling Laws: 当 Test-Time Scaling 改写预训练的最优解"
date: 2026-04-05T07:30:00+08:00
tags: [llm, scaling-laws, test-time-compute]
description: "T² scaling laws 证明：如果部署时会做 repeated sampling，训练时更优的策略往往不是 Chinchilla 式的大模型，而是更小但训练更久的 overtrained 模型。"
showToc: true
---

**TL;DR:** 威斯康星大学 + 斯坦福的新论文提出 Train-to-Test (T²) scaling laws，首次将预训练和推理时的 repeated sampling 统一建模。结论是：如果你知道模型会在推理时做 test-time scaling，那预训练时就应该训更小、过训练更多的模型——这和 Chinchilla 的建议完全相反。

---

## 背景：两套 Scaling Law 各管各的

过去几年 LLM 领域有两条互不相干的 scaling law 线索：

一条是预训练侧的 Chinchilla scaling（Hoffmann et al., 2022），核心结论是模型参数 N 和训练 token 数 D 应该同比例增长，最优比大约 N:D = 1:20。这条定律指导了 Chinchilla、LLaMA 等模型的训练配置。

另一条是推理侧的 test-time scaling，核心方法是 repeated sampling（也叫 pass@k）：对同一个问题生成 k 个独立答案，只要有一个对就算对。最近的研究（Brown et al., 2025）发现，小模型 + 大量采样有时候能打败大模型的单次推理。

问题是：这两条线从来没连起来过。Chinchilla 在设计最优训练配置时根本没考虑推理成本，而 test-time scaling 的研究把预训练好的模型当作既定事实，不去讨论“如果知道要做 repeated sampling，模型应该怎么训”。

## T² 的核心思路：统一优化 N、D、k

这篇论文把预训练和推理的 compute 放进同一个优化框架。总预算分两部分：

- 训练预算 C_train ≈ 6ND（标准 FLOPs 估计）
- 推理预算 C_inf ≈ 2Nk（每次推理的 FLOPs × 采样次数）

优化目标是在总预算约束下，找到最优的 (N*, D*, k*) 组合，使得 pass@k 最大化。

作者提出了两种建模方式：

第一种是对 loss 建模。把 Chinchilla 的损失函数扩展一个 k 的 power law 项：L(N, D, k) = E + A/N^α + B/D^β + G/k^γ。当 k=1 时退化为标准 Chinchilla。这个建模的理论基础是 pass@k 的 negative log 在 Beta 分布假设下呈 power law。

第二种是对准确率建模。先用 Chinchilla 预测 per-sample 正确率，再通过 pass@k 公式 1-(1-p)^k 计算多次采样后的准确率。

两种方法的结论高度一致：最优配置大幅偏向 overtraining 区域——模型应该比 Chinchilla 建议的更小，但训练得更久。

## 实验：100+ 模型，8 个任务

作者在 Porian et al. (2024) 的 Chinchilla scaling 实验基础上扩展，训练了超过 100 个模型，覆盖 12 个 compute level、三个数量级的计算量。评估任务涵盖知识、推理和语言理解。

核心发现：

1. 当推理预算固定时，T² 一致推荐训练更小、过训练程度更高的模型。这些模型单次推理弱，但推理便宜，可以采样更多次，总体表现反而更好。

2. 按 T² 预测的最优配置从头训练 overtraining 模型，结果确实一致优于 Chinchilla 最优配置的模型。这说明 T² 不仅是理论上的，外推到实际训练也成立。

3. 这个结论在 SFT post-training 之后依然成立。也就是说，不管后面怎么微调对齐，预训练阶段的 overtraining 优势不会被抹掉。

## 为什么这篇重要

这篇论文解释了一个业界已经在做但缺乏理论基础的事情：LLaMA 3 系列、Qwen 2.5、Gemma 2 这些开源模型家族，小尺寸版本全都远超 Chinchilla optimal 的训练量。比如 LLaMA 3 8B 用了 15T token 训练，远超 Chinchilla 建议的约 160B token。业界的直觉是“小模型多训一些推理时更划算”，T² 终于给出了严格的数学证明。

对边缘部署的启示更直接：在 Jetson 这类设备上，模型大小受硬件限制，但 repeated sampling 的成本相对可控。T² 的框架可以帮助量化“在固定 VRAM 下，用多小的模型 + 多少次采样”能达到最优性价比。

更深层的意义在于，它把“训练”和“部署”的决策统一了。以前是两拨人各自优化各自的，现在有了一个联合优化框架。这对从事 MLOps 和模型选型的工程师来说非常实用。

原文链接: <https://arxiv.org/abs/2604.01411>

---

## 面试关联知识点

### 1. Scaling Laws（训练侧）

Chinchilla scaling law 核心结论是 N 和 D 应同比例增长（N ∝ C^0.5, D ∝ C^0.5）。实际工程中的 overtraining（D/N 远大于 20）牺牲训练 FLOPs 效率，但降低推理成本。面试追问点：为什么 overtraining 不会导致 overfitting？答：LLM 预训练数据极大且只训一个 epoch，不存在传统意义上的过拟合，overtraining 这里指的是超过 compute-optimal 的 token 数。

### 2. Test-Time Compute / Inference Scaling

pass@k = 1-(1-p)^k，k 次独立采样中至少一次正确的概率。这是 test-time scaling 的基础公式。面试常见变体：如何高效估计 pass@k（unbiased estimator），以及 pass@k 和 majority voting 的区别（pass@k 是上界，majority voting 更实际）。

### 3. KV Cache 与推理成本

T² 的推理成本建模用 2Nk FLOPs，但实际部署中 KV Cache 让 repeated sampling 的边际成本低于 full forward pass（prefill 共享、只需独立 decode）。面试考点：KV Cache 在 prefill 和 decode 阶段的作用差异，以及为什么 Speculative Decoding 可以进一步降低采样成本。
