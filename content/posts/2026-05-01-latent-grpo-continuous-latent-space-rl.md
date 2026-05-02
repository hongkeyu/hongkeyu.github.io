---
title: "Latent-GRPO：在连续潜空间做 GRPO，推理链压缩 3-4x 且效果更好"
date: 2026-05-01T07:30:00-04:00
tags: [reinforcement-learning, GRPO, latent-reasoning, chain-of-thought, inference-efficiency]
description: "中科院团队诊断出 GRPO 在连续潜空间的三个耦合瓶颈，提出 Latent-GRPO 通过无效样本遮蔽、单侧噪声和最优首 token 选择，以 3-4x 更短的推理链超越标准 GRPO。"
showToc: true
---

## 背景：Latent Reasoning 的 RL 困境

Reasoning model 的核心范式是 Chain of Thought——让模型用自然语言写出推理过程，再给出答案。问题在于，长 CoT 带来的计算冗余巨大：一个 GSM8K 题目的显式 CoT 可能写 100 多个 token，但真正有信息量的推理步骤只有几步。

Latent reasoning 的思路是把中间推理步骤压缩成连续向量表示（latent token），而非离散的自然语言 token。具体做法是在每一步不采样离散 token，而是对 vocabulary 上的 top-K 概率分布做加权平均，得到连续 embedding 向量作为下一步输入。推理链大幅缩短，信息密度更高。

之前的 Latent-SFT 已证明这种方法可以匹配甚至超过显式 CoT 的 SFT 效果。自然的下一步是用 RL（具体是 GRPO）来进一步提升——但这里遇到了根本性的困难。

## 三个耦合瓶颈

作者诊断出 GRPO 在潜空间失败的三个根因：

### 瓶颈一：潜空间流形缺失

从零开始的 RL 无法自发学会 latent reasoning——模型必须先通过 SFT 获得有效的潜空间结构。即使有了 SFT 初始化，无约束的 Gumbel 噪声探索仍会把 rollout 推离有效流形，产生混乱的、不终止的生成。

### 瓶颈二：探索-优化错位

这是最致命的问题。标准 GRPO 用 Gumbel 噪声模拟离散采样来做探索，但在连续空间里，双侧 Gumbel 噪声会造成梯度方向与 advantage 符号不一致：即使某条轨迹的 advantage 为正，噪声扰动为负的分量的概率反而会被压低。

从梯度层面看，更新方向正比于 $\hat{A} \cdot (1 - \exp(-\Delta_i))$，当扰动余量 $\Delta_i$ 为负时，整个乘积的符号会翻转，直接破坏 RL 的优化语义。

### 瓶颈三：潜空间混合态的非封闭性

标准 GRPO 中，同一组采样里多条正确轨迹同时获得正 advantage 信号是无害的，因为离散 token 的 categorical 分布天然支持多模态。但在连续空间里，对多条正确潜在路径做梯度更新相当于把多个有效状态做加权平均——而这个平均态可能落在有效流形之外。这个问题在第一个推理步最严重，因为所有轨迹共享同一个 prompt prefix。

## Latent-GRPO 的三个设计

### Invalid Sample Advantage Masking

如果一条轨迹在最大长度内没有生成 EOS token（即"跑飞了"），就把它的 advantage 设为零，不参与梯度更新。Group baseline 也只在有效轨迹上计算。这防止了 off-manifold 的垃圾轨迹污染优化。

### One-sided Noise Sampling

核心改动：把 Gumbel 噪声限制为严格正值。具体是对标准 Gumbel 噪声做 clamp-and-shift 变换，使扰动余量恒为正。这保证了对正 advantage 轨迹，所有分量的更新方向都是增大概率；负 advantage 则反之。

为了应对 PPO 多 epoch 更新中目标被超越的情况，还加了一个条件式 Straight-Through Estimator，在 forward pass 保持原值、backward 翻转梯度。消融实验显示这是三个设计中最关键的——去掉它会导致训练崩溃。

### Optimal Correct Path First Token Selection

在同一组里有多条正确轨迹时，只在第一个 latent step ($t=1$) 保留"最优路径"（按 surrogate log-probability 最高的那条），其他正确路径在 $t=1$ 的 advantage 被遮蔽。$t>1$ 的步骤照常更新，因为此时各轨迹的 prefix 已经分化，不再有混合态问题。

## 实验结果

| 模型 | 任务 | Latent-GRPO vs Latent-SFT | Latent-GRPO vs 显式 GRPO | 推理链压缩比 |
|------|------|---------------------------|--------------------------|-------------|
| LLaMA-3.2-1B | GSM8K-Aug 等低难度 | +7.86 Pass@1 | — | 4.44x |
| Qwen2.5-Math-7B | Math500/AIME24/AIME25/GPQA | +14.77 Pass@1 | +4.27 Pass@1 | 3.31x |

在 Gumbel sampling 下的 pass@k 表现也更强：AIME24 上 pass@64 达 36.7（显式 GRPO 仅 23.3）。

代码已开源：[GitHub - Latent-GRPO](https://github.com/DJC-GO-SOLO/Latent-GRPO)

## 为什么重要

这篇文章的价值不只在于"又一个 GRPO 变体"：

1. **系统性诊断了 RL 在连续动作空间上的失败模式**——对所有尝试在连续表示上做 policy optimization 的工作都有参考意义
2. **Latent reasoning + RL 的可行性**——如果 latent reasoning 在 RL 阶段也能稳定工作，就意味着可以用更短的推理链达到更好的效果
3. **3-4x 推理链压缩**——KV cache 占用、prefill 时间、decode 步数都相应缩减，对推理成本的控制是根本性的

原文链接：[arXiv:2604.27998](https://arxiv.org/abs/2604.27998)

## 面试关联知识点

### GRPO 的核心机制

对同一个 query 采样一组 $G$ 个输出，计算 rule-based reward，用组内均值和标准差做归一化得到 advantage，然后用 PPO-style clipped surrogate objective 更新策略。相比 PPO 不需要单独训练 value network，相比 DPO 可以用 rule-based reward 不需要偏好数据。DeepSeek-R1 的核心训练算法。

### Chain of Thought 的计算冗余和缓解方向

显式 CoT 把推理过程展开成自然语言，token 数远超实际推理所需的信息量，导致 inference latency 和 KV cache 开销线性增长。缓解方向包括：

- **Latent reasoning**：压缩为连续表示
- **Early exit**：中间层提前输出
- **CoT distillation**：用短链蒸馏长链的推理能力
- **Speculative decoding**：用小模型加速 decode

### 为什么 RL 在连续动作空间比离散空间更难

离散空间的 categorical 分布天然支持多模态（softmax 可以同时给多个 token 高概率），但连续空间的加权平均会导致模式坍缩（多个有效状态的平均态可能无效）。此外，连续空间的探索噪声会引入梯度方向与 advantage 符号不一致的问题，需要特殊处理（如本文的单侧噪声）。
