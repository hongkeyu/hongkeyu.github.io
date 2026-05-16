---
title: "SDAR: 用 Sigmoid Gate 让 Agent RL 训练学会选择性蒸馏"
date: 2026-05-15T07:30:00-04:00
tags: [reinforcement-learning, knowledge-distillation, LLM-agents]
description: "浙大与美团提出 SDAR，在 GRPO 上加 sigmoid 门控 self-distillation，让 multi-turn agent 训练选择性从 teacher 学习，ALFWorld +9.4%，WebShop +10.2%。"
showToc: true
---

## 背景

用 RL（特别是 GRPO）做 LLM agent 的 post-training 已经是标准范式，但问题很明显：trajectory 级别的 reward 信号太粗了。一个 multi-turn 交互可能跑了几十步，最后只拿到一个"成功/失败"，中间哪步对哪步错，模型根本分不清。

On-Policy Self-Distillation (OPSD) 是一种补救思路：用一个能看到特权信息（比如参考答案、技能模板）的 teacher 分支，在 token 级别给 student 提供密集指导。这在 single-turn 推理任务上效果不错，但搬到 multi-turn agent 场景就出了两个致命问题。

## 核心问题

**Multi-turn 累积不稳定。** 一旦 student 在某一轮偏离了 teacher 的轨迹，后续每一轮的 teacher 指导都建立在错误的上下文上，KL 散度会指数级膨胀，最终训练崩溃。论文的实验图非常直观——纯 OPSD 在 Search-QA 上直接掉到接近零。

**特权信息的不对称可信度。** Teacher 靠技能检索获得额外上下文，但检索质量参差不齐。当检索到的 skill 和当前任务不匹配时，teacher 给出的"拒绝"信号未必可靠——可能只是 teacher 自己被错误的 skill 误导了。

## SDAR 的设计

SDAR 的核心思路是把 OPSD 降级为"辅助目标"，RL（GRPO）始终是主优化目标不受干扰。具体做法：

对每个 token 位置 $t$，计算 Teacher-Student log-probability gap：

$$\Delta_t = \log \pi_T(y_t \mid s_t^+) - \log \pi_\theta(y_t \mid s_t)$$

正值意味着 teacher 比 student 更看好这个 token，负值意味着 teacher 反而不如 student 有信心。

然后用一个 sigmoid 门控 $g_t = \sigma(\beta \cdot \Delta_t)$ 来调制蒸馏强度：

| $\Delta_t$ 方向 | Gate 值 | 效果 |
|---|---|---|
| $> 0$（teacher 更好） | 接近 1 | 强蒸馏 |
| $< 0$（teacher 不靠谱） | 接近 0 | 自动衰减 |

$\beta$ 控制过渡的锐利度，实验中 $\beta=5$ 最优。

最终损失函数：$\mathcal{L} = \mathcal{L}_{\text{GRPO}} + \lambda \cdot \mathcal{L}_{\text{SDAR}}$，其中 $\lambda=0.01$。Gate 是 detached 的，梯度只通过 student 的 log probability 流动，不会反向污染 RL 的 advantage 估计。

### 门控策略对比

论文对比了三种门控策略——基于 entropy（student 不确定的地方多学）、基于 gap（teacher 更好的地方多学）、以及两者的 soft-OR 组合。结论是纯 gap gating 效果最好，因为它最直接地衡量了"teacher 在这个 token 上是否真的比 student 强"。

## 实验结果

在 Qwen2.5 和 Qwen3 系列（1.7B/3B/7B）上，跨三个 benchmark 做了全面评测：

| Benchmark | SDAR | GRPO | 提升 |
|---|---|---|---|
| ALFWorld（文本世界家务） | 84.4% | 75.0% | +9.4% |
| WebShop（网页购物） | — | — | +10.2% |
| Search-QA（搜索增强问答） | — | — | +7.0% |

关键发现：Skill-GRPO（训练时注入 skill、推理时也用）在测试时去掉 skill 会暴跌（80.5% → 60.2%），甚至不如 vanilla GRPO，说明它只是学会了依赖 skill 而没有内化知识。SDAR 推理时完全不需要 skill 输入，却比带 skill 的 Skill-GRPO 还好。

对 skill 检索质量的鲁棒性测试也很有说服力：即使用完全随机的 skill 检索，SDAR 仍然比 GRPO 有 +1.9% 的提升，因为 sigmoid gate 天然地过滤掉了不靠谱的 teacher 信号。

## 为什么重要

这篇论文解决的是一个非常实际的工程问题：怎么在 agent RL 训练中充分利用 dense supervision 而不引入不稳定性。之前要么只用稀疏 reward（GRPO），要么加 dense signal 但冒着训练崩溃的风险。SDAR 用一个极其简单的 sigmoid gate 就实现了"选择性蒸馏"，工程复杂度几乎为零——只需要在 GRPO 的 loss 上加一项，不改训练流程，不改推理架构。

更深层的 insight 是：在 multi-turn agent 场景下，teacher 不一定比 student 强。这和传统知识蒸馏"teacher 总是对的"的假设完全不同。SDAR 的 gap gating 本质上是在做 token 级别的"可信度评估"，只在 teacher 确实有优势的地方学习。

## 延伸

这个方向和 privileged information learning 有深层联系——推理时不可用的信息如何在训练时发挥最大价值。SDAR 的 gating 机制也可以推广到其他场景，比如 RAG 训练中检索结果质量不稳定时的 selective distillation。

- 原文：[arXiv 2605.15155](https://arxiv.org/abs/2605.15155)
- 代码：[ZJU-REAL/SDAR](https://github.com/ZJU-REAL/SDAR)

## 面试关联知识点

### GRPO 的核心机制

GRPO 对每个 prompt 采样一组 response，用组内的相对 reward 计算 advantage（无需 critic 网络），配合 importance sampling ratio 和 KL 惩罚做策略更新。优点是不需要 value model，比 PPO 简单很多；缺点是 trajectory 级 reward 对 long-horizon multi-turn 场景监督信号太稀疏。

### Reverse KL vs Forward KL

Forward KL $D(p_T \| q_S)$ 是 mean-seeking 的，student 试图覆盖 teacher 的所有 mode；Reverse KL $D(q_S \| p_T)$ 是 mode-seeking 的，student 集中在 teacher 概率高的区域。SDAR 用 reverse KL 正是因为 teacher 信号"部分不可靠"，mode-seeking 让 student 只跟踪 teacher 真正自信的部分。

### Speculative Decoding 与 Self-Distillation 的联系

两者都涉及大小模型的 token 级协作。Speculative Decoding 用小模型提 draft、大模型验证，加速推理；Self-Distillation 用大/特权模型做 teacher 指导小模型训练。SDAR 的 gap gating 思路理论上也可以用于 speculative decoding 的 acceptance 策略——在 draft model 和 target model 分歧大的位置做更细粒度的处理。
