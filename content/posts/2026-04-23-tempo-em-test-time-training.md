---
title: "TEMPO: 用 EM 算法让推理模型在测试时持续自我进化"
date: 2026-04-23T07:30:00-04:00
tags: [test-time-training, reinforcement-learning, reasoning]
description: "TEMPO 用 EM 框架交替校准 critic 和优化 policy，解决 TTT 的性能饱和与多样性坍塌，OLMO3-7B 在 AIME 2024 上从 33% 提升到 51%。"
showToc: true
---

## 核心问题

Test-time training (TTT) 让模型在推理阶段继续用未标注的测试题更新参数，突破离线训练的天花板。但现有方法有两个致命问题：

1. **性能快速饱和**——几轮之后就不再提升
2. **多样性坍塌**——模型只会一种解题套路，pass@k 反而下降

根本原因：模型用自己的输出生成 reward 信号做自我训练，随着 policy 更新，reward 信号漂移，形成正反馈循环。

## TEMPO 的 EM 框架

TEMPO 把 TTT 重新理解为 Expectation-Maximization：

### E-Step：Critic 校准

在有标注数据集上周期性重新训练 critic 模型，估计后验分布 P(y|x, Correct)。关键是 **不冻结** critic，每隔几轮就校准一次，跟上 policy 的变化。

### M-Step：Policy 优化

用校准后的 critic 作为 reward 信号，在未标注测试题上用 PPO 更新 policy。因为 critic 有外部监督接地（grounded），梯度信号不会漂移。

论文严格推导了 ELBO，证明现有的 TTRL 和 EMPO 是省略 E-Step 的不完整 EM 变体。重新引入 critic 校准步骤能收紧 ELBO，让模型在更多 test-time compute 下持续提升。

## 实验结果

| 模型 | 基准 | TEMPO 后 | 提升 |
|------|------|----------|------|
| OLMO3-7B | 33.0% (AIME 2024) | 51.1% | +18.1 |
| Qwen3-14B | 42.3% (AIME 2024) | 65.8% | +23.5 |
| OLMO3-7B | BigBenchHard | — | +21.4 |
| OLMO3-7B | AGI Eval | — | +24.5 |

在完全未见过的 AIME 2026 和 OlymMath holdout 上同样显著提升——不是记忆测试题，而是真正增强了推理能力。一个 7B 模型 TTT 后能超过 General-Reasoner-7B 和 MiMo-Zero-RL-7B。

## 关键消融

- **冻结 critic**：前 100 步还能跟上 TEMPO，之后饱和——周期性校准是必需的
- **继续在标注数据上做 supervised PPO**：几乎零提升——模型已在训练分布收敛，新增益只能从未标注测试分布获取

## 面试知识点

### Scaling Inference Compute 的两条路线

| 路线 | 方式 | 是否改变模型 |
|------|------|------------|
| Test-time compute scaling | 更长 CoT、更多 sample、beam search | 否 |
| Test-time training | 推理阶段更新参数 | 是 |

TEMPO 属于后者，且证明 TTT 可以持续 scale。

### EM 算法为什么适合 TTT？

测试题没有标注，response 正确性是隐变量。E-Step 用 critic 近似后验 + 标注数据定期校准，M-Step 用 critic reward 在未标注数据上优化 policy。交替执行保证 reward 信号不漂移。

### Self-training 为什么导致 diversity collapse？

模型强化当前最自信的推理模式 → reward 高估这些模式 → 其他路径被抑制 → pass@k 下降。TEMPO 通过外部校准的 critic 打破正反馈循环。

---

📄 [论文原文](https://arxiv.org/abs/2604.19295) ｜ 💻 [代码](https://github.com/QingyangZhang/TEMPO)
