---
title: "CausalRM: 用因果推断修复 RLHF 的 Reward Model"
date: 2026-03-25T07:30:00+08:00
tags: [RLHF, 因果推断, reward-model]
description: "CausalRM 提出因果理论框架，用噪声校正和 propensity reweighting 从有偏观测反馈中学习无偏 reward signal"
showToc: true
---

当 RLHF 遇到真实世界的用户反馈，reward model 学到的可能不是偏好，而是偏见。CausalRM 提出了一套因果理论框架，从有噪声、有偏差的观测反馈中学习无偏的 reward signal。

---

## 背景：RLHF 的数据瓶颈

RLHF 三阶段（SFT -> Reward Model -> PPO/DPO）的核心假设是：reward model 能准确捕捉人类偏好。但现有做法严重依赖实验性标注数据——雇标注员在受控环境下做 pairwise comparison。这套流程贵、慢、规模受限。

一个自然的想法是：用户在日常使用中产生的观测反馈（observational feedback）——点赞、复制、收藏——能不能替代？数据量大得多，成本几乎为零。

问题是，观测反馈有两个致命缺陷，都是经典的因果推断问题。

## 挑战一：标注噪声（Annotation Noise）

用户的点赞/点踩不等于真实偏好。手滑、误读、注意力不集中都会引入噪声。如果直接用这些 noisy label 训练 reward model，学到的是「用户实际点了什么」而不是「用户真正偏好什么」。

CausalRM 的解法：显式建模标注错误的生成过程（annotation error generation process），引入 noise-aware surrogate loss。核心思路是，假设真实偏好标签 y* 经过一个噪声通道变成观测标签 y，如果能估计这个噪声通道的翻转概率，就能构造一个替代损失函数，在数学上等价于在无噪声条件下直接优化原始损失。

这不是新想法——noise-robust learning 在分类任务中有大量文献（Natarajan et al. 2013, Patrini et al. 2017）。CausalRM 的贡献在于把这套理论严格地适配到了 preference learning 的 pairwise 设定中，并给出了可证明的等价性保证。

## 挑战二：选择偏差（Selection Bias）

这是更有意思的问题。用户不是对所有 response 都给反馈——他们倾向于对感受强烈的回答做出反应（特别好或特别差），对「还行」的回答直接跳过。这造成了训练数据和推理数据之间的分布偏移（distribution shift）：训练时模型只看到了极端样本，推理时要对所有样本打分。

这本质上是因果推断中经典的 selection bias / missing not at random 问题。CausalRM 用 propensity score reweighting 来解决——估计每个 response 被用户反馈的概率（propensity score），然后用这个概率的倒数对训练样本加权。直觉上：如果某个 response 几乎所有人都会反馈（propensity 高），权重小；如果某类 response 很少有人反馈但偶尔有（propensity 低），权重大——这样重新平衡了分布。

这和因果推断中的 Inverse Probability Weighting（IPW）完全同源。在 observational study 中估计 ATE（Average Treatment Effect）时，IPW 用 treatment assignment 的 propensity score 来纠偏；CausalRM 把同样的思路迁移到了 preference learning 的 feedback assignment 上。

## 实验结果

在多个 LLM backbone（包括 Llama 系列）和 benchmark 上测试：
- WildGuardMix 上 reward accuracy 提升 49.2%
- HarmBench 上提升 32.7%
- 下游 RLHF 任务的对齐效果显著改善

关键 ablation：去掉噪声校正或去掉 propensity reweighting，性能都会明显下降，说明两个组件各自解决了不同的问题，缺一不可。

## 为什么这篇值得读

1. 它把因果推断的经典工具（noise-robust loss、IPW）和 LLM alignment 连接起来，方向非常对。RLHF 领域一直在找更便宜的标注数据源，但很少有人认真处理观测数据的偏差问题。
2. 从工程角度看，如果这条路走通，reward model 训练可以直接用产品侧的用户行为数据，标注成本趋近于零。这对中小团队做 alignment 的意义巨大。
3. Propensity score 的估计质量决定了整套方法的天花板。论文假设 propensity 可以用一个辅助模型估计，但在实际部署中，用户反馈的 missing mechanism 可能比论文假设的更复杂（比如设备差异、场景差异）。这是一个值得关注的局限。

原文：[arxiv.org/abs/2603.18736](https://arxiv.org/abs/2603.18736)

---

## 面试关联知识点

### 1. SFT -> Reward Model -> PPO 三阶段 RLHF

Reward model 是 RLHF 的核心环节。标准做法是用 Bradley-Terry model 把 pairwise preference 转化为 pointwise reward score，损失函数是 cross-entropy over preference pairs。CausalRM 修改的正是这个损失函数——加入噪声校正项和 propensity 权重。面试追问：DPO 绕过了 reward model，直接从 preference data 优化 policy，好处是更稳定、计算量更小，但隐式 reward 的表达能力可能不如显式 reward model。

### 2. Inverse Probability Weighting（IPW）与因果效应估计

IPW 是因果推断中纠正选择偏差的标准方法。核心公式：E[Y(1)] = E[Y * T / e(X)]，其中 e(X) = P(T=1|X) 是 propensity score。直觉：对被低概率选中的样本赋予高权重，使加权后的样本近似随机化实验。局限：当 propensity score 接近 0 时权重爆炸，实践中常用 truncated IPW 或 doubly robust estimator 来稳定估计。这和 ACIC 竞赛中的 causal effect estimation 方法论直接相关。

### 3. Noise-Robust Learning / Label Noise

当训练标签有噪声时，标准 cross-entropy loss 不再是无偏估计。经典解法是 forward correction（Patrini et al. 2017）：如果已知噪声转移矩阵 T（T_ij = P(观测标签=j | 真实标签=i)），可以构造校正后的损失函数使其在期望下等价于 clean loss。面试常见变体：label smoothing 也可以看作一种隐式的噪声处理。
