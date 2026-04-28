---
title: "Abstract Chain-of-Thought: 用离散抽象 token 替代冗长推理链"
date: 2026-04-27T07:30:00-04:00
tags: [LLM, Chain-of-Thought, Inference-Efficiency, IBM-Research]
description: "IBM Research 提出 Abstract-CoT，用少量离散抽象 token 替代完整自然语言推理链，token 数减少最高 11.6 倍，性能基本持平。"
showToc: true
---

## 背景

Reasoning model 已经成为 LLM 的标配方向。从 DeepSeek-R1 到 Qwen3-Think 再到 Claude 的 extended thinking，核心范式都是让模型在回答前先生成一段长长的自然语言推理过程（verbalized CoT）。问题是：这段推理链太贵了。一道 MATH-500 题目，Qwen3-8B 的 baseline 推理链平均 1205 个 token；经过 SFT+RL 后涨到 1671 个。这些 token 需要逐个自回归解码，延迟和成本线性增长。

之前也有人试过用连续隐向量替代文本推理（比如 Coconut），但效果普遍不如 verbalized CoT。这篇论文的核心问题是：能不能用一小段离散 token（非自然语言）来替代，同时保留 CoT 带来的推理增益？

## 核心机制

Abstract-CoT 的做法分三步：

### 第一步：扩展词表

在原始 tokenizer 之外新增 M 个从未见过的保留 token（TOKEN_A, TOKEN_B, ..., TOKEN_Z, TOKEN_AA, ...），加上 `<beginabstract>` 和 `<endabstract>` 两个定界符。这些 token 的 embedding 随机初始化，一开始没有任何语义。

### 第二步：Policy Iteration Warm-up（核心创新）

这是一个交替训练循环，迭代 T=3 轮：

- **Phase A（信息瓶颈 SFT）：** 输入 prompt + verbal CoT + abstract tokens + response。关键操作是用 attention mask 让 response 只能看到 abstract tokens，看不到 verbal CoT。这迫使 abstract token 的表示学会压缩 verbal CoT 中的有用信息，形成信息瓶颈。
- **Phase B（Self-distillation）：** 丢掉 verbal CoT，只用 prompt 作为输入，让模型自己通过 constrained decoding 生成 abstract token 序列，再用这些 on-policy 生成的序列做 SFT。

这个循环的直觉是：先从 verbal CoT "蒸馏"信息到 abstract token 的表示空间，再让模型学会自主生成这些 abstract 序列，不再依赖 verbal CoT 作为拐杖。

### 第三步：RL 精调

Warm-up 之后，用 generative reward model 做强化学习，在 constrained decoding（只允许生成 abstract 词表中的 token）的约束下进一步优化 abstract 序列的生成策略。

## 关键实验结果

在 Qwen3-8B 上的数据最具说服力：

| Benchmark | Abstract-CoT | SFT+RL (Baseline) | Token 压缩比 |
|---|---|---|---|
| MATH-500 | 90.8% / 144 tokens | 92.6% / 1671 tokens | **11.6×** |
| AlpacaEval-LC-2.0 | 60.8% win-rate / 225 tokens | 58.4% / 496 tokens | 2.2× |
| HotpotQA | 58.8 F1 / 171 tokens | 58.1 F1 / 735 tokens | 4.3× |

在指令跟随和多跳问答上，Abstract-CoT 不仅更高效，甚至比完整 verbal CoT 更好。数学推理稍有损失但幅度很小。

该方法在 Qwen3-4B 和 Granite-4.0-Micro (3B) 上表现出类似趋势，说明跨模型家族的泛化性成立。

## 一个有趣的发现

论文观察到 abstract token 的使用频率呈现 power law 分布，类似自然语言中的 Zipf 定律。这个分布在训练的不同阶段还会演化。这暗示模型确实在 abstract token 空间中发展出了某种"语言"结构，而非简单的噪声填充。

## 为什么重要

1. **推理成本是当前最紧迫的工程问题之一。** Reasoning model 的推理 token 动辄上千，直接拉高延迟和 API 费用。Abstract-CoT 提供了一条把推理 token 压缩到两位数的路径。
2. **纯 post-training 方案。** 不需要 continued pre-training，不需要改模型架构，只需要扩展词表 + 两阶段训练。工程落地成本低。
3. **离散 vs 连续的路线之争。** 之前 Coconut 等工作走连续隐向量路线，Abstract-CoT 证明离散 token 也能做到，且更容易 debug 和审计。
4. **对 edge deployment 友好。** 如果推理链从 1000+ token 降到 ~150 token，在端侧设备上跑 reasoning model 的可行性大幅提升。

## 延伸思考

这篇论文其实在挑战一个隐含假设：推理过程必须对人类可读。如果模型可以用自己的"内部语言"高效推理，那 CoT 的可解释性价值就需要重新审视。这和 DeepSeek-R1-Zero 在推理链中自发出现语言混杂的现象异曲同工——模型并不天然偏好人类可读的推理格式，那只是我们训练出来的。

论文链接：[arXiv:2604.22709](https://arxiv.org/abs/2604.22709)

---

## 面试关联知识点

### Q1: Chain of Thought 为什么能提升推理能力？它的主要代价是什么？

CoT 通过让模型显式生成中间推理步骤，将复杂问题分解为多个子问题，每一步的输出作为下一步的条件输入，增加了模型的"有效计算深度"。本质上是用 autoregressive decoding 的序列长度换取更强的推理能力。主要代价是推理 token 数量大幅增加（通常 5-20 倍），直接导致 inference latency 和成本线性增长。Abstract-CoT 的核心贡献就是证明这些中间 token 不需要是自然语言。

### Q2: Speculative Decoding 和 Abstract-CoT 在优化推理效率上的思路有什么区别？

Speculative Decoding 不改变输出内容，用小模型做 draft 再让大模型验证，加速的是每个 token 的生成速度。Abstract-CoT 则直接减少需要生成的 token 数量——用 ~150 个 abstract token 替代 ~1500 个自然语言 token。两者正交，理论上可以叠加使用。

### Q3: 信息瓶颈（Information Bottleneck）在这篇论文中如何体现？

Warm-up Phase A 中，response 的 attention mask 被设置为只能 attend to abstract tokens，不能看到 verbal CoT。这意味着 abstract tokens 必须从 verbal CoT 中"压缩"出足够的信息来支持 response 生成，形成经典的信息瓶颈结构：verbal CoT → abstract tokens → response。这迫使 abstract token 的 embedding 学习到高密度的推理信息表示。
