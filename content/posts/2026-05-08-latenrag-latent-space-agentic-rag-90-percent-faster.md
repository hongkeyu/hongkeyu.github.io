---
title: "LatentRAG：把 Agentic RAG 的推理和检索搬进隐空间，延迟砍 90%"
date: 2026-05-08T07:30:00-04:00
tags: [RAG, LLM, Latent-Reasoning, Information-Retrieval]
description: "LatentRAG 用 latent token 替代自然语言 thought/subquery 生成，一次 forward pass 完成推理+检索，延迟降低约 90%。"
showToc: true
---

## 背景：Agentic RAG 好用但太慢

传统 single-step RAG 对复杂多跳问题力不从心。Agentic RAG（如 Search-R1、AutoRefine）让 LLM 充当 search agent，迭代生成 thought + subquery，每轮检索后再推理下一步。效果确实好，但代价很大：每一步都要 autoregressive 地生成几十上百 token 的 thought 和 query 文本，平均延迟是 single-step RAG 的 15 倍。在 multi-hop QA 上更严重，因为迭代轮数多。

LatentRAG 要解决的核心矛盾：**多步推理的质量 vs. 逐 token 生成的延迟**。

## 核心机制

LatentRAG 的核心思路是把 thought 和 subquery 从离散语言空间搬到连续隐空间。

### Latent Token 替代自然语言生成

不再让 LLM 一个 token 一个 token 地写出 "I need to find the capital of France"，而是在输入序列里插入固定数量的 special token（`<think_1>...<think_m>` 和 `<query_1>...<query_n>`）。LLM 做一次 forward pass，这些 special token 位置上的 last-layer hidden states 就是 latent thought 和 latent subquery。

原本需要几十步 decode 的过程被压缩到单次 prefill。

### Latent Retrieval：跨空间对齐

LLM 输出空间和检索模型输入空间不一样。LatentRAG 加了一个轻量 projector（一层双向 self-attention + FFN），把 latent subquery token 投影到检索模型的 embedding 空间。

训练时用 frozen 的 reference retrieval model 编码自然语言 subquery 作为 teacher，通过 KL 散度蒸馏检索分布（而不是简单对齐 cosine distance）。论文 ablation 证明 KL 散度优于直接 cosine 对齐。

### Latent Decoding：可解释性保障

隐空间操作的代价是不透明。LatentRAG 加了一个 parallel latent decoding 模块，能把 latent token 解码回自然语言。关键是这些解码可以跨 step 并行（因为不存在 autoregressive 的串行依赖），所以即使开启解码，延迟仍比 explicit 方法低 47-63%。

### 联合训练

总 loss 由三项组成：

| 损失项 | 作用 |
|--------|------|
| L_gen | action token + final answer 的交叉熵 |
| L_ret | 检索分布 KL 散度蒸馏 |
| L_dec | latent decoding 的交叉熵 |

三项加权求和，端到端优化 LLM、projector 和 retrieval model。

## 实验数据

基座模型 Qwen2.5-7B，默认检索器 Qwen3-Embedding-0.6B，在 7 个 QA 数据集上测试：

| 对比方法 | EM 差距 | 延迟变化 |
|----------|---------|----------|
| vs. Search-R1 | 5% 以内 | 5372ms → 593ms（-89%）|
| vs. AutoRefine | — | 4827ms → 512ms（-89.4%）|
| 开启 latent decoding | — | 1970-2540ms，仍显著低于 explicit 方法 |

检索模型从 0.6B 扩展到 8B、LLM 从 7B 扩展到 32B 均有一致提升。e5-base-v2 表现较差，原因是 embedding anisotropy。

## 为什么重要

这篇论文的价值不只是 "快了 10 倍"。它提出了一个更深层的范式转移：**Agent 的中间推理步骤不一定要落到自然语言上。**

Latent token 本质上是给 LLM 额外的"思考槽位"，让模型在 hidden state 空间里完成推理，只在最终输出时才回到语言空间。这个思路和 Coconut（continuous chain-of-thought）、Quiet-STaR 一脉相承，但 LatentRAG 把它具体落地到了 RAG 的检索-推理循环里，而且给出了完整的训练方案和可解释性方案。

对工程落地来说，90% 的延迟降低意味着 Agentic RAG 终于有可能用在实时场景。之前 5 秒一个 query 的系统现在可以做到 500ms，这对搜索、客服、实时问答产品的可用性是质变。

## 面试关联知识点

### Multi-hop 检索的核心挑战

单步检索无法处理需要多跳推理的问题（如 "A 的导师 B 在哪所大学任教？"）。Agentic RAG 通过迭代生成 subquery 解决，但每步 autoregressive 生成带来高延迟。LatentRAG 证明中间推理可以在隐空间完成，不需要显式生成自然语言。

### KL 散度蒸馏 vs. 直接 cosine 对齐

Cosine 对齐只约束 query embedding 的方向，KL 散度约束的是整个检索分布（query 对所有 candidate document 的相似度分布）。后者保留了更丰富的排序信息，尤其在训练数据少、pseudo-relevant document 有噪声时更鲁棒。

### Embedding Anisotropy

Anisotropy 指 embedding 向量高度集中在超球面的一个窄锥区域，导致任意两个 embedding 的 cosine similarity 都很高、区分度低。e5-base-v2 在本文实验中因此表现差。解决方案包括 whitening、isotropy 正则化、或选用分布更均匀的模型。

## 延伸阅读

- [LatentRAG 原文](https://arxiv.org/abs/2605.06285)
- [Search-R1](https://arxiv.org/abs/2503.09516) — 当前最强的 training-based agentic RAG baseline
- Coconut（Continuous CoT）— 同一思路在纯推理任务上的先驱
- 检索器 anisotropy 问题值得注意：选检索模型时不能只看 MTEB 分数，要关注 embedding 分布是否均匀
