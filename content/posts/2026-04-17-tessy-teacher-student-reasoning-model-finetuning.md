---
title: "TESSY：Teacher-Student 协作合成数据，解决 Reasoning Model 微调中的灾难性遗忘"
date: 2026-04-17T07:30:00-04:00
tags: [reasoning-models, knowledge-distillation, sft, catastrophic-forgetting]
description: "上海 AI Lab 提出 TESSY 框架，通过 teacher 生成能力 token、student 生成风格 token 的交替策略，将 Qwen3-8B 在代码任务上提升 11.25%，避免直接蒸馏导致的性能退化。"
showToc: true
---

## 背景：为什么 Reasoning Model 不能直接蒸馏？

知识蒸馏的标准做法是：用大模型生成高质量合成数据，拿来 SFT 小模型。这招在 Base/Instruct 模型上屡试不爽，但到了 reasoning model 这里彻底失灵。

原因很直觉但之前没人系统性地拆解过：reasoning model（比如 Qwen3-8B、DeepSeek-R1）已经经历了大量预训练和 RL 对齐，形成了自己独特的"思维风格"。用 GPT-OSS-120B 生成的合成数据去训它，等于强迫它学一套完全不同的表达习惯——不光是"Hmm, let me think"还是"Wait, I need to..."这种过渡语的差异，连推理链的组织方式都不一样。

实验数据很残酷：直接用 teacher 生成的完整数据 SFT Qwen3-8B，在 LiveCodeBench-Pro 上掉 3.25%，OJBench 上掉 10.02%。越"完整"地学 teacher，student 掉得越狠。

## 核心机制：能力 Token vs 风格 Token 的解耦

TESSY 的核心洞察是把模型输出的 token 分成两类：

- **Capability tokens（能力 token）**：直接跟解题相关的内容——代码、数学推导、关键逻辑步骤
- **Style tokens（风格 token）**：过渡性的连接语——"Okay, let's see"、"Wait, but..."、"So the answer is"

训练目标随之分解为两个 loss：能力 loss 和风格 loss。传统 SFT 不区分这两者，但对 reasoning model 来说，风格 loss 的优化会严重干扰能力 loss 的学习。

TESSY 的解法：**能力 token 让 teacher 生成，风格 token 让 student 自己生成**。具体流程是一个交替生成的 pipeline：

1. Student 先生成一段 style span（通常是开头的过渡语）
2. Teacher 接手生成一段 capability span（核心推理步骤）
3. 回到 student 生成下一段 style span
4. 如此交替，直到完成整个 response

每次切换时，模型先生成固定 k=20 个 token，然后用一个 boundary predictor 判断该在哪里截断（generate-then-rollback 策略）。这个 boundary predictor 本质上是一个分类器，判断当前 token 属于 capability 还是 style。

## 技术细节

实现上有几个值得注意的点：

**Prefix caching**：因为 teacher 和 student 交替生成，前缀不断增长。TESSY 基于 vLLM 实现，利用 prefix caching 避免重复计算已生成的部分。

**词表不匹配处理**：当 teacher 和 student 使用不同 tokenizer 时（比如 GPT-OSS vs Qwen），在切换点丢弃最后一个 word，防止 subword 拼接导致语义错乱。

**训练配置**：32 张 H200，batch size 128，lr 5e-5，最多 9 个 epoch。80K 训练样本来自 OpenThoughts 和 Nemotron 数据集的编程竞赛子集。

## 实验结果

以 Qwen3-8B 为 student、GPT-OSS-120B 为 teacher：

| 方法 | LCB-V5 | LCB-Pro | OJBench |
|------|--------|---------|---------|
| Baseline (Qwen3-8B) | 55.09 | 25.35 | 18.75 |
| Teacher-Only SFT | 41.32 | 22.10 | 8.73 |
| **TESSY** | **62.87** | **36.69** | **25.43** |
| 变化 | +7.78 | **+11.25** | +6.68 |

数学和科学的 out-of-domain 测试也没掉，AIME2024 甚至从 76.67 升到 80.42。

论文还对比了多种中间方案（只用 teacher 的 answer、只用 thinking、混合等），全部不如 TESSY 的交替生成策略。

## 为什么重要

这篇论文揭示了一个被广泛忽视的问题：reasoning model 时代的 SFT 不能简单套用 Base model 时代的蒸馏范式。风格分布的 mismatch 不是小事，它直接导致灾难性遗忘。

更实际的意义：如果你要用 GPT-4o / Claude 的输出去训自己的小 reasoning model，不能直接拿全量输出做 SFT。你需要某种形式的 on-policy 数据合成——TESSY 提供了一个具体且开源的方案。

代码和数据集都已开源：[GitHub - CoopReason/TESSY](https://github.com/CoopReason/TESSY)，训练集 [CoopReason/TESSY-Code-80K](https://huggingface.co/datasets/CoopReason/TESSY-Code-80K)。

**原文链接：** [arXiv:2604.14164](https://arxiv.org/abs/2604.14164)

## 面试关联知识点

### SFT 中的 Catastrophic Forgetting

Catastrophic forgetting 指模型在学习新任务/新数据时遗忘已学到的能力。对 reasoning model 特别严重，因为已经经过 RL 对齐，新的 off-policy 数据会破坏已有的推理模式。缓解方法包括：数据混合（replay）、on-policy 数据合成（如 TESSY）、EWC 等正则化方法、以及 LoRA 等参数高效微调减少对原始权重的扰动。

### On-policy vs Off-policy 数据

On-policy 数据的分布与 student 模型自身的生成分布一致，训练时 KL divergence 小，学习更稳定。Off-policy 数据来自另一个模型（teacher），分布差异大，尤其是 style token 的分布偏移会放大训练不稳定性。TESSY 的关键创新就是通过让 student 生成 style token，将 off-policy 数据"半在线化"。

### 知识蒸馏在 LLM 中的变体

经典 KD 用 soft label（teacher 的 logits）。LLM 时代更常用 sequence-level KD：teacher 生成完整回答，student 用 SFT loss 学习。进阶方法包括：on-policy distillation（student 自己采样再用 teacher 打分）、TESSY 式的 token-level 协作生成、以及 DPO/RLHF 中用 teacher 作为 reward model。核心取舍是数据质量 vs 分布匹配。
