---
title: "OLMo Hybrid: 混合架构凭什么比纯 Transformer 省一半数据"
date: 2026-03-16T07:30:00+08:00
tags: [hybrid-architecture, transformer, linear-rnn]
description: "Ai2 发布 OLMo Hybrid 7B，将 75% 的 attention 层替换为 Gated DeltaNet，在 MMLU 上用 49% 更少的 token 达到同等精度，数据效率翻倍。"
showToc: true
---

原文: https://allenai.org/blog/olmohybrid

## 背景：Transformer 的天花板在哪

Transformer 统治 LLM 领域快十年了，核心武器是 self-attention：每个 token 可以直接看到序列中所有前文，擅长"精确回忆"——你问第三段说了什么，它能翻回去找到。但 attention 有两个结构性弱点：

第一，计算量随序列长度二次增长。上下文从 4K 拉到 64K，计算量涨 256 倍。虽然 Flash Attention、稀疏注意力等工程优化能缓解，但二次方的本质没变。

第二，Transformer 不擅长"状态追踪"（state tracking）。比如模拟一盘棋，每走一步棋盘状态都在变，Transformer 需要重新扫描整个历史才能算出当前状态。这种递推式计算本质上更适合 RNN。

线性 RNN（如 Mamba、RWKV、DeltaNet）正好互补：推理时只维护一个固定大小的隐藏状态，复杂度线性，天然适合状态追踪。但代价是"有限记忆"——过去的信息被压缩进固定维度的状态向量，精确回忆能力弱。

所以一个自然的想法：能不能把两者混在一起？

## OLMo Hybrid 的做法

Ai2 的方案很直接：在 OLMo 3 7B 的基础上，把 75% 的 sliding-window attention 层替换成 Gated DeltaNet 层，保留 25% 的全局注意力层。具体排列是 3:1 模式——三层 DeltaNet 接一层 multi-head attention，循环堆叠。

为什么是 3:1 而不是 1:1？因为实验表明大部分 layer 做的是"状态更新"类计算，DeltaNet 足以胜任；只需要少量 attention 层来做"长距离精确检索"，防止信息在有限的 RNN 状态中被遗忘。这个比例是通过 1B 规模的消融实验确定的。

Gated DeltaNet 是什么？它是 DeltaNet 的改进版，核心思想是用一个可学习的 delta rule 来更新隐藏状态：每一步根据当前输入决定"遗忘多少旧信息、写入多少新信息"。相比标准 RNN 的 sigmoid gate，delta rule 的更新更具表达力。关键是它在训练时可以并行化（通过 chunkwise 并行），不需要逐 token 串行，所以训练速度和 Transformer 相当。

## 核心结果

最硬的数字：在完全相同的训练数据和硬件条件下，OLMo Hybrid 在 MMLU 上用 49% 更少的 token 就追平了 OLMo 3 的分数。在 Common Crawl 评估上，token 节省 35%。由于两种架构的训练吞吐量基本一致（参数量相同，每 token 计算量接近），token 节省直接等于计算量节省。

6T token 训练完成后，OLMo Hybrid 在数学和科学 benchmark 上明显超过 OLMo 3，但在代码任务上略逊。经过 mid-training（继续训练阶段）后，所有评估维度上 Hybrid 都反超了 OLMo 3。

长上下文表现更惊艳：在 RULER 长上下文 benchmark 上，64K 长度时 OLMo Hybrid（使用 DRoPE 位置编码）得分 85.0，而 OLMo 3（使用 YaRN）只有 70.9。差距在上下文越长时越大——这正是混合架构的优势区间，RNN 层的线性复杂度在长序列上的优势开始兑现。

## 为什么混合架构更高效？理论解释

Ai2 团队给出了一个理论层面的解释：混合模型的表达能力（expressivity）严格大于纯 Transformer 或纯线性 RNN。Transformer 能表达 recall 但不擅长 state tracking；线性 RNN 能表达 state tracking 但不擅长 recall；混合模型两样都能做，而且能表达一些两者单独都做不到的计算模式。

表达能力更强意味着：面对同样的数据分布，混合模型能用更少的参数/数据拟合到相同的 loss。这就是 2x 数据效率的根本来源——不是训练 trick，是架构本身"能学到更多东西"。

他们的 scaling law 拟合还预测：这种 token 节省因子会随模型规模增大而增长。1B 时大约 1.3x，7B 时 1.9x，外推到 70B 可能更高。如果这个趋势成立，对大规模预训练的成本控制意义重大。

## 对边缘部署的意义

这对 Jetson 这类边缘设备是好消息。RNN 层推理时是 O(1) 内存（相对于序列长度），不需要像 attention 那样维护随序列增长的 KV Cache。75% 的层是 DeltaNet 意味着长上下文推理时显存占用大幅下降。虽然 7B 模型本身对 Jetson Orin 还是偏大，但这个架构方向 scale down 到 1-3B 时，优势会更明显。

## 趋势判断

OLMo Hybrid 不是孤例。Nemotron-H、Qwen3-Next、Kimi Linear、Qwen 3.5 都在走混合路线。纯 Transformer 作为 LLM 唯一架构的时代可能正在结束。2026 年的关键问题不再是"要不要用混合架构"，而是"怎么混、混多少"。

完全开源（权重、数据、代码、中间 checkpoint 全公开），这也是 Ai2 一贯的风格。想复现或在此基础上做实验，门槛很低。

## 面试关联知识点

### 1. Attention 时间复杂度与优化

Self-attention 的时间和空间复杂度都是 O(n^2)，n 为序列长度。优化方向包括：Flash Attention（通过 tiling 减少 HBM 访问，不改变复杂度但大幅提速）、线性 Attention（用核函数近似 softmax，复杂度降到 O(n)，但精度有损）、滑动窗口 Attention（限制每个 token 只看局部窗口，O(n*w)）。OLMo Hybrid 的策略是直接把大部分 attention 换成线性 RNN，比近似 attention 更彻底。

### 2. Speculative Decoding 原理

用一个小的 draft model 快速预测多个 token，再用大模型一次性验证。如果预测对了就直接用，错了就回退到大模型的结果。本质上是用并行验证替代串行生成，不改变输出分布。OLMo Hybrid 的 DeltaNet 层因为推理更快，理论上可以作为更高效的 draft 层使用。

### 3. KV Cache 与长上下文推理

标准 Transformer 推理时需要缓存每一层每个已生成 token 的 K/V 向量，显存占用随序列长度线性增长。对于 7B 模型 + 64K 上下文，KV Cache 可能占到数 GB。混合架构中 RNN 层不需要 KV Cache（只维护固定大小的隐藏状态），所以 75% 的层免于 KV Cache 开销，长上下文场景下显存节省显著。
