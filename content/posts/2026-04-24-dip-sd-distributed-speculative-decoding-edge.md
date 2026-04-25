---
title: "DiP-SD：分布式流水线 Speculative Decoding，边缘场景吞吐提升 17.89 倍"
date: 2026-04-24T07:30:00-04:00
tags: [speculative-decoding, edge-inference, distributed-systems]
description: "清华提出 DiP-SD，端侧 draft + 边缘 verify 的流水线架构，联合优化 batch 分配与 draft 长度，在 Qwen3-1.7B/32B 上实现 17.89x 吞吐提升"
showToc: true
---

## TL;DR

清华团队提出 DiP-SD，在 edge 场景下将 speculative decoding 的 draft 阶段分散到用户设备、verify 阶段集中在边缘服务器，通过流水线调度和联合优化 batch 分配与 draft 长度，在 Qwen3-1.7B/Qwen3-32B 组合下实现比纯 autoregressive decoding 高 17.89 倍的吞吐。

## 背景

Speculative decoding (SD) 的核心思路：用小模型快速生成 draft tokens，大模型一次性 verify，通过 reject-and-resample 保持目标分布不变，一次 forward pass 接受多个 token。从 2023 年 Leviathan 和 Chen 的两篇奠基论文开始，SD 已经成了 LLM 推理加速的标配技术。

但 SD 有天然的扩展性问题：多用户并发时，验证端成为共享瓶颈。现有边缘 SD 系统（SLED、SPIN）通常采用静态 batching 或启发式 batch size 决策，没有认真优化 batch 分配和 draft 长度的耦合关系。

## 核心机制

### 部署架构

用户侧设备（手机、平板）运行小的 draft 模型（Qwen3-1.7B），边缘服务器跑大的 target 模型（Qwen3-32B）做验证。在端侧模型越来越能跑的今天，这个架构非常实际。

### 两个维度的并行性

**Device-level distributed drafting**：每个用户设备独立、并行地生成 draft tokens，互不干扰。

**Phase-level draft-verify pipelining**：把多个用户分成 N 个 batch，当第 n 个 batch 在服务器上做 verify 时，第 n+1 个 batch 的用户已经在各自设备上 draft 了。Draft 和 verify 形成流水线，消除 pipeline bubble。

## 技术细节

### 优化问题

核心是一个分数混合整数规划：目标函数 throughput = 期望接受 token 数 / pipeline 总时间 S。需要联合优化三组变量：

| 变量 | 含义 | 求解方式 |
|------|------|----------|
| Batch 数量 N | 用户分成几组 | 外层暴力扫描 N=2..M |
| 用户-to-batch 分配 x | 哪些用户同 batch verify | 固定 draft 长度后解 MILP |
| 每用户 draft 长度 l_m | 每个用户 draft 多少 token | 固定分配后用 Dinkelbach + binary selector 解 MILP |

每个 batch 的 verify 延迟取决于 batch 中最长的 draft 长度和最长的 prefix 长度（padding），所以把 draft 长度差异大的用户分到一起会浪费计算。这个 coupling 让问题变得非平凡。

### 求解方法

交替优化：固定 draft 长度解 batch 分配（MILP），固定 batch 分配解 draft 长度（分数目标用 Dinkelbach 方法转化，draft 长度用 binary selector 编码后也变成 MILP）。用 SCIP 求解器。

### 显存约束

模型参数 + KV cache 不能超过 GPU 显存上限。KV cache 占用 = 4 × 层数 × hidden_dim × batch_size × max_prefix_length。

## 实验结果

在 Qwen3-1.7B (RTX 3090) + Qwen3-32B (A100-80GB) 的设置下：

| 对比基线 | 吞吐提升 |
|----------|----------|
| 纯 Autoregressive Decoding | **17.89x** |
| AD + Greedy Batching | 1.93x |
| 不做 Batching 的 SD（14 并发） | 1.38x |

- 联合优化 draft 长度比固定 l=7 还能额外提升，说明 per-user adaptive draft length 确实有价值
- 吞吐随用户数近乎线性增长——新用户可被吸收进现有 batch 而几乎不增加 pipeline span
- 默认 acceptance rate α=0.78，来自 100 条 prompt 统计；α 越高 DiP-SD 优势越显著

## 为什么值得关注

这篇论文的价值不在于算法本身有多 fancy（交替优化 + Dinkelbach 都是经典方法），而在于它把 speculative decoding 放到了一个非常实际的系统设计问题里：当你真的要在边缘部署 SD 服务多个用户时，batch 怎么分、draft 长度怎么设、流水线怎么排，这些工程决策之间的耦合关系是什么？

更现实的意义：随着 Qwen3、Llama、Phi 等小模型越来越强，"端侧 draft + 服务器 verify" 这个架构正在从论文走向产品。Apple Intelligence 的 on-device + cloud 模式本质上就是这个方向。DiP-SD 给出了 multi-user scheduling 的参考答案。

## 原文链接

[arXiv: 2604.20919](https://arxiv.org/abs/2604.20919)

## 面试关联知识点

### Speculative Decoding 的基本原理是什么？为什么能保证输出分布不变？

SD 用小模型（draft model）自回归生成 K 个 token，然后大模型（target model）对这 K 个 token 做一次并行 forward pass 得到每个位置的概率分布。对每个位置，如果 draft 概率 q(x) ≤ target 概率 p(x)，直接接受；否则以 1 - p(x)/q(x) 的概率拒绝并 resample。这个 accept/reject 过程在数学上等价于从 target 分布采样——这是 SD 和普通近似方法的本质区别：**无损加速**。

### Prefill 和 Decode 阶段的区别？为什么 decode 是推理瓶颈？

Prefill 阶段处理整个输入 prompt，是 compute-bound 的（大矩阵乘法，GPU 利用率高）。Decode 阶段逐 token 生成，每步只处理一个 token，是 memory-bandwidth-bound 的（需要反复读取全部模型权重，但计算量很小）。Decode 延迟与输出长度成正比，且 token 间有依赖关系无法并行，所以是交互式推理的核心瓶颈。SD 通过一次 verify 多个 token 来摊平 decode 的 per-token 开销。

### KV Cache 的显存占用怎么算？为什么它是 batch serving 的主要限制？

每层 Transformer 的 KV Cache 占用 = 2 (K+V) × hidden_dim × seq_len × precision_bytes。对于 64 层、hidden_dim=5120 的模型（如 Qwen3-32B），FP16 精度下单个请求 512 token 的 KV Cache = 2 × 5120 × 512 × 2 × 64 = 671MB。batch_size=8 就是 5.4GB。当 batch size 增大或 context 变长时，KV Cache 很快吃满显存，所以 KV Cache 管理（PagedAttention、量化、offloading）是 serving 系统的核心工程问题。
