---
title: "Mamba-3: 用一半 State Size 打平 Transformer 的 SSM 逆袭"
date: 2026-03-23T07:30:00+08:00
tags: ["SSM", "Mamba", "Transformer"]
description: "Mamba-3 通过二阶离散化、复数 SSM 和 MIMO 解码，在 1.5B 规模上超越 Transformer baseline，同时 state size 减半。"
showToc: true
---

## TL;DR

CMU、Princeton、Together AI 和 Cartesia AI 联合发布 Mamba-3，通过三个核心改进（二阶离散化、复数 SSM、MIMO 解码）在 1.5B 规模上超越 Mamba-2 和 Transformer baseline，同时 state size 减半，推理效率显著提升。

---

## 背景：为什么还在折腾 SSM

Transformer 的 attention 机制有两个根本性瓶颈：prefill 阶段的 O(n^2) 计算复杂度，以及 decode 阶段线性增长的 KV Cache 内存占用。对于长序列推理（比如 128K context），这两个问题直接决定了部署成本和延迟。

State Space Model（SSM）从 Mamba-1 开始就试图用线性复杂度的循环结构替代 attention，但之前的版本在下游任务质量上始终和 Transformer 有差距。Mamba-3 的目标很明确：把这个差距彻底抹平，同时保持 SSM 的效率优势。

## 核心改进一：Exponential-Trapezoidal 离散化

SSM 本质是连续时间系统，需要离散化才能处理 token 序列。Mamba-1/2 用的是一阶 exponential-Euler 方法，精度有限。Mamba-3 换成了二阶的 exponential-trapezoidal 离散化，状态更新公式从两项变成三项：

```
h_t = e^(Delta_t * A_t) * h_{t-1} + (1 - lambda_t) * Delta_t * e^(Delta_t * A_t) * B_{t-1} * x_{t-1} + lambda_t * Delta_t * B_t * x_t
```

这个三项公式等价于在状态-输入乘积 B_t * x_t 上做了一个 data-dependent 的宽度为 2 的卷积。实际效果是：Mamba-3 不再需要之前 SSM 架构必须依赖的外部短因果卷积层，模型结构更干净。

## 核心改进二：复数 SSM 与 RoPE Trick

实值线性模型有一个理论盲区：无法解决 state-tracking 类任务（比如判断 bit 序列的 parity）。根本原因是实数域的转移矩阵特征值无法表达"旋转"动态。

Mamba-3 引入复数值 SSM 来解决这个问题。关键发现是：离散化后的复数 SSM 等价于在 B 和 C 投影上施加 data-dependent 的 Rotary Positional Embedding（RoPE）。这就是论文所说的"RoPE Trick"——通过跨时间步的累积旋转，模型获得了处理 parity、modular arithmetic 等合成任务的能力，而 Mamba-2 在这些任务上表现和随机猜测无异。

这个发现也很有理论意义：它建立了 SSM 离散化和位置编码之间的等价关系，为两个领域的交叉研究打开了新方向。

## 核心改进三：MIMO 解码

SSM decode 阶段的核心问题是 arithmetic intensity 太低——在标准 SISO（Single-Input Single-Output）模式下，每字节只有约 2.5 次运算，远低于 H100 等 GPU 的计算密集区间。换句话说，decode 是纯粹的 memory-bound 操作，GPU 算力大量浪费。

Mamba-3 的解法是把 SISO 升级为 MIMO（Multi-Input Multi-Output），将输入/输出投影的 rank 从 1 提升到 R。状态更新从外积变成矩阵乘法，FLOPs 增加最多 4 倍，但因为这些计算和已有的内存 IO 重叠执行，实际 wall-clock 延迟几乎不变。

在 R=4 的 MIMO 配置下，1.5B 模型的下游平均准确率从 SISO 的 56.4% 提升到 57.6%，perplexity 从 10.35 降到 10.24。作为对比，同规模 Transformer baseline 是 55.4% / 10.51。

## 实验结果

在 FineWeb-Edu 数据集上，180M 到 1.5B 四个规模的实验表明：
- Mamba-3 SISO 就已经超过 Mamba-2（state size 64 打平 Mamba-2 的 128）
- MIMO (R=4) 进一步拉开差距，全面超过 Transformer baseline
- 优化后的 Triton（prefill）和 CuTe DSL（decode）kernel 确保新增计算开销可控

## 为什么这篇值得关注

1. SSM vs Transformer 不再是"理论上更优但实际不行"的状态。Mamba-3 在 1.5B 规模上已经实打实超过 Transformer，随着规模进一步增大，趋势会更明确。

2. 对边缘部署意义重大。SSM 的 O(1) 内存和线性时间复杂度天然适合 Jetson 这类设备。state size 减半意味着同等硬件条件下可以跑更大的模型或更长的上下文。

3. Hybrid 架构（SSM + 少量 Attention 层）可能成为下一代标配。论文也提到 Mamba-3 和 Attention 混合使用时，加入 pre-gate grouped RMSNorm 可以显著改善长度泛化能力。

## 延伸阅读

论文本身还有很多工程细节值得深挖，包括 BC/QK Normalization 的 stabilization 策略、head-specific bias 的设计，以及和 Gated DeltaNet 的对比分析。如果对 SSM 方向感兴趣，建议精读。

- 原文：https://arxiv.org/pdf/2603.15569
- 技术博客：https://www.together.ai/blog/mamba-3
- 代码：https://github.com/state-spaces/mamba

---

## 面试关联知识点

### 1. Attention 时间复杂度 O(n^2) 及优化方案

Transformer self-attention 对序列长度 n 的计算复杂度是 O(n^2 * d)，内存也是 O(n^2)。主要优化路径包括：线性 Attention（用核函数近似 softmax，降到 O(n)）、Flash Attention（通过 tiling 和 online softmax 把 O(n^2) 的中间矩阵从 HBM 移到 SRAM，不改复杂度但大幅减少内存访问）、以及 SSM 类方案（完全放弃 attention，用循环结构实现 O(n) 复杂度和 O(1) 推理内存）。

### 2. KV Cache 原理及其瓶颈

Transformer decode 时需要缓存所有历史 token 的 Key 和 Value 向量，内存随序列长度线性增长。对于 70B 模型 + 128K context，KV Cache 可能占到 50GB+。优化手段包括 GQA（多个 query head 共享 KV head）、KV Cache Quantization（FP8/INT4 压缩）、以及 SSM 方案（固定大小 state 替代无限增长的 KV Cache）。Mamba-3 的 MIMO 在保持固定 state 的前提下提升了质量，是 KV Cache 问题的另一条解题路径。

### 3. RoPE 位置编码

Rotary Positional Embedding 通过对 query/key 向量施加旋转变换来编码相对位置信息。核心思想是把位置信息融入内积计算：两个 token 的 attention score 只取决于它们的相对距离。Mamba-3 的"RoPE Trick"揭示了一个有趣的等价关系——复数 SSM 的离散化天然产生了类似 RoPE 的旋转效果，只是作用在 B/C 投影上而非 Q/K 上。
