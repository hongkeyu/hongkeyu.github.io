---
title: "Compute-Optimal QAT: 量化训练的 Scaling Law"
date: 2026-03-05T07:30:00+08:00
tags: ["quantization", "scaling-law", "QAT"]
description: "Apple ICLR 2026 论文证明 QAT 阶段的最优训练比例随计算预算增长而增大，并提出 scaling law 预测最优分配策略。"
showToc: true
---

做过模型量化的人都知道，Quantization-Aware Training (QAT) 是目前得到高质量量化模型的最佳方案。和 Post-Training Quantization (PTQ) 不同，QAT 在训练过程中就引入量化模拟，让模型提前适应低精度带来的信息损失。

实际操作中，QAT 通常不从头开始，而是先用全精度 (FP) 训练一段时间，再切换到 QAT 阶段。这就引出了一个关键问题：给定固定的计算预算（token 总量），FP 和 QAT 之间应该怎么分配？

此前 Liu et al. (2025) 的结论是花 10% 的训练步数在 QAT 上就够了。但这篇来自 Apple（一作是 EPFL 实习生 Aleksandr Dremov）的论文用大量实验推翻了这个结论。

## 核心发现：最优 QAT 比例随计算规模增长

作者在 86M 到 2.2B 参数的模型上，跑了从 2.3B 到 1.4T token 的实验，覆盖 1-bit、2-bit、4-bit、6-bit 四种量化位宽。关键发现：

1. 最优 QAT fraction f* 不是常数，而是随着总 token 数增大而增大。计算预算越大，越应该把更多比例分配给 QAT 阶段。

2. 这个最优比例可以用一个简单的统计量来预测：tokens-per-parameter-byte，即总 token 数除以模型参数量再除以量化位宽对应的字节数。这个指标同时编码了模型大小、训练长度和量化难度三个维度的信息。

3. 更大的模型更容易量化（同等 token/param 比率下 QAT 需要更少比例），训练更久的模型更难量化（FP 阶段积累的 pattern 越多，量化后要修正的偏差越大），更低 bit-width 需要更多 QAT 比例。

## Scaling Law 的形式

论文在 Chinchilla scaling law 的基础上扩展，提出了一个同时建模 FP token 数 D_fp、QAT token 数 D_qat 和量化位宽 B 的 loss 预测公式。核心思路是把 QAT 引入的误差建模为一个和 FP 训练长度正相关、和模型大小负相关的惩罚项。

这个 scaling law 能做到几件事：
- 给定总预算，预测最优 FP/QAT 分配比例
- 预测不同 bit-width 下的最终 loss
- 在内存约束下，判断应该选择较大的高 bit 模型还是较小的低 bit 模型

## Cooldown + QAT Fusion：省计算的实用技巧

论文还提出了一个工程上很实用的方法：把 learning rate cooldown 和 QAT 阶段合并。传统做法是 FP 训练完做 cooldown（LR 衰减），然后再启动 QAT。但 cooldown 阶段的 FP 更新其实是"浪费的"——模型马上就要被量化，这些精细的 FP 调整会被量化噪声覆盖。

把 cooldown 和 QAT 融合在一起，同时做 LR 衰减和量化训练，实验显示可以在相同 token 预算下达到更好的精度。这对资源受限的团队来说是个免费的优化。

## 为什么这篇论文值得关注

量化不再是"训完模型顺便做一下"的事后步骤。随着 on-device deployment 成为主流需求（手机、Jetson 等边缘设备），QAT 的计算预算规划变成了训练流程设计的核心问题。这篇论文的贡献在于把这个问题从"拍脑袋"变成了"有公式可算"。

对于在 Jetson 上部署量化模型的场景，这篇论文的启示是：如果你在做 QAT fine-tuning，不要默认用 10% 的比例，应该根据你的模型大小和数据量来调整。数据量越大、bit-width 越低，QAT 阶段应该占更大比例。

论文链接：https://arxiv.org/abs/2509.22935

## 面试关联知识点

### QAT vs PTQ 的区别和适用场景

QAT 在训练时插入 fake quantization 节点，用 Straight-Through Estimator (STE) 近似不可导的量化操作的梯度，让模型在训练中适应低精度。PTQ 在训练后直接量化权重/激活值，速度快但精度损失大，尤其在低 bit-width（2-bit 以下）时差距显著。QAT 精度更高但需要额外训练计算；PTQ 适合快速部署且对精度要求不极端的场景。

### Scaling Law 的基本形式和意义

Chinchilla scaling law: L(N,D) = E + A/N^alpha + C/D^beta，预测 loss 随模型参数量 N 和训练 token 数 D 的变化。核心结论是给定计算预算存在最优的 N 和 D 分配。这篇论文将其扩展到 QAT 场景，加入了量化位宽维度。面试时常问"为什么不能无限增大模型/数据"——因为两者的边际收益递减且存在最优比例。

### KV Cache Quantization 和模型量化的关系

模型权重量化（W4A16、W8A8 等）减少存储和计算开销，KV Cache 量化则减少推理时的内存占用。两者是互补的优化手段。QAT 主要针对权重和激活值的量化，KV Cache 量化通常用 PTQ 方法（如 per-channel/per-token 量化）。在边缘设备上两者经常一起用。
