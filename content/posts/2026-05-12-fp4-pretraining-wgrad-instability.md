---
title: "用 FP4 预训练 LLM：Wgrad 才是真正的不稳定根源"
date: 2026-05-12T07:30:00-04:00
tags: [low-precision-training, quantization, AMD, Hadamard-rotation, LLM]
description: "AMD 在 MI355X 上实测 MXFP4 预训练 Llama 3.1-8B，定位 Wgrad 量化为 FP4 训练发散主因，确定性 Hadamard 旋转是目前唯一有效的稳定手段。"
showToc: true
---

## 背景

FP8 训练已经成为大模型训练的实用选择，但 FP4（4-bit floating point）训练一直是个硬骨头。核心难题在于：4-bit 的动态范围太窄，activation 和 gradient 中的 outlier 会严重放大量化误差，导致训练发散。之前的工作（包括 NVIDIA 的 NVFP4 方案）大多依赖软件模拟，无法真正验证硬件级行为。

这篇论文的独特之处在于：**它是第一个在原生支持 FP4 tensor core 的硬件（AMD Instinct MI355X）上做全流程 MXFP4 预训练的工作**。不靠模拟，直接上硬件。

## 核心机制：逐阶段量化诊断

训练一个 Transformer 线性层涉及三个 GEMM 运算：

| 运算 | 含义 | 作用 |
|------|------|------|
| **Fprop** | 前向传播 | activation × weight |
| **Dgrad** | 激活梯度 | 反向传播中计算输入梯度 |
| **Wgrad** | 权重梯度 | 反向传播中计算权重更新量 |

作者用"逐阶段开启 MXFP4"的控制实验方法，在 MLPerf C4 数据集上预训练 Llama 3.1-8B（目标 validation perplexity 3.3），逐步把每个阶段从 FP8 切换到 FP4：

| 量化范围 | 额外 token 开销 | 稳定性 |
|----------|-----------------|--------|
| 仅 Fprop | +8-9% | 影响很小 |
| Fprop + Dgrad | +10-11% | 依然可控 |
| Fprop + Dgrad + Wgrad | +26-27% | 训练轨迹不稳定 |

结论很清楚：**Wgrad 量化是 FP4 训练不稳定的主要驱动因素**。

## 为什么 Wgrad 这么敏感？

MXFP4 使用 micro-scaling 量化策略——不是给整个 tensor 一个 scale factor，而是每 32 或 16 个元素共享一个 scale。当 Wgrad 中存在结构性的数值分布不均匀时，micro-scaling 产生的量化误差不是随机噪声，而是沿着特定梯度方向的**系统性偏差**。

### 失败的方案

Stochastic rounding 和 randomized Hadamard 旋转。逻辑是"加入随机性来打散 outlier 的影响"，但实验表明它们反而放大了量化误差——加噪声无法修正结构性错误。

### 成功的方案

**确定性 Hadamard 旋转（deterministic Hadamard rotation）。** 它在量化前对 tensor 做一个固定的正交变换，把集中在少数维度上的 outlier 能量均匀分散到所有维度。这不是加噪声，而是改变数据的几何结构，让 micro-scaling 的分组量化更均匀。

使用 16 维 Hadamard 旋转（H16）的全流程 MXFP4 训练，收敛轨迹几乎与 FP8 baseline 重合，而且由于 FP4 的带宽优势，端到端吞吐量反而更高。

## 关键数字

- H16 kernel 比 H32 快 8%（1.08x vs 1.00x）
- 稳定后的全流程 MXFP4 在 MI355X 上的 step throughput 超过 FP8 baseline
- 但作者诚实指出：这个 recipe 不是通用的，换模型、换数据集、换微调方法可能需要重新验证

## 为什么重要

FP4 训练的意义不只是"省一半显存"。当训练 precision 从 FP8 降到 FP4，内存带宽压力大幅降低，而带宽正是大 batch 训练的核心瓶颈。如果 FP4 训练能稳定工作，意味着：同样的硬件能训练更大的模型，或者同样的模型用更少的卡。

这篇论文的价值在于它不只是"又一个 FP4 方案"，而是系统性地回答了"FP4 训练到底在哪里坏掉、为什么坏掉"：

1. **Wgrad 是瓶颈** — 不是前向传播，不是 Dgrad
2. **Stochastic methods 无效** — 随机性治不了结构性偏差
3. **Deterministic rotation 有效** — Hadamard 变换重新分布 outlier 能量

这三个结论对后续所有低精度训练工作都有指导意义。

同时，AMD MI355X 原生支持 MXFP4 意味着硬件厂商已经在为 FP4 训练铺路，这不是纯学术研究，而是即将落地的工程能力。

## 面试关联知识点

### Micro-scaling vs per-tensor scaling

Per-tensor scaling 用一个 scale factor 覆盖整个 tensor，outlier 会撑大 scale 导致非 outlier 区域精度损失。Micro-scaling（如 MXFP4）将 tensor 分成小组（如每 32 个元素），每组独立 scaling，能更好地适应局部数值分布。代价是需要存储更多 scale factor，但能显著降低量化误差。

### 训练量化 vs 推理量化

推理只需要 Fprop，weight 是固定的，可以离线校准。训练需要 Fprop + Dgrad + Wgrad 三个 GEMM，梯度的数值分布随训练动态变化，没有"校准集"可用。特别是 Wgrad 对参数更新方向的精度非常敏感——量化误差直接累积到模型权重中，错误会在迭代中复合放大。

### Hadamard 旋转的作用

Hadamard 矩阵是一个正交矩阵，对 tensor 做 Hadamard 变换等价于把数据在正交基上旋转。效果是把集中在少数维度的 outlier 能量分散到所有维度，使得后续的分组量化更均匀。关键特性：它是确定性的（不引入随机噪声），计算开销极低（可以用递归的蝶形运算实现），且正交变换不改变向量的 L2 范数——信息没有丢失，只是重新分布了。

## 原文链接

- 论文：[arXiv:2605.09825](https://arxiv.org/abs/2605.09825)
