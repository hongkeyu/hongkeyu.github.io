---
title: "XFP: 用质量阈值反向驱动量化，把 397B MoE 塞进两张消费级 GPU"
date: 2026-05-16T07:30:00-04:00
tags: [quantization, moe, inference, blackwell, vllm]
description: "XFP 提出质量驱动的自适应 codebook 量化，用 per-layer learned codebook + sparse outlier 分离，在两张 RTX PRO 6000 上以 3.4 effective bits 跑通 Qwen3.5-397B MoE。"
showToc: true
---

## 背景：为什么现有量化在 MoE 上撞墙

现有量化方案——无论是线性量化（INT4/INT8）还是对数量化（NVFP4）——本质上都是"一套 codebook 打天下"。INT4 用均匀网格，NVFP4 用固定 16 个浮点值做 lookup，所有层、所有 expert、所有 projection matrix 共享同一套表示模板。

问题在于不同层的权重分布差异极大。XFP 作者在 GLM-4.7-Flash 的 attention 层观察到 49 sigma 的离群值，而同一模型的 routed expert 层权重却是规整的高斯分布。用同一个 codebook 处理这两种分布，前者把表示预算浪费在 0.3% 的极端值上，后者被过度配置了根本用不到的条目。对 MoE 架构尤其致命：512 个 routed expert 各有不同分布，统一量化导致质量快速崩塌。

更讽刺的是，NVIDIA 宣传的 NVFP4 在消费级 Blackwell（SM120，RTX PRO 6000）上根本没有专用 Tensor Core 加速——那是 datacenter Blackwell（SM100+）的专利。所以消费级用户拿到的是一个既没硬件加速、codebook 又不可学习的格式。

## 核心机制：质量驱动的自适应 Codebook 量化

XFP 的设计哲学完全反转了传统流程：

| 范式 | 流程 |
|------|------|
| **传统方式** | 操作者选定 bit 数（比如 4-bit）→ 量化 → 检查质量是否可接受 |
| **XFP 方式** | 操作者设定质量底线（per-channel cosine similarity 阈值）→ XFP 自动决定每层的 codebook 大小、outlier 预算、packing 方式 |

具体实现有三个关键组件：

### 双阈值自动选择机制

操作者设两个 cosine similarity 阈值：strict 阈值给 attention 和 shared expert（这些层对精度敏感），lazy 阈值给 routed expert（数量多但单个影响小）。XFP 对每层逐步尝试不同 codebook 配置，直到满足对应阈值。

### Sparse Outlier 分离

每个权重矩阵被分解为两部分：一个 sparse fp16 outlier 残差（存放极端值），加一个 dense sub-byte index tensor 指向 per-group learned codebook（group size 128）。离群值不再污染主 codebook 的表示能力。

### Per-layer Lloyd Codebook

V2 模式下，每个 channel 用 Lloyd 算法学出自己的 codebook，条目数量不固定——有些层可能只需要 4 个条目，有些需要 16 个。V2a 模式则用每层共享的 32 个 codebook 库，per-group 选择，节省 SMEM。两种模式共享同一个 fused decode kernel。

## 关键实验结果

### Qwen3.5-122B-A10B（V2 模式）

| 指标 | 数值 |
|------|------|
| Decode 速度 | 138 tok/s（RTX PRO 6000 Blackwell TP=2） |
| GSM8K strict-match | 94.49%（3 seeds，n=3957） |
| vs Marlin INT4（TP=1） | 快 49% |
| Effective bits | ~3.97 |

### Qwen3.5-397B-A17B（H-Process，H1.5 配置）

| 指标 | 数值 |
|------|------|
| 原始权重 | 超过 700GB，压到 2×96GB |
| Decode 速度 | 100.9 tok/s（long-output） |
| GSM8K strict-match | 66.72%（全部 1319 题） |
| Effective bits | ~3.4 |
| vs INT4 + expert pruning | 内存、吞吐、精度全面超越 |

H-Process 是论文提出的质量驱动搜索过程：在双阈值空间中迭代，找到"模型刚好装得下且输出还没崩"的最优操作点。搜索空间由三个边界定义：操作者设定的阈值上限、quantize-on-load 的 OOM 边界、生成质量的垃圾边界。

## 为什么重要

**MoE 量化不应该用统一策略。** 512 个 expert 的分布各不相同，给每个层/expert 学独立 codebook 比强行统一压到 4-bit 效果好得多。这个思路对 GGUF 生态和 vLLM 部署都有直接影响——作者已经把代码提交到 vLLM 的 fork 上了。

**"质量驱动"比"bit 数驱动"更符合实际需求。** 部署者真正关心的是"模型在我的任务上还能不能用"，而不是"它是几 bit 的"。XFP 把这个需求直接编码进了量化流程。

**消费级 Blackwell 的 NVFP4 不能用 Tensor Core 加速。** 这意味着 learned codebook 方案在消费级硬件上反而更有竞争力——因为大家都在做 LUT decode，区别只在 codebook 质量。

## 延伸阅读

论文同期还有一篇值得关注的安全方向工作：[Widening the Gap (arXiv:2605.15152)](https://arxiv.org/abs/2605.15152) 展示了攻击者可以通过注入 outlier 来操控量化行为——在全精度下模型表现正常，量化后触发恶意行为。这对 AWQ、GPTQ、GGUF I-quants 都有效。两篇论文从正反两面说明了同一件事：outlier 是量化的核心战场。

- 原文链接: [arXiv:2605.14844](https://arxiv.org/abs/2605.14844)
- 代码: [flash7777/vllm (multiquant branch)](https://github.com/flash7777/vllm/tree/multiquant)

## 面试关联知识点

### 模型量化的核心方法对比：INT4 vs GPTQ vs AWQ vs Codebook Quantization

INT4 是均匀线性量化，每个权重映射到等间距网格。GPTQ 用 Hessian 信息逐列量化、补偿误差（OBQ 思路）。AWQ 通过 activation-aware 的 per-channel scaling 保护重要通道。Codebook quantization（如 XFP、AQLM）则学习非均匀 codebook，per-layer 或 per-group 适配权重分布。核心区别在于：线性量化假设权重均匀分布，codebook 方法不做这个假设，因此对 outlier-heavy 的分布更鲁棒。

### KV Cache 量化 vs Weight 量化的区别

Weight quantization 作用于模型参数，一次量化多次使用，可以离线优化（calibration data、Hessian 等）。KV Cache 量化作用于推理过程中动态生成的 key-value 对，必须在线完成、不能回头校准。KV Cache 量化直接影响 context length 上限（内存瓶颈），weight quantization 影响模型大小和 decode 吞吐。两者在 MoE 模型上的交互尤为重要：weight 压得越小，留给 KV Cache 的显存越多，能支持的 context 越长。

### Speculative Decoding 的原理和局限

Draft model 生成 k 个候选 token，target model 一次性 verify。接受率取决于 draft-target 分布的 KL 散度。理论上不改变输出分布（rejection sampling 保证）。局限：在高负载 serving 场景下，effective batch size 增大导致 verify 阶段的计算成本上升，speedup 会随 server load 衰减（同期论文 [arXiv:2605.15051](https://arxiv.org/abs/2605.15051) 用 Little's Law 建模了这个现象）。
