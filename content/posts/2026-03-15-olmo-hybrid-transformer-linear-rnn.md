---
title: "OLMo Hybrid: Transformer + 线性 RNN 混合架构，数据效率翻倍"
date: 2026-03-15T07:30:00+08:00
tags: [hybrid-architecture, linear-rnn, scaling-law]
description: "Ai2 发布 OLMo Hybrid 7B，用 Gated DeltaNet 替换 75% 的 attention 层，在 MMLU 上用 49% 更少的 token 达到与纯 Transformer 同等精度，长上下文性能大幅领先。"
showToc: true
---

TL;DR: Ai2 发布 OLMo Hybrid 7B，用 Gated DeltaNet 替换 75% 的 attention 层，在 MMLU 上用 49% 更少的 token 达到与纯 Transformer 同等精度，长上下文性能大幅领先。

---

## 背景

自 2017 年以来，Transformer 一统语言建模的天下。Self-attention 的全局并行访问能力让它在训练和 in-context recall 上表现出色，但代价也很明显：attention 的计算复杂度是 O(n^2)，序列越长推理越贵。更深层的问题是，Transformer 在 state tracking 类任务上天然不擅长——比如跟踪棋盘状态变化这种需要持续更新隐状态的场景。

另一边，RNN 天然适合 state tracking，逐 token 更新隐状态，推理复杂度线性。但传统 RNN 的顺序依赖让它无法高效并行训练。近两年，Mamba、DeltaNet 等 parallelizable linear RNN 重新让 RNN 回到了竞争舞台，但它们把历史压缩到有界状态里，精确 recall 能力受限。

所以问题变成了：能不能把两者的优势拼在一起？

## 核心设计

OLMo Hybrid 的做法很直接：在 7B 模型的层堆叠中，用 3:1 的比例交替排列 Gated DeltaNet 层和标准 Multi-Head Attention 层。也就是说，每 4 个 sublayer 中有 3 个是 DeltaNet（负责 state tracking），1 个是 attention（负责 precise recall）。这样 75% 的 attention mixing 被替换掉了。

Gated DeltaNet 是一种现代线性 RNN 设计，训练时可以并行化（类似 Mamba 的 scan 操作），推理时保持线性复杂度。它的 gate 机制让状态更新更具选择性，能决定哪些信息写入、哪些信息遗忘。

关键约束：训练吞吐量与 OLMo 3 匹配。两个模型参数量相同、训练速度相当，所以效率提升纯粹来自架构本身，不是用速度换性能。

## 实验结果

数据效率方面，OLMo Hybrid 在 MMLU 上用 49% 更少的 token 达到 OLMo 3 同等精度，相当于 2 倍数据效率。在 Common Crawl 评估集上，35% 更少的 token 即可持平。因为训练吞吐量一致，token 节省直接等价于算力节省。

预训练结束时，Hybrid 在数学和科学 benchmark 上明显更好，但 coding 和通用 QA 略逊。经过 mid-training 后，这些差距消失，Hybrid 在所有主要评估域上全面超越 OLMo 3。

长上下文是 Hybrid 架构的杀手级优势。在 RULER benchmark 上，64k context 下 OLMo Hybrid（使用 DRoPE）得分 85.0，而 OLMo 3（使用 YaRN）只有 70.9。差距在 8k 以上开始拉开，context 越长优势越大。这符合直觉：RNN 层的线性复杂度在长序列上天然占优。

## 理论解释：表达能力决定 scaling 效率

Ai2 团队给出了一个很有意思的理论解释。他们证明了：在计算复杂度理论的框架下，Transformer 能表达 TC^0 复杂度类（加上 padding），而 Hybrid 模型能 capture 整个 NC^1，这是一个严格更大的类。NC^1 包含了 boolean formula evaluation 这样的计算，纯 Transformer 做不到。

更重要的是他们对 scaling law 的分析。语言建模本质上是学习大量离散子任务。每个子任务要么在架构的表达范围内（最终会被学到），要么不在（贡献到 irreducible loss）。如果 Hybrid 能表达更多自然语言中出现的子任务，那它每多看一个 token 就能比 Transformer 降更多 loss。他们用 quantization model 形式化证明了这一点：更高的表达能力确实意味着更高效的 scaling。

Scaling law 拟合还预测，token 节省因子随规模增长：1B 参数时约 1.3 倍，70B 参数时约 1.9 倍。如果这个趋势成立，对大规模预训练的成本影响是巨大的。

## 对边缘部署的意义

值得注意的是，75% 的层换成了线性 RNN，推理时这些层的复杂度是 O(n) 而非 O(n^2)。对于 Jetson 这类内存和算力受限的平台，长序列推理的内存占用和计算量都会显著下降。虽然 Ai2 没有专门讨论边缘场景，但 Hybrid 架构天然对推理友好的特性值得关注。

## 位置编码对比与长上下文扩展

RoPE 通过旋转矩阵编码相对位置，外推能力有限。YaRN 通过修改 RoPE 的频率基来扩展上下文窗口。DRoPE 是 OLMo Hybrid 采用的方案，在 64k context 上比 YaRN 高出 14 分（RULER: 85.0 vs 70.9）。

## Scaling Law 与 irreducible loss

Chinchilla scaling law 描述了模型大小和数据量的最优比例关系。OLMo Hybrid 的研究进一步揭示：架构的表达能力直接影响 scaling 效率。可以这样理解：语言建模 = 学一堆子任务，架构能表达的子任务越多，irreducible loss 越低，每多训练一个 token 的边际收益越大。这是一个比单纯"模型大 / 数据多"更深的视角。

## 延伸

Hybrid 架构不是 Ai2 首创。Samba、Nemotron-H、Qwen3-Next、Kimi Linear 等项目都在探索类似方向。但 OLMo Hybrid 的贡献在于提供了迄今最干净的对照实验——与 OLMo 3 完全对齐训练条件，只改架构，证明了收益来自架构本身。同时，完全开源（模型、数据、代码）也让复现和进一步研究成为可能。

一个值得思考的问题：如果 Hybrid 架构的 scaling 优势确实随参数规模增长（70B 时 1.9 倍），那纯 Transformer 在预训练阶段的统治地位可能真的会在未来一两年被打破。

原文：https://allenai.org/blog/olmohybrid

技术报告：https://allenai.org/papers/olmo-hybrid

模型：https://huggingface.co/collections/allenai/olmo-hybrid
