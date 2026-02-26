---
title: "Qwen3.5：3B 激活参数干翻上一代 235B"
date: 2026-02-25T19:30:00+08:00
tags: ["Qwen", "MoE", "线性注意力", "LLM", "推理优化"]
description: "阿里 Qwen3.5 Medium 系列发布，35B-A3B 模型用 3B 激活参数超越上一代 235B。首次融合 Gated DeltaNet 线性注意力和 Softmax Attention 混合架构，256 experts 选 8+1。"
showToc: true
---

## 背景：MoE 的效率革命走到哪了

Mixture-of-Experts 不是新概念（Google 的 Switch Transformer 2021 年就有了），但过去两年的趋势是：MoE 的激活比例在急剧下降。

- Qwen3-235B-A22B：235B 总参数，22B 激活（9.4%）
- DeepSeek-V3：671B 总参数，37B 激活（5.5%）
- Qwen3.5-35B-A3B：35B 总参数，3B 激活（8.6%）

关键不是百分比，而是绝对数字：3B 激活参数打赢了 22B 激活参数的上一代旗舰。这说明模型质量的提升越来越多来自训练方法（数据质量 + RL）而非粗暴堆参数。

## 架构核心：Gated DeltaNet + Softmax Attention 混合

Qwen3.5 最值得研究的技术点不是 MoE，而是它的注意力层设计。

传统 Transformer 全部使用 Softmax Attention，复杂度 O(n²)。长上下文（128K+）时 prefill 和 KV Cache 都成为瓶颈。

Qwen3.5 用了一个混合方案：40 层中，每 4 层有 1 层标准 Softmax Attention，其余 3 层用 **Gated DeltaNet**（一种线性注意力变体）。比例大约 3:1。

**Gated DeltaNet 是什么？**

标准 Attention 的核心运算是 Q·K^T，对序列中所有 token pair 做相似度计算，复杂度 O(n²)。线性注意力的思路是：把 softmax(Q·K^T) 分解为 φ(Q)·φ(K)^T，其中 φ 是某种核函数。分解后可以利用矩阵乘法结合律，先算 φ(K)^T·V（复杂度 O(d²)），再和 φ(Q) 相乘，总复杂度变成 O(n·d²)——对 d << n 的长序列来说是线性的。

DeltaNet 在线性注意力基础上加了 delta update rule（增量更新规则），让模型在处理序列时维护一个压缩的"记忆矩阵"，每步只做增量修改而非全量重算。Gated 版本再加上门控机制，控制信息的保留和遗忘。

**为什么混合而不是全换？**

纯线性注意力的表达能力不如 Softmax Attention——它无法精确计算 token 间的尖锐相关性（softmax 的"赢者通吃"特性）。所以 Qwen 选择保留 25% 的 Softmax 层来处理需要精确检索的场景（如引用、精确匹配），其余 75% 用线性注意力来高效处理长距离依赖。

效果：Qwen3.5 在 32K 上下文下 decode 速度是 Qwen3-Max 的 8.6 倍，256K 下是 19 倍。

## MoE 细节：256 experts 选 8+1

MoE 层同样激进：256 个 routed experts，每个 token 激活 8 个 + 1 个 shared expert。只有约 3.5% 的 experts 被激活。

Shared expert 的作用是处理所有 token 都需要的"通用知识"（语法、常见模式），routed experts 负责特定领域的专业知识。

对比 GLM-4.7-Flash（智谱）：64 experts 选 4+1，传统全 Softmax Attention。两者参数规模相近（35B vs 30B），但 Qwen3.5 在知识和推理上大幅领先（MMLU-Pro 85.3 vs ~60，GPQA 84.2 vs 75.2），差距主要来自架构设计和训练数据质量。

## 训练方法：四阶段后训练

1. **Long CoT Cold Start**：用长链推理数据做 SFT
2. **Reasoning RL**：推理任务上的强化学习
3. **General RL**：扩展到通用任务和 agent 场景
4. **Alignment**：安全性和用户偏好对齐

FP8 原生训练：激活内存降 50%，速度提升 10%+，支持万亿 token 级训练。

## 延伸思考

Qwen3.5 的混合注意力架构指向一个趋势：未来的 Transformer 可能不会全部替换成线性注意力或 SSM，而是按比例混合——让不同类型的层各司其职。

对边缘部署来说，线性注意力层不需要 KV Cache 是个巨大优势。如果一个模型 75% 的层是线性注意力，KV Cache 直接缩小到原来的 25%。

---

*资源：*
- *HuggingFace: [Qwen3.5 Collection](https://huggingface.co/collections/Qwen/qwen35)*
- *对比评测: [Qwen3.5-35B-A3B vs GLM-4.7-Flash](https://awesomeagents.ai/tools/qwen-3-5-35b-a3b-vs-glm-4-7-flash/)*

---

## 面试关联知识点

**Q1：MoE 的核心原理是什么？如何实现"大参数、小计算"？**

MoE 在 FFN 层中设置多个并行的 expert 网络，每个 token 通过可学习的 router 选择激活 top-k 个 experts。总参数量大但每个 token 只经过少数 expert 的计算。关键挑战是 load balancing——防止 expert 坍缩，通常通过辅助 loss 或 noise 注入缓解。

**Q2：线性注意力和标准 Softmax Attention 的本质区别？为什么不能完全替代？**

标准 Attention 通过 softmax 归一化使权重分布呈"尖锐"形态，少数 token 获得极高权重。线性注意力去掉 softmax 用核函数分解，复杂度从 O(n²) 降到 O(n·d²)，但失去了尖锐的选择性，无法精确"锁定"关键 token。所以 Qwen3.5 采用混合方案。

**Q3：KV Cache 在混合注意力架构中有什么特殊考虑？**

MoE 的 KV Cache 只涉及 Attention 层，与 expert 数量无关。混合架构中 75% 线性注意力层不需要传统 KV Cache（维护固定大小记忆矩阵），只有 25% Softmax 层需要 KV Cache，内存需求缩小到传统模型的约 1/4。
