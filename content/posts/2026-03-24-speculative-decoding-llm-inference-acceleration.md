---
title: "Speculative Decoding: 从研究到生产的 LLM 推理加速标准"
date: 2026-03-24T07:30:00+08:00
tags: ["LLM推理优化", "Speculative Decoding", "EAGLE3"]
description: "Speculative decoding 用小模型猜、大模型验的方式把 LLM 推理速度提升 2-3 倍，且输出分布与原模型数学上完全一致，已成为 2026 年生产级推理框架的标配。"
showToc: true
---

**TL;DR: Speculative decoding 用小模型"猜"、大模型"验"的方式，把 LLM 推理速度提升 2-3 倍，且输出分布与原模型数学上完全一致。2025-2026 年已内置于 vLLM、SGLang、TensorRT-LLM，成为生产级推理框架的标配。**

---

## 为什么 LLM 推理慢？

LLM 生成每个 token 都需要完整的一次 forward pass。70B 参数的模型生成 500 个 token，就是 500 次串行的全模型推理。问题不在于 GPU 算力不够，而在于 memory bandwidth 瓶颈：GPU 的计算单元大部分时间在等权重从显存加载过来，arithmetic intensity 大约只有 1 FLOP/byte，远低于 compute-bound 的拐点。GPU 大部分时间在空转。

Speculative decoding 的核心洞察是：既然每一步都要付出显存带宽的代价，不如一次性验证多个 token，而不是一个一个生成。

## Draft-Then-Verify 机制

系统由两个模型协作：

- Draft model（小模型）：快速串行生成多个候选 token
- Target model（大模型）：一次 forward pass 并行验证所有候选 token

关键在于 Transformer 的 attention 机制天然支持并行计算所有 position，所以大模型可以在一次推理中同时检查多个 draft token 是否与自身的概率分布一致。这不是近似，数学上保证了被接受的 token 分布与大模型独立生成完全相同。

举个例子：用 Llama 3.2-1B 作为 draft model，Llama 3.3-70B 作为 target model。Draft model 连续猜 5 个 token，target model 一次 pass 全部验证。如果 5 个都通过，还能额外生成第 6 个 bonus token。一次推理产出 6 个 token，而不是 1 个。

## Acceptance Rate 决定一切

Acceptance rate (alpha) 是 draft token 被 target model 接受的概率，直接决定实际加速比。在猜 5 个 token 的设定下：

- alpha = 0.5 时，每轮平均接受约 2 个 token
- alpha = 0.7 时，约 2.9 个
- alpha = 0.8 时，约 3.8 个
- alpha = 0.9、猜 8 个时，约 6.1 个

当 alpha 低于 0.5，speculative decoding 反而会拖慢速度——猜了一堆全被拒，白白浪费计算。

Acceptance rate 高度依赖任务类型。代码补全、结构化输出等可预测任务通常在 0.75-0.85；创意写作、高度领域特化的内容则降到 0.5-0.65。

## 四种 Draft Model 方案

### 1. External Draft Model（经典方案）

同系列小模型做 draft，比如 Llama 3.2-1B 配 70B。优点是开箱即用，缺点是要多加载一份模型权重，batch size 大时收益递减。

### 2. EAGLE3 Draft Head（当前最优解）

在 target model 上接一个小型辅助网络，复用 target model 的中间层表示和 KV cache。EAGLE3 融合了多个中间层的 hidden state（不只是最后一层），acceptance rate 显著高于外部 draft model。内存开销极小，只增加几亿参数。目前已有 Llama 3.3-70B、Llama 3.1-8B、Qwen3 等预训练 head。这是 2026 年的工业标准方案。

### 3. N-gram / Prompt Lookup

在上下文中查找重复的 n-gram 模式，用后续 token 作为 draft 候选。零开销、零额外模型，在代码补全和文档编辑等输出重复输入内容的场景效果很好，但对开放式生成无效。

### 4. Self-Speculative（LayerSkip）

模型用自身前几层做 draft，跳过后面的层。不需要第二个模型，但 acceptance rate 通常低于 EAGLE。适用于连小 draft model 都放不下的极端显存限制场景。

## 实测 Benchmark

vLLM + EAGLE3，Llama 3.3-70B 在 4xA100 上：baseline 约 18 tok/s，加上 EAGLE3 达到约 42 tok/s，加速 2.3 倍。

Red Hat Speculators（EAGLE-based），Qwen3-32B 在 2xA100 上：2.7 倍加速。数学推理任务个别场景超过 4 倍。

但规律很明确：低并发（1-10 请求）效果最好，高并发（32+）收益消失甚至负优化。因为高并发下推理变成 compute-bound，memory bandwidth 不再是瓶颈。

## 适用场景判断

**用 speculative decoding：** 交互式低延迟应用（chatbot、coding assistant）、低到中等并发（20 以下）、模型 13B+ 参数、输出可预测。

**不用：** 高吞吐批处理（continuous batching 更合适）、高并发 40+、输出高度创意/领域特化且没有匹配的 draft model、模型本身已经很小或已量化过。

## 对 Jetson 等边缘设备的启示

Jetson Orin Nano 这类 8GB 显存设备，显存带宽更低、更容易 memory-bound，理论上 speculative decoding 的收益空间更大。但实际操作中，连一个外部 draft model 都很难同时放进显存。Self-speculative（LayerSkip）或 n-gram lookup 可能是更现实的路线——零额外显存开销，对边缘场景更友好。

原文链接：[Speculative Decoding: 2-3x Faster LLM Inference](https://blog.premai.io/speculative-decoding-2-3x-faster-llm-inference-2026/)

---

## 面试关联知识点

### 1. Speculative Decoding 原理

**考点：** 为什么能加速？为什么不损失质量？

LLM 推理是 memory-bound 的，GPU 算力利用率低。小模型 draft、大模型一次 forward pass 并行验证多个 token，利用了 Transformer attention 的并行特性。验证使用 rejection sampling，被接受的 token 概率分布与原模型数学上一致，所以输出质量无损。

### 2. KV Cache 与推理优化的关系

**考点：** KV Cache 解决了什么问题？与 speculative decoding 如何配合？

KV Cache 缓存已计算的 key/value，避免重复计算历史 token 的 attention。EAGLE3 draft head 直接共享 target model 的 KV cache，不需要维护独立缓存，这是它比 external draft model 内存效率高的核心原因。

### 3. Prefill vs Decode 阶段区别

**考点：** 为什么 speculative decoding 只在 decode 阶段有效？

Prefill 阶段处理整个 prompt，天然并行，是 compute-bound 的。Decode 阶段逐 token 生成，每步只产出一个 token，是 memory-bound 的。Speculative decoding 解决的正是 decode 阶段的 memory bandwidth 浪费问题。
