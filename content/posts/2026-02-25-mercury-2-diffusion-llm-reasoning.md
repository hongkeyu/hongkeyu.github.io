---
title: "Mercury 2: 扩散语言模型终于能做推理了，速度是自回归模型的10倍"
date: 2026-02-25T07:30:00+08:00
tags: ["diffusion-LLM", "inference-optimization", "reasoning"]
description: "Inception发布Mercury 2，首个推理型扩散语言模型(dLLM)，吞吐量达1000 tokens/s，比Claude 4.5 Haiku快10倍，质量接近同级别模型。"
showToc: true
---

## TL;DR

Inception发布Mercury 2，首个推理型扩散语言模型(dLLM)，吞吐量达1000 tokens/s，比Claude 4.5 Haiku快约10倍，推理质量与Claude 4.5 Haiku和GPT-5.2 Mini相当。这不是硬件优化的胜利，而是架构层面的范式转换。

## 背景：自回归生成的天花板

目前所有主流LLM——GPT、Claude、Gemini、Llama——都是自回归(autoregressive)模型：逐token生成，每一步依赖前一步的输出。这种串行本质意味着无论你怎么优化serving stack、换多贵的芯片、做多狠的量化压缩，生成速度都有一个硬上限。

更棘手的是，当模型需要做长链推理(reasoning)时，生成的token数暴增，延迟和成本线性放大。这就是为什么reasoning model在benchmark上表现惊艳，但实际部署中经常因为太慢、太贵而被降级或砍掉。

行业过去几年在这个问题上的投入方向主要是三条路：专用芯片(如Groq)、推理服务栈优化(vLLM/TensorRT-LLM)、模型压缩(量化/蒸馏)。这些都是在自回归框架内做优化，收益递减。

## 核心：用扩散模型生成文本

Inception走了一条完全不同的路。他们把图像生成领域的扩散(diffusion)方法搬到了文本生成上。

具体机制：Mercury 2不是逐token预测下一个词，而是先生成整个输出的"粗略草稿"(噪声状态)，然后通过多轮去噪(denoising)迭代精炼。关键在于，每一轮去噪可以同时修改多个token——这意味着一次前向传播产生的有效工作量远大于自回归模型的单token输出。

速度优势直接来自模型架构本身，而不是硬件trick。

另一个有趣的副产品：因为扩散模型是迭代精炼而非逐token不可撤回地commit，它可以在生成过程中自我纠错。这对结构化输出(JSON/函数调用)和Agent场景下的可靠性有实际意义。

## 数据：到底有多快

按Artificial Analysis的标准化测试方法：

- Mercury 2：约1000 tokens/s 输出吞吐量
- Claude 4.5 Haiku Reasoning：约89 tokens/s
- GPT-5 Mini：约71 tokens/s

质量方面(reasoning benchmarks)：

- AIME 2025: 91.1
- GPQA: 73.6
- LiveCodeBench: 67.3
- IFBench: 71.3
- SciCode: 38.4

这些分数把Mercury 2放在Claude 4.5 Haiku和GPT-5.2 Mini的竞争区间内。换句话说：质量差不多，但快了一个数量级。

## 为什么这件事重要

扩散语言模型(dLLM)的概念并不新，之前有MDLM、Plaid等工作，但一直停留在"有趣的研究方向"阶段，主要问题是质量上不去。Mercury 2第一次把dLLM推到了production-grade的水平，而且直接对标的是reasoning任务——这是当前LLM应用中价值最高、延迟最敏感的场景。

实际影响最大的几个方向：

Agent循环：多步Agent工作流中，每一步的延迟会复合累积。如果单步推理从1秒降到0.1秒，一个10步的Agent循环总时间从10秒降到1秒，体验完全不同。

实时语音/搜索：p95/p99延迟决定用户体验。1000 tokens/s的吞吐量让reasoning model第一次有可能塞进实时SLA。

大规模编码辅助：快速prompt-review-tweak循环需要低延迟，Mercury 2的速度让"边写边推理"成为可能。

## 团队背景

Inception由Stanford、UCLA和Cornell的研究者创立。CEO Stefano Ermon是扩散模型领域的核心人物之一(DDPM/score matching方向的奠基工作)。团队成员还参与过Flash Attention、Decision Transformer、DPO等重要工作的研发。投资方包括Menlo Ventures、Mayfield，个人投资者包括Andrew Ng和Andrej Karpathy。

## 延伸思考

一个值得关注的问题是：扩散语言模型的scaling law是否与自回归模型不同？如果dLLM在scale up时能保持同样的质量-速度比优势，那自回归架构作为LLM默认范式的地位可能真的会被动摇。

另一个问题是边缘部署。扩散模型的迭代去噪过程在计算pattern上与自回归模型有本质区别——它可能更适合批量并行计算但对内存带宽的要求不同。这对Jetson Orin这类边缘设备上的部署有什么影响，目前还没有公开数据，但值得持续关注。

原文链接: https://www.businesswire.com/news/home/20260224034496/en/

---

## 面试关联知识点

**1. 自回归模型 vs 扩散模型的生成机制区别是什么？**

自回归模型按序列顺序逐token生成，每步条件依赖所有已生成token，时间复杂度与序列长度线性相关。扩散模型从噪声出发，通过多轮去噪迭代并行精炼所有位置的token，单次前向传播可以修改多个token，因此理论吞吐量上限更高。

**2. Speculative Decoding（投机解码）的原理是什么？它和dLLM的加速思路有什么区别？**

Speculative Decoding用一个小的draft model快速生成候选token序列，再用大模型一次性验证/修正，本质上是在自回归框架内用"猜测+验证"减少大模型的前向传播次数。dLLM的加速则来自架构本身——不再逐token生成，而是并行去噪。前者是工程优化，后者是范式变更。

**3. KV Cache在自回归模型中的作用？扩散语言模型还需要KV Cache吗？**

KV Cache缓存已生成token的Key/Value矩阵，避免每步重复计算attention。这是自回归模型inference的核心优化。扩散语言模型的生成方式不同——每轮去噪是对整个序列做全局精炼，不存在"已生成/未生成"的分界，因此传统KV Cache机制不直接适用，但可能有类似的中间状态缓存策略。