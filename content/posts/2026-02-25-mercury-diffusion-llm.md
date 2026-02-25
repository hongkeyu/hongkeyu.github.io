---
title: "Mercury：把图像扩散模型的范式搬到文本生成，从架构层面打破自回归速度瓶颈"
date: 2026-02-25T12:30:00+08:00
tags: ["Diffusion LLM", "推理优化", "Transformer", "Mercury"]
description: "Inception Labs 的 Mercury 论文解读：第一个商业级扩散语言模型如何在 H100 上实现 1109 tokens/s，以及扩散模型相比自回归的结构性优劣。"
showToc: true
---

## 为什么需要新范式

自回归（AR）模型的推理瓶颈在 KV Cache 文章里已经解释过：Decode 阶段每步只生成一个 token，GPU 大部分时间在等内存搬运数据，算力严重浪费。过去几年的优化方向——Speculative Decoding、KV Cache 压缩、量化——都是在 AR 框架内做局部改良，收益递减。

Mercury 的思路是：既然 AR 的串行瓶颈是结构性的，不如换掉生成算法本身。

## 核心机制：离散扩散

Mercury 的生成过程和图像扩散模型（Stable Diffusion、DALL-E）原理相同，但作用在离散 token 空间上。

**前向过程（加噪）：** 取一段干净文本 x，逐步用随机 token 替换其中的内容，直到变成完全随机的 token 序列 z_T。

**反向过程（去噪）：** 从随机噪声 z_T 出发，用一个 Transformer 网络预测"如果把这些噪声去掉，原始文本最可能是什么"，反复迭代直到输出干净文本。

训练目标：

```
L(x) = -E_t[ γ(t) · E_{z_t ~ q} log p_θ(x | z_t) ]
```

关键区别：AR 模型的一次前向传播只产出一个 token 的信息量；扩散模型的一次前向传播同时修改所有位置的 token。假设去噪迭代 T 步，输出长度为 N，AR 需要 N 次前向传播，扩散模型只需要 T 次——T 通常远小于 N。

**架构仍是 Transformer**——扩散是训练和推理算法，不是网络架构。可以直接复用 FlashAttention、混合精度训练、张量并行等优化。但 Attention 模式不同：AR 用 causal mask（下三角矩阵），扩散 Transformer 需要双向 attention。

## 速度数据

Mercury Coder 两个版本，在 H100 上的表现：

- Mini：1109 tokens/s，质量对标开源 speed-optimized 模型
- Small：737 tokens/s，质量对标商业 frontier 模型

同质量 AR 模型通常 70-200 tokens/s。加速来源：扩散模型的每步前向传播涉及所有 token，arithmetic intensity 远高于 AR decode，GPU 利用率接近 prefill 水平。

Copilot Arena（真实开发者盲测）排名质量第二、速度第一。

## 结构性弱点

**1. 短输出场景没有优势。** 扩散模型每次生成固定长度 block，6 个 token 的回答仍需 8-12 轮迭代，反而更慢。

**2. 长上下文更贵。** 双向 attention 每步对整个输出 block 做全量计算，而 AR decode 每步只算一行。128K 上下文下 prefill 代价更高。

**3. 质量天花板尚未证明。** Mercury 2 在 GPQA Diamond 上 74 vs Gemini 3 Flash 的 90。扩散语言模型的 scaling behavior 基本是空白。

**4. 生态工具链缺失。** LoRA、RLHF/DPO、GGUF 量化、KV Cache 优化、Speculative Decoding——整个 AR 推理工具链不适用。

**5. 流式输出不友好。** 需要整个 block 去噪完毕才输出，用户体验是"等→大段出现"而非"打字机"。

**6. 参数量和训练细节未公开。** 没有参数量、层数、hidden dim、去噪步数等具体数字。

## 开源替代和学术基础

- **MDLM**（Masked Diffusion Language Model）：Mercury 的学术基础
- **LLaDA**（arxiv 2502.09992）：8B 参数级别，有开源权重和代码

## 延伸思考

扩散和 Transformer 是正交的。未来 Mamba、RWKV 等线性注意力架构也可以和扩散算法结合——用线性注意力解决双向 attention 的 O(n²) 问题，同时保留并行生成优势。

扩散模型天然支持 infilling——把前缀和后缀作为条件固定，只对中间部分去噪。对代码编辑场景特别有用。

---

*原文：[arxiv 2506.17298](https://arxiv.org/abs/2506.17298)*
*相关：[LLaDA](https://arxiv.org/abs/2502.09992) | [MDLM](https://arxiv.org/abs/2406.07524)*
