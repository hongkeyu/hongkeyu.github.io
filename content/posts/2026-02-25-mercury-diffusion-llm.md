---
title: "Mercury：把图像扩散模型的范式搬到文本生成"
date: 2026-02-25T12:30:00+08:00
tags: ["Diffusion LLM", "推理优化", "Transformer", "LLM"]
description: "Inception Labs 发布第一个商业级扩散语言模型 Mercury Coder，在 H100 上实现 1109 tokens/s，比同质量自回归模型快 3-10 倍。架构仍是 Transformer，但生成算法从逐 token 预测切换为迭代去噪。"
showToc: true
---

## 为什么需要新范式

自回归（AR）模型的推理瓶颈在 KV Cache 文章里已经解释过：Decode 阶段每步只生成一个 token，GPU 大部分时间在等内存搬运数据，算力严重浪费。过去几年的优化方向——Speculative Decoding、KV Cache 压缩、量化——都是在 AR 框架内做局部改良，收益递减。

Mercury 的思路是：既然 AR 的串行瓶颈是结构性的，不如换掉生成算法本身。

## 核心机制：离散扩散

Mercury 的生成过程和图像扩散模型（Stable Diffusion、DALL-E）原理相同，但作用在离散 token 空间上。

**前向过程（加噪）：** 取一段干净文本 x，逐步用随机 token 替换其中的内容，直到变成完全随机的 token 序列 z_T。这个过程定义了一条从"有意义的文本"到"纯噪声"的 Markov 链。

**反向过程（去噪）：** 从随机噪声 z_T 出发，用一个 Transformer 网络预测"如果把这些噪声去掉，原始文本最可能是什么"，然后根据预测结果部分去噪，得到 z_{T-1}，反复迭代直到输出干净文本。

训练目标是让模型学会在任意噪声水平下还原原始文本：

```
L(x) = -E_t[ γ(t) · E_{z_t ~ q} log p_θ(x | z_t) ]
```

γ(t) 是噪声级别的权重函数，控制模型在不同去噪阶段的学习侧重。

关键区别在于：AR 模型的一次前向传播只产出一个 token 的信息量；扩散模型的一次前向传播同时修改所有位置的 token，"coarse-to-fine"逐步精炼。假设去噪迭代 T 步，输出长度为 N，那 AR 需要 N 次前向传播，扩散模型只需要 T 次——而 T 通常远小于 N（论文没有公布具体 T 值，但从速度推测大约 8-16 步）。

## 架构选择：仍然是 Transformer

这一点很重要：扩散是训练和推理算法，不是网络架构。Mercury 的骨架仍然是标准 Transformer，因此可以直接复用现有的 FlashAttention、混合精度训练、张量并行等工程优化。这大幅降低了部署门槛——不需要重写推理栈。

但 Attention 模式不同。AR Transformer 用 causal mask（下三角矩阵），每个 token 只看前面的 token。扩散 Transformer 需要双向 attention——去噪时每个位置都要看全局信息才能做出最优修改。这意味着 prefill 阶段的计算量更大。

## 速度数据

Mercury Coder 两个版本，在 H100 上的表现：

- Mini：1109 tokens/s，质量对标开源 speed-optimized 模型
- Small：737 tokens/s，质量对标商业 frontier 模型

对比参考（Artificial Analysis 独立测试）：同质量的 AR 模型通常在 70-200 tokens/s 范围。Mercury 的加速不靠 batch size 堆吞吐量，而是单用户延迟的真实改善。

论文明确指出速度来源：扩散模型的每步前向传播涉及所有 token 的计算，arithmetic intensity（计算/访存比）远高于 AR decode，GPU 利用率接近 prefill 水平。同样的硬件，扩散模型让 GPU 真正在"算"而不是在"等"。

Copilot Arena（真实开发者盲测）中 Mercury Coder 排名质量第二、速度第一。

## 结构性弱点：扩散模型不是银弹

论文对劣势几乎没有讨论，但从架构分析和外部评测可以总结出几个真实问题：

**1. 短输出场景没有优势。** 扩散模型每次生成固定长度的 block，不管实际需要多少 token。如果只需 6 个 token（分类、简短回答），AR 模型 6 步完成，扩散模型可能仍需 8-12 轮迭代去噪一个完整 block，反而更慢。

**2. 长上下文更贵。** 双向 attention 的计算量是 O(n²)，AR 在 decode 时每步只算新 token 对全部历史 token 的 attention（一行），而扩散模型每步要对整个输出 block 做全量 attention（整个矩阵）。128K 上下文下 prefill 代价更高。

**3. 质量天花板尚未证明。** Mercury 2 在 GPQA Diamond 上 74 vs Gemini 3 Flash 的 90。AR 模型有极其成熟的 scaling law，扩散语言模型的 scaling behavior 基本是空白。

**4. 生态工具链缺失。** LoRA 微调、RLHF/DPO 对齐、GGUF 量化、KV Cache 优化、Speculative Decoding——整个 AR 推理工具链不适用于扩散模型。

**5. 流式输出不友好。** AR 模型生成一个 token 立刻返回，用户看到"打字机"效果。扩散模型需要整个 block 去噪完毕才能输出有意义的文本。

**6. 参数量和训练细节未公开。** 论文只说了"trillions of tokens"训练数据和两个模型尺寸（Mini/Small），没有给出参数量、层数、hidden dim、去噪步数等任何具体数字。

## 开源替代和学术基础

Mercury 闭源，但这个方向有两个重要的开源项目：

- **MDLM**（Masked Diffusion Language Model）：Kuleshov 的前期工作，Mercury 论文直接引用为基础方法。学术级别实现，适合理解核心数学。
- **LLaDA**（Large Language Diffusion with mAsking, arxiv 2502.09992）：用 masking 代替加噪作为前向过程，8B 参数级别，有开源权重和代码。目前最接近"可复现的大规模扩散语言模型"的项目。

## 延伸思考

Mercury 论文最有价值的一个观察：扩散和 Transformer 是正交的。Transformer 是网络架构，扩散是训练/推理算法，两者可以自由组合。这意味着未来 Mamba、RWKV 等线性注意力架构也可以和扩散算法结合——用线性注意力解决双向 attention 的 O(n²) 问题，同时保留扩散的并行生成优势。

另一个未被充分讨论的方向：扩散模型天然支持 infilling（在给定前缀和后缀之间填充内容），因为去噪过程可以把前缀和后缀作为条件固定不动，只对中间部分做去噪。这对代码编辑场景特别有用——不需要重新生成整个文件，只修改变动的部分。

---

*原文：[arxiv 2506.17298](https://arxiv.org/abs/2506.17298)*
*相关项目：[LLaDA](https://arxiv.org/abs/2502.09992) · [MDLM](https://arxiv.org/abs/2406.07524)*
