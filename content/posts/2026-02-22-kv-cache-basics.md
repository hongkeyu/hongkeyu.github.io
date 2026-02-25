---
title: "KV Cache 从零讲清楚"
date: 2026-02-22T07:30:00+08:00
tags: ["LLM", "推理优化", "KV Cache", "Transformer"]
description: "从自回归生成的基本问题出发，解释 KV Cache 是什么、为什么成为长上下文瓶颈，以及 Attention Matching 压缩方案的思路。"
showToc: true
---

## 一、先从 Transformer 的一次前向传播说起

最简单的情景：你给模型输入一句话「今天天气很好」，想让它续写下一个字。

Transformer 内部对输入的每个 token（字）做三件事：生成三个向量 Q（Query）、K（Key）、V（Value）。然后用 Attention 机制让每个 token 去"看"其他所有 token——用自己的 Q 去和其他人的 K 做点积算相关度，再用这个相关度去加权所有人的 V，拿到自己的输出。

这是 Self-Attention 的核心运算：

```
输出 = softmax(Q · K^T / √d) · V
```

理解这一步的关键是：每个 token 的输出是由「它自己的 Q」和「所有其他 token 的 K、V」共同决定的。

## 二、自回归生成的问题

生成文字时，模型每次只输出一个 token，然后把这个 token 加到输入里，再生成下一个。比如：

```
输入：今天天气很好         → 生成：，
输入：今天天气很好，       → 生成：适
输入：今天天气很好，适     → 生成：合
输入：今天天气很好，适合   → 生成：出
...
```

现在问题来了：生成「适」的时候，模型需要重新计算所有 token 的 K、V 向量——尽管这些 K、V 和上一步生成「，」时完全一样，因为这些 token 没有变。

这是纯粹的浪费。每多生成一个 token，就要把前面所有 token 的 K、V 全部重新算一遍。如果已经生成了 1000 个 token，生成第 1001 个时要做 1000 次多余的 K、V 计算。

## 三、KV Cache 的解决方案

思路很直接：既然前面 token 的 K、V 不会变，算一次之后就存起来。下次生成新 token 时，直接取出来用，不重算。

这就是 KV Cache——一个内存里的"K/V 仓库"，每生成一个新 token，就往仓库里加入这个 token 的 K 和 V，之后永远不用再算它们了。

有了 KV Cache，生成第 1001 个 token 的计算量和生成第 2 个 token 完全相同——只需要算当前这一个新 token 的 K、V，然后和仓库里已有的所有 K、V 做 Attention。

## 四、KV Cache 为什么会成为瓶颈

好，省了大量计算，代价是内存。来算一下实际数字：

以 LLaMA-3 70B 为例：

- 80 层 Transformer
- 每层每个 token 存 K 和 V，各是一个 [num_kv_heads × head_dim] 的向量
- 用 GQA（Grouped Query Attention），KV heads = 8，head_dim = 128
- FP16 存储（每个数 2 字节）

一个 token 的 KV 大小 = 2（K+V）× 8（heads）× 128（dim）× 80（层）× 2（bytes）= **320 KB**

| 序列长度 | KV Cache 大小 |
|---------|-------------|
| 4K tokens | 1.2 GB |
| 32K tokens | 9.8 GB |
| 128K tokens | 39 GB |

一张 80GB A100 跑 128K 上下文，光 KV Cache 就占掉近一半。如果同时处理多个用户请求（batch size > 1），直接爆显存。

## 五、Decode 阶段为什么特别慢

有了 KV Cache 之后，推理时有两个阶段：

**Prefill**（处理输入 prompt）：所有 token 并行计算，GPU 全速运转，效率高。

**Decode**（逐 token 生成）：每步只有一个新 token，但要从内存里读出所有历史 token 的 K、V，算完再写回去。

问题是：现代 GPU 的计算速度远远快于内存读写速度。Decode 阶段每步实际用于"算"的时间极短，大部分时间花在"从内存搬运 KV Cache 数据"上。GPU 在等内存，算力严重浪费。这叫 **memory-bandwidth-bound**。

生成 1000 个 token，就要把整个 KV Cache 搬进搬出 1000 次。序列越长，每次搬运的数据越多，速度越慢。

## 六、连接到 Attention Matching 压缩论文

现在你就能理解 Attention Matching（arxiv: 2602.16284）在解决什么问题了。

传统压缩方法在 50× 压缩比下精度太差，原因是丢失了 token 之间 attention 的相对关系。Attention Matching 的做法是：不直接压缩权重，而是找一组新的 (C_k, C_v)，让它们产生的 attention 输出和原始 KV Cache 产生的 attention 输出尽量相同。

这个"找"的过程是个最小二乘问题，有封闭解（不用梯度下降），所以几秒就算完。

打个比方：原来你有一本 1000 页的书，读完要 10 小时。Attention Matching 说：我能用 20 页的摘要，让你读完后知道的关键内容和读完 1000 页一样多——而且这 20 页摘要几秒就能生成，不用几小时。

那个 scalar bias β 是额外的修正项：压缩成 20 页后章节间的引用关系会系统性失真，β 就是补偿这个失真的。

---

*相关资料：Attention Matching 原论文 <https://arxiv.org/abs/2602.16284>*
