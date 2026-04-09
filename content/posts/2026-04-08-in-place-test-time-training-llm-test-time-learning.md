---
title: "In-Place Test-Time Training：让 LLM 在推理时边读边改自己"
date: 2026-04-08T07:30:00+08:00
tags: ["LLM", "test-time-training", "long-context"]
description: "ByteDance Seed 提出 In-Place TTT，在推理阶段只更新局部 MLP fast weights，让 Transformer 能边读上下文边临时吸收新信息。"
showToc: true
---

今天的晨读：In-Place Test-Time Training：让 LLM 在推理时边读边改自己

TL;DR：ByteDance Seed 这篇 4 月 7 日刚挂 arXiv 的工作，核心不是再造一种新架构，而是把 Test-Time Training 直接塞进现有 Transformer 里：在推理阶段只更新 MLP 里的最后投影矩阵，让模型能一边处理长上下文、一边临时吸收新信息。它瞄准的是一个很现实的问题：现在的 LLM 本质上还是“训练完就冻结”，面对持续流入的新信息，只能靠 KV cache 记住，不能真正把信息写进参数。

背景先说清楚。传统 Test-Time Training 以前主要在视觉里玩得多，放到 LLM 上很别扭：一是架构不兼容，二是在线更新太贵，三是优化目标经常不对路。很多方法让模型在测试时做 reconstruction 或 auxiliary objective，但 LLM 真正做的是 next-token prediction，这俩不完全是一回事。于是这篇文章干了件很聪明的事：不碰大部分参数，只把每个 MLP block 里最后那层 projection matrix 当成 fast weights。这样好处很直接——不用从头设计 TTT 模型，也不用大改训练流程，现成 LLM 基本就能“插拔式”升级。

更关键的是它的训练目标。作者没有沿用泛化意义上的重建损失，而是专门为自回归语言建模设计了和 next-token prediction 对齐的目标。你可以把它理解成：模型在读当前 chunk 时，不只是把 token 存进 cache，而是在局部参数里做一个小步更新，让后续 token 预测更贴着当前上下文走。这个思路和单纯拉长 context length 不一样。长上下文本身只是“看得见更远”，In-Place TTT 则是“看完以后顺手记一点”。

工程上它还做了 chunk-wise update，也就是不是每来一个 token 就暴力更新一次，而是按块更新，降低开销，并且兼容 context parallelism。这个细节很重要，因为如果 online update 把吞吐干碎了，那论文再漂亮也没法落地。文中给出的结果是：作为 in-place enhancement，它能让 4B 模型在最长 128k context 的任务上拿到更好的表现；如果从头预训练，效果也稳定优于其他 TTT 路线。

## 为什么这篇值得盯

我觉得这篇最值得盯的，不是它今天就能替代 RAG 或长上下文，而是它把“推理”和“轻量学习”之间那堵墙捅了个洞。现在大家默认的范式还是：参数负责长期知识，KV cache 负责当前会话，外部工具负责额外记忆。In-Place TTT 提出的是第四种可能：模型在推理时拥有受控、局部、临时的可塑性。这对几个方向都很有启发。

第一，对长上下文任务，它可能比一味堆 context 更划算。因为很多信息不是需要被原样保留，而是需要被提炼成对后续预测有用的状态。第二，对边缘部署也有意思。Kevin 这边如果以后在 Jetson 上玩长文档问答，本地小模型最怕的就是上下文长了以后注意力成本爆炸、同时记忆又不牢。如果这种 fast-weight update 能被做成低开销 kernel，价值会比“再塞更长窗口”大得多。第三，它和 RAG 其实不是竞争关系，更像互补：RAG 负责把对的证据捞回来，TTT 负责让模型在当前任务期间真正吸收这些证据。

当然，别过早上头。这里面有几个硬问题还没解决：在线更新带来的稳定性、灾难性覆盖、额外 latency，以及 fast weights 到底学到的是“事实”还是“局部模式”。如果更新策略控制不好，模型可能会被上下文牵着鼻子走，出现短期过拟合。换句话说，这方向很酷，但离“会思考、会学习的在线 LLM”还差不少工程血汗。

原文链接：<https://arxiv.org/abs/2604.06169>
