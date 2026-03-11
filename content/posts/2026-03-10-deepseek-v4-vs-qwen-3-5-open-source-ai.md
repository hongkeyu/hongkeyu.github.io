---
title: "DeepSeek V4 vs Qwen 3.5: 开源模型正在改写游戏规则"
date: 2026-03-10T07:30:00+08:00
tags: [open-source-ai, MoE, LLM]
description: "DeepSeek V4 和 Qwen 3.5 在多个 benchmark 上超越 GPT-5.2，API 成本仅为 OpenAI 的 1/20，开源模型份额一年从 1% 升到 15%。"
showToc: true
---

**TL;DR:** DeepSeek V4（1T参数/32B激活）和 Qwen 3.5（397B/17B激活）在三月初相继发布，多个 benchmark 上超越 GPT-5.2 和 Claude Opus 4.6，API 成本仅为 OpenAI 的 1/20。开源模型从份额 1% 到 15%，只用了 12 个月。

---

## 背景：开源的拐点

2025年1月，OpenAI 占全球 AI 市场的 55%，DeepSeek 和 Qwen 各占 0.5%。一年后，OpenAI 降到 40%，两家合计升到 15%。Qwen 在 HuggingFace 上累计下载超过 7 亿次，成为全球下载量最大的模型家族。

驱动这个转变的核心因素有两个。第一，能力差距被抹平了。DeepSeek V3 用 560 万美元的训练成本（对比 OpenAI/Anthropic 每个前沿模型上亿美元），证明了 Mixture-of-Experts 架构能以极低的计算成本匹配 dense model。第二，生态成熟了。vLLM、Ollama、TensorRT-LLM 都提供 OpenAI 兼容 API，切换成本降到接近零。

## DeepSeek V4: 1T 参数但只激活 32B

V4 的总参数量约 1 万亿，比 V3 增加 50%，但 MoE 架构下每个 token 只激活约 32B 参数——反而比 V3 的 37B 还少。更大的模型，更低的单次推理成本，这就是 MoE 的魅力。

架构上的几个关键创新：

1. 原生多模态。不是在文本模型上拼接 vision adapter，而是从预训练阶段就内置了文本、图像、视频的多模态能力。处理带图表的金融报告、带影像的医疗记录，单次推理搞定，不需要路由到多个子模型。

2. 100 万 token 上下文窗口。从 V3 的 128K 扩展到 1M+，靠两个技术实现：DeepSeek Sparse Attention（DSA）+ Lightning Indexer 把 attention 复杂度从 O(n^2) 降到线性，Engram Conditional Memory 用哈希实现 O(1) lookup。实测可以把整个中型代码库（50-100 个文件）或一份 600 页技术手册一次性喂进去。

3. 训练稳定性。引入 Manifold-Constrained Hyper-Connections，每个 token 走 16 条 expert pathway（V3 是 top-2/top-4 选择）。训练成本虽未官方公布，估计仍在千万美元以内。

4. 硬件适配。V4 针对华为昇腾和寒武纪芯片做了优化，甚至对 NVIDIA 和 AMD 延迟了早期访问。这是个有意为之的架构赌注。

API 定价：输入约 $0.14/M tokens，输出约 $0.28/M tokens，大约是 GPT-5 的 1/20。

## Qwen 3.5: 为 Agent 而生

Qwen 3.5 于 2 月 16 日发布，旗舰版 397B 总参数、17B 激活，比 V4 更精简但优化更激进。

几个值得关注的点：

1. Apache 2.0 协议。这是所有前沿级模型中最宽松的开源协议——商用、修改、微调、卖产品，零法律顾虑。DeepSeek 的自定义许可虽然也宽松但需要法务审查，OpenAI 和 Anthropic 的条款每个季度都在变。对企业来说，Apache 2.0 就是一个已知量。

2. Benchmark 表现。在与生产落地最相关的几个方向上领先：MathVision 88.6（GPT-5.2 83.0），MMMU 多模态理解 85.0（GPT-5.2 83.2），指令遵循 IFBench 76.5（GPT-5.2 75.4），BrowseComp 网页浏览 78.6（GPT-5.2 76.1）。纯数学推理（AIME 2026）和复杂编程（SWE-bench）上仍然是 GPT-5.2 和 Claude 领先，但差距在收窄。

3. 原生 Agent 支持。内置 thinking/non-thinking 两种推理模式，API 层面切换，不需要 prompt engineering。原生 tool use 和多步规划，Tau2-Bench 得分 86.7，仅次于 Claude Opus 4.6。

4. 速度。FP8 原生训练 pipeline + 混合注意力架构（Gated Delta Networks + 标准 gated attention），解码吞吐量是 Qwen3-Max 的 8.6-19 倍，activation memory 减半。

5. 201 种语言支持，模型家族从 0.8B 到 397B 全覆盖。可以用 32B 版本在单卡上开发验证，生产环境再切到 397B。

## 怎么选？

需要超长上下文（100万 token）、强编程、视频多模态 → DeepSeek V4
需要 Agent 能力、多语言、Apache 2.0 法律简洁性、更低自托管门槛 → Qwen 3.5

自托管盈亏平衡点：月处理 1500-4000 万 token 以上就值得自建。低于这个量，直接用 API 已经比 OpenAI 便宜 10-30 倍。

## 对边缘部署的意义

两个模型都采用 MoE 架构，激活参数远小于总参数。Qwen 3.5 的 17B 激活参数 + FP8 pipeline 意味着量化后有可能在消费级硬件上跑。对 Jetson Orin 这样的边缘设备来说，关注 Qwen 3.5 的小尺寸变体（0.8B-32B）在 GGUF 格式下的表现会更有实际价值。

原文链接：https://particula.tech/blog/deepseek-v4-qwen-open-source-ai-disruption
