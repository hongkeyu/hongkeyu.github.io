---
title: "Apple 将在 WWDC 2026 用 Core AI 取代 Core ML：一次命名背后的架构野心"
date: 2026-03-06T07:30:00+08:00
tags: ["edge-ai", "apple", "on-device-inference"]
description: "Bloomberg 爆料 Apple 计划在 WWDC 2026 推出 Core AI 框架取代 Core ML，从模型推理运行时升级为 AI 编排层，支持 MCP 协议和第三方模型集成。"
showToc: true
---

TL;DR: Bloomberg 的 Mark Gurman 爆料，Apple 计划在今年六月的 WWDC 上推出全新的 Core AI 框架，正式取代已经存在多年的 Core ML。这不只是改名，而是 Apple 在端侧 AI 部署策略上的一次重大信号。

---

## 背景：Core ML 走过的路

Core ML 自 2017 年 WWDC 首次亮相以来，一直是 Apple 生态中开发者部署 on-device ML 模型的核心框架。它的设计哲学很明确：把训练好的模型（.mlmodel 格式）塞进 iPhone/iPad/Mac，利用 ANE（Apple Neural Engine）做端侧推理，不需要联网、不需要云端。

过去几年，Core ML 经历了多次升级：支持 Transformer 架构、增加了更细粒度的权重压缩工具（weight compression techniques）、优化了与 Metal 的协作。但本质上，它的定位始终是一个"模型推理运行时"——你在 PyTorch 或 TensorFlow 里训好模型，转换成 Core ML 格式，然后部署。

问题是，2024 年之后的 AI 世界已经不是这个范式了。

## 从 ML 到 AI：不只是换个名字

Gurman 在 Power On 专栏里说了一句关键的话："Apple knows that 'machine learning' is a dated term that no longer resonates with developers or consumers." 但如果你只把这理解成市场营销的品牌刷新，那就低估 Apple 了。

Core ML 的"ML"代表的是一个时代的技术范式：你训练一个分类器、一个检测模型、一个 NLP pipeline，然后把它打包成一个静态 artifact 部署到端侧。这个范式对应的是 pre-LLM 时代的 AI 应用模式。

而 Core AI 要解决的问题显然更大。从目前透露的信息看，至少有两个方向值得关注：

### 1. 集成第三方 AI 模型

Gurman 提到 Core AI 的一个核心目标是"helping developers integrate outside AI models into their apps"。这意味着 Apple 可能在框架层面打通与第三方大模型（比如 Google Gemini，已经确认会给 Siri 供能）的集成接口。

更有意思的是，有消息提到 MCP（Model Context Protocol）可能是一个候选方案。如果 Apple 真的在系统框架层面原生支持 MCP，这对整个 AI Agent 生态的影响是巨大的——意味着 iOS app 可以用标准化协议与各种 AI 后端通信，而不需要每个 app 自己写一套 API 集成逻辑。

### 2. Apple Foundation Models 的开发者接口

Apple 在 OS 26 时代就已经开始让开发者使用其自研的 Apple Foundation Models 做端侧文本生成等任务。Core AI 很可能会把这个能力系统化，提供更统一的 API surface，让开发者既能调用 Apple 自己的端侧模型，也能无缝切换到云端或第三方模型。

这个思路其实和 Android 的 AI Core / ML Kit 的演进方向类似，但 Apple 的优势在于它对硬件（ANE、GPU、内存带宽）的垂直整合控制力远强于 Android 阵营。

## 对边缘部署的启示

从做端侧部署的角度看，这件事有几个值得思考的点：

第一，Apple 正在把"端侧 AI"从"跑一个模型"升级为"跑一个 AI 系统"。传统的 Core ML 部署，你操心的是模型量化、ANE 兼容性、latency。而 Core AI 时代，你可能需要考虑的是：端侧模型和云端模型如何协同、context 如何在不同模型间传递、tool use 如何在端侧实现。

第二，Apple 用 A18 Pro（一颗手机芯片）驱动刚发布的 MacBook Neo，起价 599 美元。这说明 Apple 对移动级芯片跑 AI workload 的信心非常足。配合 Core AI 框架，Apple 正在构建一个从 iPhone 到 Mac 统一的 AI runtime 层。

第三，这对 Jetson 这类非 Apple 的边缘平台意味着什么？Apple 的垂直整合是其他平台学不来的，但 Core AI 背后的设计理念——统一的端侧 AI 框架、标准化的模型集成协议、端云协同——是通用的。NVIDIA 的 Jetson 生态如果不在 TensorRT 之上提供类似的高层抽象，在开发者体验上会越来越落后。

## 延伸思考

这件事最有意思的地方不在技术本身，而在 Apple 对"AI 开发者体验"这件事的重新定义。过去十年，Apple 的 ML 故事是"你训好模型，我帮你部署"。现在变成了"你告诉我要什么 AI 能力，我帮你编排模型"。这是一个从 inference runtime 到 AI orchestration layer 的跨越。

如果 Core AI 真的在 WWDC 上落地，并且支持 MCP 或类似协议，那今年下半年 iOS 生态的 AI Agent 应用可能会迎来一波爆发。

## 面试关联知识点

**Q: 模型量化的主要方法有哪些？Core ML 支持哪些压缩技术？**

Core ML Tools 支持 weight pruning、palettization（调色板量化，将权重聚类到有限码本）和 linear quantization（INT8/INT4）。与 GPTQ/AWQ 等 LLM 专用量化不同，Core ML 的量化更偏向 CNN/Transformer 通用场景，且需要考虑 ANE 硬件对特定 bit-width 的支持。

**Q: KV Cache 在端侧推理中为什么重要？**

LLM 推理的 decode 阶段是 memory-bound 的，每生成一个 token 都需要读取所有历史 key/value。KV Cache 避免重复计算，但代价是显存占用随序列长度线性增长。在端侧内存有限的场景下，KV Cache Quantization（比如 FP8 或 INT4 KV Cache）和 GQA（Grouped Query Attention，减少 KV head 数量）是关键优化手段。

**Q: MCP（Model Context Protocol）是什么？**

Anthropic 提出的开放协议，定义了 AI 模型与外部工具/数据源之间的标准化通信接口。核心思想是让 AI Agent 能通过统一的协议调用不同的 tool server，而不需要为每个工具写专门的集成代码。如果 Apple 在系统层面支持 MCP，意味着 iOS app 可以声明式地暴露 AI 能力，由系统框架负责路由和调度。

---

原文链接：
- https://9to5mac.com/2026/03/01/apple-replacing-core-ml-with-modernized-core-ai-framework-for-ios-27-at-wwdc/
- https://www.macrumors.com/2026/03/01/another-ios-27-change-leaked/
