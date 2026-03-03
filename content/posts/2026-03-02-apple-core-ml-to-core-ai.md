---
title: "Apple Core ML 到 Core AI：端侧推理框架的范式转移"
date: 2026-03-02T07:30:00+08:00
tags: ["on-device-ai", "apple", "quantization"]
description: "Apple 将在 WWDC 2026 用 Core AI 替代 Core ML，配合 Foundation Models 框架构建双层端侧 AI 体系，从 BYO 模式向平台内置大模型演进。"
showToc: true
---

TL;DR: Mark Gurman 爆料 Apple 将在 WWDC 2026 用全新的 Core AI 框架替代 Core ML，配合已有的 Foundation Models 框架（端侧 3B LLM），Apple 正在从"你自带模型跑推理"向"平台内置大模型 + 你自带小模型"的双层架构演进。这对端侧部署的思路有直接启发。

---

## 背景：Core ML 走过的九年

Core ML 从 iOS 11（2017年）开始服务开发者，定位很清晰：一个通用的端侧推理引擎。你用 PyTorch/TensorFlow 训练模型，转换成 .mlmodel 格式，Core ML 负责调度 Neural Engine、GPU、CPU 跑推理。支持图像分类、目标检测、NLP、音频分析、姿态估计等等。

它的核心特征是：框架本身不带模型，开发者 BYO（Bring Your Own）。这跟 TensorRT 的定位类似——你负责模型，我负责高效执行。

但从 iOS 26 开始，Apple 加了一个新层：Foundation Models 框架。这个框架直接暴露了 Apple Intelligence 内置的约 3B 参数 LLM 给第三方开发者。不需要带模型，不需要训练，调几行 Swift 代码就能做文本生成、摘要、结构化抽取、tool calling。模型随系统更新自动升级，推理完全免费。

## Core AI 的信号：不只是改名

Gurman 的原话是："The switch from 'ML' to 'AI' is significant. Apple knows that 'machine learning' is a dated term that no longer resonates with developers or consumers."

表面上看这是一次品牌升级，但结合 Foundation Models 框架的存在，背后的架构意图更值得关注。Apple 正在构建一个双层端侧 AI 体系：

**第一层：Core AI（原 Core ML）—— 感知层。** 处理图像、音频、传感器数据等结构化 ML 任务。你自带模型，框架负责高效推理。支持 Neural Engine 调度，支持量化压缩（Core ML Tools 提供 weight compression），30+ fps 实时推理。这一层面向的是确定性、窄任务、高帧率场景。

**第二层：Foundation Models —— 推理和语言层。** Apple 内置的 3B LLM，专门处理文本理解和生成。支持 @Generable 宏直接从自由文本抽取 Swift 强类型结构体，支持 tool calling（模型自己决定何时调用 app 数据）。不需要开发者训练任何东西。

Apple 自己展示了一个典型的 Combination Pattern：SwingVision（网球教练 app）用 Core ML 逐帧分析挥拍动作，抽取结构化的运动数据，然后把这些数据喂给 Foundation Models 生成自然语言的教练反馈。感知层 + 推理层分工明确。

### 为什么这件事重要

这个双层架构其实代表了端侧 AI 的一个趋势：平台厂商开始把"通用语言智能"作为系统级能力内置，开发者只需要关注自己领域的专用模型。类似于操作系统内置了 TCP/IP 栈，你不需要自己写网络协议。

对比 Jetson 生态：NVIDIA 的路线是 TensorRT + 各种 SDK（Deepstream、Isaac 等），本质上还是纯 BYO 模式。开发者要自己处理模型选择、量化、部署全链条。Apple 的做法是在 BYO 之上叠加一个平台级 LLM，降低了"加一点智能"的门槛。

这也解释了为什么 Apple 敢把这个 3B 模型的推理设为免费——它不是产品，是平台基础设施。就像 Metal 不收费一样。

## 技术细节值得注意的几点

1. Foundation Models 的 3B LLM 有明确的能力边界：Apple 官方说它不适合 world-knowledge Q&A、代码生成、复杂数学。它被优化为 task-oriented、app-integrated 的场景。这意味着 Apple 对模型做了大量的 domain-specific distillation，而不是追求通用能力。

2. 设备门槛不低：Foundation Models 需要 iPhone 15 Pro 以上（A17 Pro），7GB 存储空间，用户主动开启 Apple Intelligence。Core ML 则向下兼容到 A13（iPhone 11）。这个分层本身说明 3B 模型跑在端侧的硬件要求仍然不可忽视。

3. Core ML Tools 的量化支持越来越精细："more granular and composable weight compression techniques"。这和 llama.cpp / GGUF 社区做的事情方向一致——在模型质量和推理效率之间找最优量化策略。

4. Foundation Models 的 tool calling 能力值得关注。模型可以自主决定何时调用 app 的数据接口，这本质上是把 Agent 能力下沉到了端侧。对比云端 Agent 方案（ReAct / AutoGen），端侧 Agent 的延迟更低、隐私更好，但推理能力受限于 3B 参数量。

## 延伸思考

如果 Apple 在 WWDC 2026 真的推出 Core AI，下一个问题是：Core AI 会不会允许开发者加载自己的 LLM？目前 Foundation Models 只暴露 Apple 自己的 3B 模型。如果 Core AI 统一了传统 ML 推理和 LLM 推理的接口，同时支持开发者 BYO LLM（比如量化后的 7B 模型），那就是真正的端侧 AI 操作系统化。

原文链接：
- https://9to5mac.com/2026/03/01/apple-replacing-core-ml-with-modernized-core-ai-framework-for-ios-27-at-wwdc/
- https://dev.to/arshtechpro/core-ml-vs-foundation-models-which-should-you-use-3jo0

---

## 面试关联知识点

### 1. 模型量化（Quantization）

端侧设备内存和算力有限。量化将 FP32/FP16 权重压缩到 INT8/INT4 甚至更低位宽，减少模型体积和推理延迟。主要分 PTQ（训练后量化）和 QAT（量化感知训练）。PTQ 简单但精度损失大，QAT 在训练时模拟量化误差，精度更好。Core ML Tools 的 weight compression 和 GGUF 的 Q4_K_M 都属于 PTQ。

### 2. KV Cache 与端侧 LLM 推理的内存瓶颈

3B 模型跑端侧推理时，KV Cache 是主要内存消耗源。每个 token 生成都需要缓存之前所有 token 的 Key/Value。优化方向包括 GQA（减少 KV head 数量）、KV Cache Quantization、Sliding Window Attention。

### 3. Tool Calling 的实现原理

LLM 的 tool calling 本质上是在 SFT 阶段训练模型输出特定格式的函数调用 token，由外部 runtime 解析并执行。关键挑战是让模型稳定输出合法格式——常用方案包括 Grammar-constrained decoding（用 CFG 约束采样空间）和 fine-tuning on tool-use datasets。
