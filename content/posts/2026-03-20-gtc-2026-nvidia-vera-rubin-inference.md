---
title: "GTC 2026: NVIDIA 的 Inference is King 宣言与 Vera Rubin 全栈平台"
date: 2026-03-20T07:30:00+08:00
tags: ["inference", "nvidia", "hardware"]
description: "NVIDIA GTC 2026 明确推理为 AI 重心，发布 Vera Rubin 全栈平台、收编 Groq LPU、预告 Feynman 架构路线图"
showToc: true
---

TL;DR — NVIDIA GTC 2026 本周在 San Jose 落幕，Jensen Huang 用两小时 keynote 明确了一个判断：AI 的重心已从训练转向推理，token 是新的大宗商品，数据中心是生产 token 的工厂。围绕这个判断，NVIDIA 发布了 Vera Rubin 全栈平台（7 芯片 + 5 机架 + 1 超算）、收编 Groq 的 LPU 推理芯片、以及下一代 Feynman 架构路线图。

---

## 背景：为什么推理突然变得这么重要

过去三年 AI 基础设施的叙事一直围绕训练：更大的集群、更多的 GPU、更长的训练时间。但 2025-2026 年发生了一个结构性转变——Agentic AI 的爆发。当 AI 不再只是回答一个问题，而是拆解任务、调用工具、生成子 agent、持续运行时，推理侧的 token 消耗量呈指数级增长。Jensen 在 keynote 中说"过去几年计算需求增长了 100 万倍"，这个数字主要来自推理侧。

换句话说：训练是一次性成本（虽然很贵），但推理是持续运营成本，而且随着 agent 架构的普及，推理量远超训练量。这就是为什么 Jensen 自豪地引用分析师的话称 NVIDIA 是 "the inference king"。

## Vera Rubin 平台：不只是一块 GPU

Vera Rubin 不是单芯片发布，而是一个完整的 rack-scale 系统，包含七个核心组件：

- **Vera Rubin NVL72 GPU** — 下一代 GPU 机架，推理吞吐比 Grace Blackwell 提升 10 倍/瓦
- **Vera CPU** — 新一代 CPU，比传统 rack-scale CPU 快 50%，效率翻倍。单机架支持 22,500+ 并发 CPU 环境，专门为 agentic workload 中的工具调用和编排设计
- **Groq 3 LPU** — 来自 NVIDIA 去年 200 亿美元授权协议收编的 Groq 推理芯片。256 块 LPU 组成 LPX Rack，可将 Rubin GPU 的 tokens/watt 提升 35 倍
- **BlueField-4 STX** — AI-native 存储架构，专门优化 LLM 推理中的 KV Cache 存储与检索
- **Spectrum-6 SPX 以太网、NVLink 6 Switch、ConnectX-9 SuperNIC** — 网络层

关键设计理念是"extreme codesign"——软硬件协同设计。Vera Rubin 不是把一堆组件拼在一起，而是从芯片到网络到软件栈作为一个整体优化。结果是：训练同等规模的 MoE 模型只需 Blackwell 平台 1/4 的 GPU 数量，推理的 cost per token 降到 1/10。

## Groq 收编的深意

去年 12 月 NVIDIA 和 Groq 签了 200 亿美元的授权协议，这次 GTC 正式亮相。Groq 的 LPU 架构本质上是确定性计算（deterministic compute）——没有缓存层级，数据流完全可预测，延迟极低。NVIDIA 的做法是把 LPU 作为 GPU 的推理加速器，GPU 负责灵活的通用计算（prefill、复杂推理），LPU 负责高吞吐的 token 生成（decode 阶段）。这种分工非常合理：prefill 是 compute-bound，decode 是 memory-bandwidth-bound，两种架构各取所长。

35x tokens/watt 的提升数字惊人，但需要注意这是 LPX Rack 相对于纯 GPU 推理的对比，实际部署中的提升取决于 workload 特征（batch size、sequence length、模型架构）。

## Feynman 路线图：Rosa CPU + LP40 LPU

Jensen 还预告了 Vera Rubin 之后的下一代架构 Feynman，包含新 CPU Rosa（致敬 Rosalind Franklin）和下一代 LP40 LPU，以及 BlueField-5 和 CX10 网络组件。值得关注的是 Kyber 互联技术同时支持铜缆和 co-packaged optics（共封装光学），后者是解决数据中心带宽瓶颈的关键技术方向。

## 太空数据中心：Vera Rubin Space-1

一个有趣的发布：NVIDIA 宣布将计算平台推向太空。Vera Rubin Space-1 模块基于 IGX Thor 和 Jetson Orin，针对尺寸、重量和功耗受限的环境做了专门工程化。合作伙伴包括 Starcloud、Axiom Space、Planet。Jensen 承认还有辐射、散热等工程挑战，但方向已经明确——这对做边缘部署的人来说是一个信号：极端环境下的推理优化会是一个持续的工程方向。

## 对从业者的意义

1. Inference 优化是当前最热的工程方向。不只是模型量化，而是从芯片架构（LPU vs GPU 分工）到存储（KV Cache 专用存储层）到网络的全栈优化
2. Agentic AI 正在重新定义基础设施需求。22,500 并发 CPU 环境的设计目标说明 NVIDIA 在赌 agent 工具调用和编排会成为主流 workload
3. 边缘与极端环境部署（Jetson Orin 进太空）表明推理侧的"小型化 + 低功耗"需求在扩大

原文链接：
- https://blogs.nvidia.com/blog/gtc-2026-news/
- https://indianexpress.com/article/technology/artificial-intelligence/every-major-reveal-nvidia-gtc-2026-jensen-huang-keynote-10586373/

---

## 面试关联知识点

### 1. KV Cache 原理及优化

Autoregressive decoding 每步生成一个 token，如果不缓存之前的 Key/Value，每步都要重新计算所有 token 的 attention，复杂度 O(n^2)。KV Cache 将已计算的 K/V 存下来，decode 阶段每步只需计算新 token 的 Q 与缓存的 K/V 做 attention，复杂度降到 O(n)。代价是显存占用随序列长度线性增长——这正是 BlueField-4 STX 存储架构要解决的问题：把 KV Cache 从 GPU HBM 扩展到高带宽共享存储层。KV Cache Quantization（INT8/INT4）可以在精度损失可控的前提下减少 50-75% 显存占用。

### 2. Prefill vs Decode 阶段区别

Prefill（prompt processing）：一次性处理整个输入序列，计算量大但高度并行，是 compute-bound。Decode（token generation）：逐 token 生成，每步只计算一个 token，计算量小但受限于内存带宽（需要读取整个 KV Cache），是 memory-bandwidth-bound。GTC 发布的 GPU + LPU 分工正是基于这个区别：GPU 擅长 prefill 的并行计算，LPU 的确定性数据流架构擅长 decode 的高吞吐低延迟生成。

### 3. 模型量化与推理部署

量化核心思想：用低精度数据类型（INT8/INT4/NF4）表示原本 FP16/FP32 的权重和激活值。主要方法分 PTQ（Post-Training Quantization，训练后量化）和 QAT（Quantization-Aware Training，训练中量化）。GGUF 格式是 llama.cpp 生态的标准量化格式，支持 Q2_K 到 Q8_0 等多种精度。量化的本质 trade-off：精度 vs 速度/显存。在 Jetson Orin 等边缘设备上，INT4/INT8 量化 + TensorRT 是标准部署路径。Jensen 提到的"cost per token 降到 1/10"背后就是量化 + 架构优化的综合效果。
