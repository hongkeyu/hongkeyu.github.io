---
title: "2026年LLM推理引擎六大选手横评：vLLM、TensorRT-LLM、SGLang、TGI、llama.cpp、Ollama"
date: 2026-03-13T07:30:00+08:00
tags: ["LLM推理", "MLOps", "模型部署"]
description: "推理引擎的选择往往比模型本身更影响延迟、成本和可扩展性。2026年3月格局：vLLM当通用主力，TensorRT-LLM拼极限吞吐，SGLang在结构化输出和RAG场景异军突起，llama.cpp继续统治边缘端。"
showToc: true
---

选模型是选智商，选推理引擎是选体能。同一个Llama-3 70B，跑在不同引擎上，吞吐量可以差3-5倍，而部署复杂度和硬件锁定程度也天差地别。n1n.ai今天发了一篇比较详细的横评，正好做个梳理。

## vLLM：稳定的工业主力

vLLM目前仍是生产环境的默认选择。它的核心贡献是PagedAttention——借鉴操作系统虚拟内存的思路，让KV Cache可以非连续存储，大幅减少显存碎片，从而支持更大的batch size。

v0.7.3的几个关键更新：一是支持NVIDIA Blackwell（B200）架构优化；二是自动FP8权重校准，针对H100 Hopper GPU可以在几乎不损失精度的情况下砍掉一半显存占用；三是架构重构为更模块化的v1 Engine，开始支持AMD Instinct和AWS Inferentia。

在H100上跑Llama-3 70B，vLLM的吞吐大约在1000-2000 tok/s范围。不是最快的，但胜在开箱即用——直接加载HuggingFace模型，不需要额外编译步骤。

## TensorRT-LLM：NVIDIA亲儿子的速度优势

TensorRT-LLM本质上是一个深度学习编译器，把PyTorch模型翻译成高度优化的CUDA Graph。在高并发场景下，它比vLLM快30-50%，吞吐可以到2500-4000+ tok/s。Perplexity等高流量平台用的就是这个。

代价是什么？部署复杂度高得多。你需要一个显式的build阶段，把模型编译成引擎文件，而且这个过程是硬件绑定的——A100编译的引擎不能直接在H100上跑。对于需要频繁切换模型的场景，这个overhead很痛苦。

这个引擎和Jetson上的TensorRT是同一套技术栈的延伸。TensorRT-LLM在数据中心端做的事情，和在Jetson Orin上用TensorRT做模型优化是一脉相承的，只不过规模和目标不同。

## SGLang：最值得关注的挑战者

SGLang来自UC Berkeley，目前是这个领域最有意思的项目。它的核心创新是RadixAttention——把KV Cache组织成一棵基数树（Radix Tree），实现极高效的前缀缓存。

这在RAG和多轮对话场景下意义重大：假设十个用户都在问同一篇10000字文档的问题，SGLang只需要处理这10000字一次，后续请求直接复用缓存的KV。传统引擎每次请求都要重新计算这些前缀token的attention，SGLang直接跳过。

另一个亮点是constrained decoding的优化。如果你需要LLM严格输出JSON格式（比如Agent的tool call），SGLang的runtime对这类「受约束解码」任务做了专门优化。v0.4.3新增了异步constrained decoding，进一步降低延迟。

在双H100上跑DeepSeek-R1的测试中，SGLang在多轮对话场景比vLLM快10-20%。

## llama.cpp + Ollama：边缘端之王

llama.cpp的定位完全不同——它让LLM跑在一切设备上，从树莓派到Mac Studio。GGUF格式和它的量化方法（4-bit、2-bit、甚至1.5-bit ternary weights）已经成为边缘部署的事实标准。

3月的一个重要更新：llama.cpp合并了1-bit权重支持。这意味着理论上可以把一个7B模型压缩到不到1GB，虽然精度损失不小，但对于某些特定任务（分类、简单问答）可能够用。

Ollama则是llama.cpp的"Docker化"封装，一行命令就能跑起来。适合快速原型验证，但不适合生产环境。

## 怎么选？

简单决策树：本地开发用Ollama；中高流量生产用vLLM或SGLang（后者在RAG/结构化输出场景更优）；极限性能且愿意承受工程复杂度的用TensorRT-LLM；边缘设备用llama.cpp。

趋势很清楚：KV Cache优化和硬件特化编译是当前推理优化的两大支柱。前者以PagedAttention和RadixAttention为代表，后者以TensorRT-LLM为代表。两条路线最终可能会融合。

原文链接：https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13
