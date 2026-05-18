---
title: "Continuous Batching 的异步化：CPU/GPU 流水线并行如何白拿 22% 推理加速"
date: 2026-05-17T07:30:00-04:00
tags: [LLM推理优化, CUDA, Continuous-Batching]
description: "把 LLM 推理的 CPU batch 准备和 GPU 前向计算解耦到不同 CUDA stream 上并行执行，GPU 利用率从 76% 提升到 99.4%，生成速度提升 22%，不改模型、不换 kernel。"
showToc: true
---

## 背景

这是 Hugging Face 关于高效 LLM 推理系列的第二篇，5 月 14 日发布。第一篇讲了 continuous batching 的基本原理（通过紧密调度消除 padding 浪费），这篇解决的是 continuous batching 没解决的第二个问题：**同步执行导致的 CPU/GPU 互相等待**。

标准的同步 continuous batching 流程是这样的：

> CPU 准备 batch（选择请求、更新 KV cache 表、驱逐完成的请求、接纳新请求）→ 传输到 GPU → GPU 做 forward pass + sampling → 结果回传 CPU → CPU 再准备下一个 batch

问题很明显：GPU 算的时候 CPU 闲着，CPU 准备的时候 GPU 闲着。在 8B 模型、batch size 32、生成 8K token 的实测中，GPU 有 24% 的时间在等 CPU。H200 一小时 5 美元，这意味着每天你白烧了将近 30 美元。

## 核心机制：三条 CUDA Stream + Event 同步

解决思路是经典的流水线并行：让 batch N 的 GPU 计算和 batch N+1 的 CPU 准备同时进行。实现依赖三个关键概念：

### CUDA Stream 分离

所有 GPU 操作必须离开 default stream（default stream 有全局同步语义，会强制等待所有其他 stream 完成）。文章使用三条 non-default stream：

| Stream | 职责 |
|--------|------|
| H2D stream | CPU → GPU 传输 |
| Compute stream | 前向计算 |
| D2H stream | GPU → CPU 传输 |

Non-default stream 上的操作对 CPU 是非阻塞的——enqueue 完立刻返回控制权。

### CUDA Event 做跨 Stream 依赖

Stream 之间天然独立，需要用 event 建立偏序关系：

1. H2D 传输完成后 record 一个 event
2. Compute stream wait 这个 event 才开始 forward pass
3. Compute 完成后同理，D2H stream wait 之后才回传结果

整个 pipeline 中 CPU 唯一的阻塞点是最后的 `d2h_done_event.synchronize()`——等 batch N 的输出落地。

### 双 Slot 交替 + Carry-Over

为了防止 batch N 和 N+1 共享 buffer 导致 race condition，使用两组 input/output tensor slot 交替使用（slot A 和 slot B）。这会翻倍 input buffer 的显存占用，但由于使用 FlashAttention 不需要 attention mask（最大的 input tensor），实际开销可控。

CUDA graph 的问题通过 memory pool 解决——两个 graph 共享一个 memory pool，只要不同时执行就不会冲突。

**Carry-over** 是一个巧妙的设计：准备 batch N+1 时，batch N 还没算完，所以那些延续请求的新 token 还不知道。解法是先用 0 占位，等 batch N 算完后在 GPU 上做一次轻量的 carry-over 操作（选择、清零、截断、加法四步），把 batch N 的输出 token 写入 batch N+1 的 input。这个操作被 capture 进 CUDA graph，几乎零成本。

## 实测结果

同样的 8B 模型配置，异步化后：

| 指标 | 同步 | 异步 | 提升 |
|------|------|------|------|
| GPU 活跃时间占比 | 76.0% | 99.4% | +23.4pp |
| 生成时间 | 300.6s | 234.5s | -22% |

理论上限是 24%（完全消除 CPU overhead），差距来自 batch 之间不可避免的那一次 sync point。

更值得注意的是：这是**纯调度层面的优化**，不涉及任何新 kernel、模型修改或量化。对于 RL 训练中常见的 16K+ 长生成场景，这类优化的累积效果非常显著。

## 为什么重要

这篇文章的价值不只在于那 22% 的加速，而在于它把 GPU inference serving 中最容易被忽视的一层——CPU/GPU 协调——讲得非常透彻。对于做推理优化的人来说，这提供了一个完整的思维框架：**先 profile CPU 和 GPU 各自的 idle 时间，再判断瓶颈在调度还是在计算**。

实现已经合入 transformers 库的 continuous batching 模块，代码入口在 `transformers/generation/continuous_batching/continuous_api.py`，异步 IO 逻辑在 `ContinuousBatchingAsyncIOs` 类中。

## 延伸

系列下一篇预告会覆盖 request offloading、decode-specific kernel、fine-grained compile 等进一步优化。这些加在一起有望让 transformers 原生推理在长生成场景逼近 vLLM / TGI 级别的吞吐。

原文链接：https://huggingface.co/blog/continuous_async

---

## 面试关联知识点

### CUDA Stream 和 Default Stream 的区别是什么？

Default stream 具有全局同步语义：它上面的操作必须等所有其他 stream flush，其他 stream 的操作也必须等 default stream flush。Non-default stream 之间天然独立，可以并行执行。要实现 CPU/GPU overlap，必须把 GPU 操作放到 non-default stream 上，否则每个操作都会隐式触发全局同步。

### KV Cache 在 Continuous Batching 中如何管理？

每个请求维护独立的 KV cache 空间。Continuous batching 的调度器在每一步决定哪些请求参与当前 batch，更新 KV cache 路由表（page table），驱逐已完成请求释放空间，接纳新请求填充空位。异步化场景下，KV cache 的路由更新属于 CPU 侧工作，在 GPU 计算期间完成，不额外占用 GPU 时间。

### Prefill 和 Decode 阶段的性能瓶颈有什么区别？

Prefill 阶段处理完整 prompt，是 compute-bound（大量矩阵乘法，GPU 利用率高）。Decode 阶段每步只生成一个 token，是 memory-bound（主要瓶颈在读取 KV cache 的显存带宽）。本文的异步优化主要针对 decode 阶段——因为 decode 的单步 GPU 计算更轻，CPU 调度开销占比更大，overlap 的收益更明显。
