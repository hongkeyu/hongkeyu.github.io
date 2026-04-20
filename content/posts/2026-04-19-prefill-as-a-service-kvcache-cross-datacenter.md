---
title: "Prefill-as-a-Service: 当 KVCache 可以跨数据中心传输"
date: 2026-04-19T07:30:00-04:00
tags: [LLM-Inference, KV-Cache, Distributed-Systems]
description: "Moonshot AI 提出 PrfaaS 架构，利用 hybrid-attention 模型压缩的 KVCache 实现跨数据中心 prefill 卸载，1T 模型吞吐提升 54%。"
showToc: true
---

## 背景：PD 分离的网络瓶颈

LLM 推理分 prefill（计算密集）和 decode（带宽密集）两个阶段。主流 PD disaggregation 架构将二者分开运行，但 prefill 产出的 KVCache 必须快速传输到 decode 节点。Dense attention 模型下，32K token 请求的 KV 传输速率高达约 60 Gbps，意味着 prefill 和 decode 必须在同一个 RDMA 高速网络域内。

这堵死了异构部署路线：prefill 用算力强的芯片（如 NVIDIA Rubin CPX），decode 用带宽大的芯片（如 Groq LPU），但两类硬件通常不在同一机房，跨集群传 KVCache 的带宽成本根本扛不住。

## 转折点：Hybrid Attention 压缩 KVCache

近一年主流大模型纷纷转向 hybrid-attention 架构——少量 full attention 层 + 大量线性注意力或 sliding window attention 层：

| 模型 | 架构比例 | KV 压缩倍数 |
|------|---------|------------|
| Qwen3.5-397B | 3:1 linear-to-full | — |
| MiMo-V2-Flash | 5:1 SWA-to-full | ~13x (4.66 vs 59.93 Gbps) |
| Ring-2.5-1T | 7:1 + MLA | ~36x |

这把跨数据中心传输 KVCache 从「不可能」变成了「值得优化」。

## PrfaaS 核心设计

Moonshot AI 和清华提出的 Prefill-as-a-Service 架构有三个关键设计：

### 选择性卸载

只把长上下文、未命中 prefix cache 的请求卸载到远端 PrfaaS 集群做 prefill，短请求继续走本地 PD 路径。用长度阈值做路由决策，避免滥用有限的跨集群带宽。

### Hybrid Prefix Cache Pool

跨集群维护全局 prefix cache 管理器，路由决策同时考虑请求长度、cache 命中位置和可用跨集群带宽。如果长请求的 prefix 恰好缓存在本地，就没必要发到远端重算。

### 双时间尺度调度

- **短期**：bandwidth-aware + cache-aware 请求路由，实时响应带宽波动
- **长期**：根据流量模式动态调整 prefill/decode 集群的资源分配比例

## 实验结果

在内部 1T 参数 hybrid 模型（Kimi Linear 架构）上：

- PrfaaS 比同构 PD 基线吞吐高 **54%**
- 比朴素异构基线高 **32%**
- 跨数据中心带宽消耗可控

规模估算：512 GPU H200 prefill 集群处理 32K 平均长度请求，Ring-2.5-1T 只需约 170 Gbps 出口带宽；128K 超长请求降到 100 Gbps 以下；万卡级别总出口约 1.8 Tbps，完全在物理跨数据中心链路承载能力之内。

## 为什么值得关注

这篇论文指出了一个结构性转变：当模型架构本身压缩 KVCache（dense → hybrid attention），推理系统的部署边界也随之松动。「prefill 和 decode 必须在同一机房」这个隐含约束正在被打破。

对做推理优化的人来说：未来 serving 架构设计不能只看单集群内优化，**跨集群甚至跨区域的异构资源编排**会成为新的竞争维度。

📄 原文：[arXiv:2604.15039](https://arxiv.org/abs/2604.15039)

---

## 面试关联知识点

### KV Cache 原理

KV Cache 存储已计算的 key/value 向量，避免 decode 阶段重复计算。压缩方案演进：

- **GQA**：多个 query head 共享 KV head（如 8:1）
- **MLA**：低秩投影进一步压缩 KV 表示
- **Hybrid attention**：仅少数 full attention 层产生序列长度相关的 cache，线性注意力层维护固定大小循环状态

### Prefill vs Decode

| | Prefill | Decode |
|---|---------|--------|
| 处理方式 | 一次性处理整个输入 | 逐 token 生成 |
| 瓶颈 | Compute-bound | Memory-bandwidth-bound |
| GPU 利用 | 高并行度 | 每步读取整个 KV Cache |

这种本质差异是 PD disaggregation 的理论基础。

### 与 Speculative Decoding 的关系

两者解决不同问题且可正交组合：
- **Speculative decoding**：小模型生成候选 token + 大模型验证，压缩 decode 延迟
- **PrfaaS**：解决 prefill 计算瓶颈和跨节点 KVCache 传输问题
