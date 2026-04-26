---
title: "Tool Attention: 用动态门控干掉 MCP 的隐性 Token 税"
date: 2026-04-25T07:30:00-04:00
tags: [MCP, Agent, Token-Optimization, Tool-Selection, Retrieval]
description: "当 Agent 接了 120 个 tool，每轮光 schema 就吃 47k tokens。Tool Attention 用 embedding + 状态门控 + 懒加载砍掉 95%。"
showToc: true
---

当你的 Agent 接了 6 个 MCP server、120 个 tool，每轮对话光 tool schema 就吃掉 47k tokens。这篇论文提出 Tool Attention 机制，用 embedding 相似度 + 状态门控 + 两阶段懒加载，把每轮 tool token 从 47.3k 砍到 2.4k，削减 95%。

## 背景：MCP Tax 到底有多重

MCP（Model Context Protocol）是 Anthropic 在 2024 年底推出的工具协议标准，现在 OpenAI、Google、Microsoft 都在用。它的核心设计是无状态的：每轮对话都要把所有已连接 server 的全部 tool schema 重新序列化塞进 prompt。

这在 tool 少的时候无所谓，但一旦进入真实的企业场景——文件系统、Git、数据库、Slack、搜索引擎各挂一个 server——token 开销就变成了一笔隐性税收。论文引用的实测数据：

| 场景 | Tool 数量 | Token 占用 |
|------|----------|-----------|
| GitHub MCP 全量 | 93 | 55k |
| 企业数据库 catalog | 106 | 54.6k |
| 普通 4-server 部署 | ~40-60 | 15k-20k |

这不只是钱的问题。作者定义了一个 effective context utilization 指标 ρ，当 tool schema 占满上下文后，留给真正任务内容（用户消息、推理过程、tool 输出）的空间急剧缩小。经验数据显示，当上下文利用率超过约 70% 时，LLM 的推理质量会出现断崖式下降——开始瞎编 tool 参数、混淆相似 tool、丢失多步任务的连贯性。

## 核心机制：三层筛选

Tool Attention 的设计灵感来自 Transformer self-attention——不再让每轮对话"看到"所有 tool，而是动态选择当前最相关的子集。具体分三层：

### 第一层：Intent-Schema Overlap（ISO）打分

用 sentence-transformers 把每个 tool 的名称+描述编码成 384 维向量（用的 all-MiniLM-L6-v2），存入 FAISS 索引。每轮对话时，把用户消息也编码，算 cosine similarity。这一步极快，120 个 tool 的检索在毫秒级。

### 第二层：状态门控

不是所有语义相关的 tool 都该出现。比如用户还没认证，需要 auth 的 tool 就不该暴露；搜索还没执行，后处理 tool 就不该出现。门控函数检查当前 agent 的执行状态是否满足每个 tool 的前置条件，不满足的直接过滤。

### 第三层：两阶段懒加载

通过前两层筛选后，只保留 top-k 个 tool（论文中 k=5）。但关键在于加载策略：上下文里平时只放一个极简的 tool summary pool（每个 tool 约 60 tokens 的名称+简述），只有进入 top-k 的 tool 才展开完整 JSON schema。这样既保留了 LLM 对全局 tool 的"模糊感知"，又把真正占空间的 schema 压到最少。

## 理论依据：Total Attention Energy

论文借用了 MindGuard 提出的 Total Attention Energy（TAE）概念。TAE 衡量的是生成 token 对上下文中某个 metadata token 的注意力权重平方和——本质上是衡量某段上下文对最终决策的因果影响。

Tool Attention 的逻辑是：如果一个 tool 的 schema 和当前 query 的 embedding 相似度很低，那它在 LLM 的 attention 层面也不会获得显著的 TAE，放不放进 prompt 对 tool call 决策没有影响。

这同时也是一个安全机制——Tool Poisoning Attack 依赖恶意 tool description 被 LLM 的 attention 读到，如果语义不匹配直接被门控掉，攻击面就大幅缩小。

## 实验结果

在 120-tool、6-server 的模拟 benchmark 上（token 数据校准自真实部署审计）：

| 指标 | 基线 | Tool Attention | 变化 |
|------|------|---------------|------|
| 每轮 tool token | 47.3k | 2.4k | -95.0% |
| Effective context utilization | 24% | 91% | +67pp |
| 任务成功率 | 72% | 94% | +22pp |
| TTFT 延迟 | — | — | -38% |

需要注意：end-to-end 指标（成功率、延迟、成本）是基于实测 token 数 + 公开部署遥测数据的**投影值**，不是在 live agent 上实测的。作者在论文中反复标注了这一点，比较诚实。

## 为什么值得关注

这篇论文切中了一个被严重低估的工程问题。大家都在讨论怎么扩展 context window、怎么做 KV cache 优化，但很少有人正式量化"工具定义本身就在吃你的上下文"这件事。作者的核心论点很简洁：**protocol-level efficiency，而不是 raw context length，才是 scalable agentic systems 的真正瓶颈。**

实现上也很务实——sentence-transformers + FAISS + tiktoken，全是现成组件，作为 LangGraph middleware 的 before_model hook 插入，不需要改 MCP 协议本身。代码已开源。

对于正在搭建多 tool Agent 系统的人来说，这个思路可以直接用：就算不用他的完整框架，"根据用户意图动态裁剪 tool schema"这个 pattern 本身就值得实践。

**原文链接：** https://arxiv.org/abs/2604.21816

**GitHub：** https://github.com/asadani/tool-attention

---

## 面试关联知识点

### KV Cache 与上下文长度的关系

KV Cache 的显存占用与 sequence length 成正比。每多一个 token 进入 prompt，每一层 Transformer 都要为它存一对 K、V 向量。Tool schema 作为 prompt 的一部分，直接膨胀 KV Cache，增加 GPU 内存压力和 TTFT。量化（KV Cache Quantization 到 INT8/INT4）可以缓解存储问题，但无法减少 attention 计算中 O(n²) 的复杂度——最根本的办法还是减少 prompt 中不必要的 token 数量。

### Agent 中的 Tool Use 稳定性问题

LLM 调用外部 tool 的稳定性受两个因素制约：一是 tool schema 的清晰度（参数命名、description 质量），二是上下文中 tool 数量过多导致的混淆。当 prompt 里塞了上百个 tool definition，模型容易出现参数混用、调错 tool 的问题。解决方案包括：grammar-constrained decoding 强制输出合法 JSON、动态裁剪 tool 集合（就是本文的思路）、以及分层 Agent 设计让每个子 Agent 只看到自己需要的 tool。

### Tool-level RAG vs 文档 RAG

本文的 ISO scoring 本质上就是一个 tool-level 的 RAG——用 dense retrieval 从 tool catalog 中检索最相关的子集。和传统文档 RAG 的区别在于：文档 RAG 检索后拼接到 prompt 供 LLM 参考，tool RAG 检索后决定哪些 tool schema 被暴露给 LLM 的 function calling 接口。两者共享同一套基础设施（embedding model + 向量索引），但 tool retrieval 还需要额外的状态门控（前置条件检查），这是文档检索不需要的。
