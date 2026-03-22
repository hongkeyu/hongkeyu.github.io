---
title: "Gemini Embedding 2：多模态 Embedding 如何重构 RAG 管线"
date: 2026-03-21T07:30:00+08:00
tags: [embedding, RAG, multimodal]
description: "Google 发布首个原生四模态统一 embedding 模型，通过 MRL 实现维度灵活压缩，将传统 RAG 管线从多模型多存储架构简化为单模型单向量库。"
showToc: true
---

## TL;DR

Google 于 3 月 10 日发布 Gemini Embedding 2，首个原生支持文本/图像/视频/音频四种模态的统一 embedding 模型，将传统 RAG 管线从多模型多存储架构简化为单模型单向量库，并通过 Matryoshka Representation Learning 实现维度灵活压缩。

---

## 背景：为什么多模态 embedding 是刚需

传统 RAG 管线的核心痛点：只能处理文本。企业内部知识库里的架构图、产品截图、会议录像、客服电话录音——这些非文本数据在 embedding 阶段被直接丢弃。你可能有一份包含关键流程图的 PDF，但检索系统对图片内容完全无感知，导致用户明明问的问题知识库里有答案，却搜不出来。

过去的解决方案是拼凑式的：文本用 text-embedding-3，图像用 CLIP，音频先 Whisper 转文字再 embed。每种模态一个模型、一个向量库、一套检索逻辑，pipeline 复杂度指数增长，且最致命的问题是跨模态检索不可能——你无法用一段文字去检索一张相关的架构图。

Gemini Embedding 2 试图从根本上解决这个问题：把所有模态映射到同一个向量空间。

## 核心技术细节

**输入能力：** 支持文本（最大 8192 tokens，100+ 语言）、图像（每次请求最多 6 张，PNG/JPEG）、视频（最长 120 秒，MP4/MOV）、音频（原生处理，不需要先转文字）、复杂文档（PDF 等混合文本+图像格式）。

**输出维度与 Matryoshka Representation Learning (MRL)：** 默认输出 3072 维向量。MRL 的核心思想是信息按重要性嵌套排列——像俄罗斯套娃一样，高维包含低维的全部信息。这意味着你可以在不重新训练的情况下把向量截断到 1536/768/256 维，核心语义信息仍然保留在前面的维度中。

实际工程价值巨大：百万级文档索引时，可以用 256 维做第一轮 ANN 粗筛（存储成本降低 87%），再用 3072 维对 top-50 候选做精排。兼顾成本和精度。

**Task Type 参数：** API 允许指定 embedding 用途——RETRIEVAL_DOCUMENT（索引文档时用）、RETRIEVAL_QUERY（查询时用）、SEMANTIC_SIMILARITY、CLASSIFICATION、CLUSTERING。关键实践：索引和查询必须用不同的 task type，这对非对称检索的性能提升显著。这是因为文档通常很长、查询通常很短，用不同的 task type 让模型能分别优化两种分布。

**竞品对比：**

- OpenAI text-embedding-3-large：仅文本，3072 维，支持 MRL，$0.13/1M tokens
- Cohere embed-v4：文本+图像，1024 维，支持 MRL，$0.10/1M tokens
- Voyage AI voyage-3：仅文本，1024 维，不支持 MRL，$0.06/1M tokens
- Gemini Embedding 2：文本+图像+视频+音频，3072 维，支持 MRL，preview 期间免费

四模态原生支持 + 最高维度 + 免费，竞争力很明确。

## 工程意义与落地考量

管线简化是最直接的收益：3-4 个 embedding 模型合并为 1 个，2-3 个向量库合并为 1 个，模态间的同步逻辑全部消除。Google 官方博客提到部分客户实现了 70% 的延迟降低。

但也有现实问题需要考虑：

- **供应商锁定：** 目前 Google 独占。如果在意多云策略，embedding 层需要设计成可替换接口，利用 MRL 的维度兼容性做模型切换的缓冲
- **数据治理：** 多模态数据发送到外部 API 意味着更复杂的合规需求。会议录像、客服音频可能包含 PII，embed 前需要脱敏。Vertex AI 上可以用 VPC-SC + CMEK 限制数据边界
- **成本：** preview 免费，GA 后必然收费。256 维粗筛 + 3072 维精排的两阶段策略是控制成本的关键

**对边缘部署的启示：** MRL 的 256 维模式天然适合资源受限场景。在 Jetson 这类设备上，256 维向量的存储和相似度计算成本远低于 3072 维，可以做本地轻量检索，复杂精排再回云端。

原文链接：<https://jangwook.net/en/blog/en/gemini-embedding-2-multimodal-rag-pipeline/>
Google 官方公告：<https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/>

---

## 面试关联知识点

### 1. Embedding 模型 vs Reranker 模型的配合

Embedding 负责从海量文档中做 ANN 粗召回（高速、低精度），Reranker 对召回结果做精排（低速、高精度）。Embedding 是双塔结构（query 和 doc 分别编码，离线计算 doc 向量），Reranker 是交叉编码器（query-doc 对拼接后打分，不能离线预计算）。两者配合是工业级 RAG 的标准范式。

### 2. 混合检索（稠密+稀疏）+ Reranker

稠密检索（embedding ANN）擅长语义匹配但对精确关键词弱，稀疏检索（BM25）擅长精确匹配但不理解语义。生产系统通常两路并行召回，合并去重后交给 Reranker 精排。Gemini Embedding 2 的多模态能力扩展了稠密检索的覆盖范围，但稀疏检索仍然不可替代。

### 3. ANN（近似最近邻）向量检索

核心算法包括 HNSW（分层导航小世界图，主流方案，查询 O(log n)）和 IVF（倒排文件索引，先聚类再搜索）。MRL 降维后向量更短，HNSW 图的内存占用和距离计算成本同步下降，直接提升 QPS。这也是为什么 MRL 对大规模部署至关重要。
