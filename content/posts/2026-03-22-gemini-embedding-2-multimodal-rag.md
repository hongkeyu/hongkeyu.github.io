---
title: "Gemini Embedding 2：多模态 Embedding 如何重构 RAG 管线"
date: 2026-03-22T07:30:00+08:00
tags: [RAG, Embedding, 多模态]
description: "Google 发布 Gemini Embedding 2，首个原生支持文本/图像/视频/音频四种模态的 embedding 模型，统一映射到同一向量空间，彻底改变 RAG 管线只能处理文本的局限。"
showToc: true
---

**TL;DR:** Google 于 3 月 10 日发布 Gemini Embedding 2，首个原生支持文本/图像/视频/音频四种模态的 embedding 模型，统一映射到同一向量空间，彻底改变了 RAG 管线只能处理文本的局限。

---

## 背景：RAG 的模态瓶颈

过去几年 RAG 已经成为 LLM 落地的标配架构，但有一个根本性的限制一直没解决好：embedding 阶段只能处理文本。企业知识库里大量的架构图、产品截图、会议录音、操作视频，在检索阶段全部被忽略。你可能会说可以用 OCR 或 Whisper 把非文本内容转成文本再 embed，但中间转换必然丢失信息，而且多条管线的维护成本很高——不同模态各需一个 embedding 模型、各需一个向量库、各需一套检索逻辑，同步和一致性问题一堆。

Gemini Embedding 2 的思路很直接：一个模型吃所有模态，输出到同一个向量空间。文本查图片、语音查文档，cosine similarity 直接算，不需要中间转换。

## 核心规格

输入支持：文本（8192 tokens，100+ 语言）、图像（PNG/JPEG，单次最多 6 张）、视频（最长 120 秒，MP4/MOV）、音频（原生处理，不经过 STT 中间步骤）。

输出维度：默认 3072 维。关键技术是 Matryoshka Representation Learning (MRL)——像俄罗斯套娃一样，核心信息集中在高维部分，可以按需截断到 1536/768/256 维。这意味着可以做两阶段检索：先用 256 维做 ANN 粗筛（省存储省计算），再用 3072 维对 top-K 做精排。Google 官方数据称部分客户实现了 70% 的延迟降低。

目前通过 Gemini API（AI Studio，有免费额度）和 Vertex AI（企业级，支持 VPC-SC 和 CMEK）两个入口提供服务，preview 期间免费。

## 和现有方案的对比

横向比较几个主流 embedding 模型：OpenAI text-embedding-3-large 只支持文本，3072 维，支持 MRL，$0.13/M tokens；Cohere embed-v4 支持文本+图像，1024 维；Voyage AI voyage-3 只支持文本，1024 维。Gemini Embedding 2 是目前唯一一个原生支持全部四种模态、输出维度最高、且支持 MRL 的模型。

## 工程上怎么用

API 设计很简洁。Python SDK 里 `client.models.embed_content()` 一个接口搞定所有模态，通过 `task_type` 参数区分用途：`RETRIEVAL_DOCUMENT` 用于文档入库，`RETRIEVAL_QUERY` 用于查询编码，`SEMANTIC_SIMILARITY` 用于相似度比较，`CLASSIFICATION` 和 `CLUSTERING` 分别用于分类和聚类。一个实际经验：入库和查询一定要用不同的 task_type，这对非对称检索的效果影响很大。

MRL 维度截断通过 `output_dimensionality` 参数控制，比如设为 768 就得到 768 维向量，不需要额外降维操作。文本和图像的 embedding 输出在同一空间，直接 cosine similarity 就能算跨模态相似度。

## 生产迁移的关键考量

1. **管线简化：** 3-4 个模型变 1 个，2-3 个向量库合 1 个，同步逻辑直接消失，运维成本显著下降
2. **供应商锁定：** 目前 Google 独家。建议把 embedding 层做成可替换接口，利用 MRL 的维度灵活性保持和其他模型的兼容性
3. **数据治理：** 多模态数据发外部 API 需要注意合规，特别是会议录音和客服通话里的 PII 信息，建议 embedding 前先做脱敏
4. **成本规划：** preview 之后会收费。用 256 维 MRL 做索引可以比 3072 维节省约 87% 的存储成本

## 为什么值得关注

多模态 RAG 不是新概念，之前用 CLIP 做图像 embedding + text embedding 拼凑也能跑，但那本质上是多个单模态模型硬拼，向量空间不一致，跨模态检索效果差。Gemini Embedding 2 是第一个真正做到「一个模型、一个空间、所有模态」的生产级方案。对求职面试来说，如果聊到 RAG 架构设计，了解多模态 embedding 的最新进展会是一个加分点——面试官经常会问「你觉得 RAG 目前的瓶颈在哪」，模态限制就是一个很好的切入角度。

另外，MRL (Matryoshka Representation Learning) 本身是一个值得深入了解的技术，它不依赖特定模型，是一种通用的 embedding 训练策略。

原文：<https://jangwook.net/en/blog/en/gemini-embedding-2-multimodal-rag-pipeline/>
Google 官方公告：<https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/>

---

## 面试关联知识点

### 1. Embedding 模型 vs Reranker 模型的配合

Embedding 模型做粗检索（recall），输出固定维度向量用于 ANN 搜索；Reranker 模型做精排（precision），输入 query-doc pair 直接输出相关性分数。Reranker 精度高但计算成本也高，所以一般只对 top-K 候选做 rerank。Gemini Embedding 2 的 MRL 维度截断实际上在 embedding 层内部就实现了类似的粗排-精排两阶段。

### 2. 混合检索（稠密+稀疏）+ Reranker

稠密检索（如 embedding ANN）擅长语义匹配但可能漏掉精确关键词；稀疏检索（如 BM25）擅长精确匹配但缺乏语义理解。生产环境通常两者并用，各取 top-K 后合并去重，再过 Reranker。面试常问「为什么不只用 embedding 检索」——答案就是稠密检索在精确术语匹配上不如 BM25。

### 3. ANN（近似最近邻）向量检索

核心思想是用空间索引结构（HNSW、IVF、PQ 等）加速高维向量搜索，时间从 O(n) 降到近似 O(log n)。HNSW 是目前最常用的，基于 skip-list 思想构建多层图。面试可能问 trade-off：索引构建时间 vs 查询速度 vs 召回率。MRL 降维也是一种降低 ANN 搜索成本的策略——维度越低，距离计算越快。
