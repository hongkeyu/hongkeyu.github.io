---
title: "Gemini Embedding 2：第一个原生多模态 Embedding 模型，统一五种模态到同一向量空间"
date: 2026-03-19T07:30:00+08:00
tags: ["embedding", "multimodal", "RAG"]
description: "Google 发布 Gemini Embedding 2，首次将文本、图像、视频、音频、PDF 五种模态原生映射到同一 embedding 空间，对 RAG 和语义搜索架构有重大影响。"
showToc: true
---

TL;DR: Google 于 3 月 10 日发布 Gemini Embedding 2，首次将文本、图像、视频、音频、PDF 文档五种模态原生映射到同一 embedding 空间，支持交叉模态检索和分类，对 RAG 和语义搜索架构有重大影响。

---

## 背景：Embedding 模型的演进

Embedding 是现代 AI 系统的基础设施。从 Word2Vec 的静态词向量，到 BERT 的上下文向量，再到 OpenAI text-embedding-3 和 Cohere Embed v3 这些专用 text embedding 模型，过去几年的进化主要集中在文本模态内部——更长的上下文、更好的多语言支持、更高的 MTEB 分数。

多模态 embedding 并不新鲜。CLIP 在 2021 年就实现了图文对齐，ImageBind 扩展到六种模态。但这些模型要么只覆盖两种模态（CLIP），要么是学术性质的（ImageBind），没有一个生产级别的、API 可用的统一多模态 embedding 模型。

Gemini Embedding 2 填补的正是这个空白。

## 核心设计

Gemini Embedding 2 基于 Gemini 架构构建，不是把多个单模态 encoder 拼在一起，而是原生的多模态理解——模型在训练阶段就同时处理多种模态，共享同一套表示空间。支持的输入：

- 文本：最长 8192 tokens，覆盖 100+ 语言
- 图像：单次请求最多 6 张，支持 PNG/JPEG
- 视频：最长 120 秒，支持 MP4/MOV
- 音频：原生音频 embedding，不需要先转录成文本
- 文档：直接处理 PDF，最多 6 页

关键能力是 interleaved input：你可以在一次请求中混合传入图片+文本，模型会捕捉跨模态的语义关系。这跟"先分别 embed 再拼接"有本质区别——后者丢失了模态间的交互信息。

## 维度灵活性：Matryoshka Representation Learning

模型默认输出 3072 维向量，但采用了 Matryoshka Representation Learning (MRL) 技术，可以动态缩减到 1536 或 768 维。MRL 的核心思想是在训练时让前 k 维的子向量也具备独立的语义表达能力——就像俄罗斯套娃一样，外层包含内层的信息。

这在工程上非常实用。你可以用 768 维做初筛（省存储和计算），然后用 3072 维做精排，不需要维护两套模型。

## 对 RAG 架构的影响

这个模型对 RAG 系统的意义比表面看起来大得多。

传统 RAG 的知识库通常只处理文本。如果你的文档包含图表、流程图、代码截图，要么 OCR 转文本（损失信息），要么直接忽略。有了统一的多模态 embedding 空间，你可以把图片、音频片段、视频都纳入检索范围，用户的文本 query 能直接跨模态召回相关内容。

更进一步，音频原生 embedding（不经过 ASR 中转）意味着可以直接检索播客、会议录音、客服通话中的语义内容，不丢失语调、停顿等文本转录会丢弃的信息。

## 局限和注意事项

当前还在 Public Preview 阶段。视频限制 120 秒、PDF 限制 6 页，对长内容场景需要分块处理。另外，benchmark 数据 Google 只给了总分对比，没有公开各模态的详细消融实验，实际跨模态检索的精度需要自己验证。

API 通过 Gemini API 和 Vertex AI 两个入口提供，集成方面 LangChain、LlamaIndex、Weaviate、Qdrant、ChromaDB 都已经适配。

## CLIP 对比学习训练

CLIP 用对比学习（contrastive learning）训练图文对齐：给定一批 image-text pair，正样本是匹配的图文对，负样本是 batch 内所有其他组合。训练目标是让正样本的 cosine similarity 最大化、负样本最小化（InfoNCE loss）。CLIP 的 batch size 极大（32768），因为对比学习的负样本数量直接影响表示质量。Gemini Embedding 2 与 CLIP 的本质区别在于：CLIP 只对齐图文两种模态，而 Gemini Embedding 2 原生支持五种模态的统一空间。

## 延伸思考

Embedding 模型是"沉默的基础设施"——用户感知不到它的存在，但它决定了检索质量的上限。一个统一的多模态 embedding 空间意味着：搜索引擎可以用文字搜视频、用图片搜音频、用语音搜文档。这不是技术 demo，是产品范式的转变。

对于做 RAG/Agent 的工程师来说，这意味着知识库的"索引"不再局限于文本，系统的信息覆盖面和召回率会有质的提升。

原文链接：https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/

---

## 面试关联知识点

### 1. Embedding 模型 vs Reranker 模型的配合

Embedding 模型负责召回阶段，把 query 和候选文档映射到同一向量空间，通过向量相似度（cosine/dot product）快速筛选 top-k。Reranker 是精排阶段，接收 query-document pair 做交叉编码（cross-encoder），计算更精确的相关性分数。Embedding 是 bi-encoder（query 和 doc 独立编码，速度快但精度有限），Reranker 是 cross-encoder（联合编码，精度高但不能预计算）。生产系统通常是 embedding 召回 100-500 → reranker 精排到 top-10。

### 2. 混合检索（稠密+稀疏）+ Reranker

稠密检索（dense retrieval）用 embedding 向量做语义匹配，擅长理解同义词和语义相近的内容；稀疏检索（BM25）做精确关键词匹配，擅长处理专有名词和罕见词。混合检索（hybrid search）结合两者，通常用 RRF（Reciprocal Rank Fusion）合并排序结果，再送入 reranker 做最终排序。这种三级流水线（sparse + dense → fusion → rerank）是当前 RAG 系统的标准架构。
