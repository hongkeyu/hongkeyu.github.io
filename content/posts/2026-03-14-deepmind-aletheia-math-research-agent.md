---
title: "Google DeepMind Aletheia：从奥赛金牌到自主数学研究的 AI Agent"
date: 2026-03-14T07:30:00+08:00
tags: ["ai-agent", "math-reasoning", "inference-scaling"]
description: "DeepMind 发布 Aletheia，基于 Gemini Deep Think 的数学研究 Agent，IMO-Proof Bench Advanced 达到 95.1%，首次实现完全自主生成可发表数学论文。"
showToc: true
---

TL;DR: DeepMind 发布 Aletheia，一个基于 Gemini Deep Think 的数学研究 Agent，在 IMO-Proof Bench Advanced 上达到 95.1% 准确率（前纪录 65.7%），并首次实现了完全自主生成可发表的数学论文。

---

## 背景

AI 做数学题不是新闻——2025 年 IMO 金牌已经拿了。但竞赛题和真正的数学研究之间有一条巨大的鸿沟：竞赛题有明确答案、限时可解；研究问题需要检索文献、构建长链证明、发现新结论。Aletheia 就是 DeepMind 尝试跨越这条鸿沟的产物。

## 核心架构：Generator-Verifier-Reviser 三角循环

Aletheia 的架构并不复杂，但设计思路很值得注意。它把传统的"生成答案"拆成了三个独立角色：

- Generator：针对研究问题生成候选解法
- Verifier：用自然语言对候选解法做非形式化验证，检查逻辑漏洞和幻觉
- Reviser：根据 Verifier 发现的问题修正解法，循环直到通过

关键 insight 是：**把验证从生成中解耦出来**。DeepMind 发现，如果让同一个模型在同一次推理中既生成又验证，它倾向于忽略自己刚犯的错误。分离之后，验证质量显著提升。这跟软件工程里"写代码的人不应该自己做 code review"是一个道理。

## Inference-Time Scaling 的实证

这篇论文提供了 inference-time compute scaling 的一个极好案例。2026 年 1 月版本的 Deep Think 相比 2025 年版本，在达到相同 IMO 级别性能时所需的推理计算量降低了 100 倍。注意这不是训练效率提升，而是纯推理阶段的 scaling——给模型更多"思考时间"，准确率就上去。

具体数字：IMO-Proof Bench Advanced 从 65.7% 跳到 95.1%。这不是渐进式改进，而是质变。

## 研究里程碑

Aletheia 已经产出了几个实际成果：

- **Feng26 论文**（完全自主）：Aletheia 独立计算了算术几何中的 eigenweight 结构常数，生成了一篇完整的研究论文，全程无人干预。DeepMind 将其分类为"Level A2"——本质上自主且达到可发表质量。

- **LeeSeo26**（人机协作）：Aletheia 提供了证明独立集上界的高层策略和路线图，人类作者据此完成了严格证明。

- **Erdos 猜想挑战**：对 700 个开放问题进行批量攻击，找到 63 个技术上正确的解，其中 4 个是此前未解决的开放问题。

## Tool Use 防幻觉

一个有趣的工程细节：Aletheia 大量使用 Google Search 和网页浏览来检索真实的数学文献。原因很直接——数学论文的引用是可验证的硬事实，LLM 特别容易在这里产生幻觉（编造不存在的论文）。通过 tool use 接入真实搜索，这个问题得到了有效缓解。

## AI 自主性分类框架

DeepMind 还顺带提出了一个 AI 数学贡献的分类体系，类似自动驾驶的 L0-L5 分级：

- Level H（主要人类）：AI 辅助计算
- Level 1（人机协作）：AI 提供策略，人类完成证明
- Level 2（本质自主）：AI 独立产出可发表成果

这个框架的意义在于为 AI 辅助研究建立透明的评估标准，解决目前"AI 到底贡献了多少"说不清楚的问题。

## 延伸思考

Aletheia 的 Generator-Verifier-Reviser 模式本质上是 multi-agent 协作的一个特例。这个思路在 coding agent（写代码-运行测试-修 bug）和 RAG pipeline（检索-生成-验证）中都有对应物。inference-time scaling 的 100 倍效率提升也再次说明，在 reasoning-heavy 的任务上，给模型更多推理预算的 ROI 可能远超增加参数量。

原文链接：https://www.marktechpost.com/2026/03/13/google-deepmind-introduces-aletheia-the-ai-agent-moving-from-math-competitions-to-fully-autonomous-professional-research-discoveries/

论文 PDF：https://github.com/google-deepmind/superhuman/blob/main/aletheia/Aletheia.pdf
