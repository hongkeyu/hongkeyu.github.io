---
title: "MMSkills：给视觉 Agent 装上多模态操作手册，OSWorld 成功率翻倍"
date: 2026-05-18T07:30:00-04:00
tags: [agent, multimodal, gui-agent, skill-learning, osworld]
description: "上海交大与小红书提出 MMSkills，将 Agent 技能从纯文本升级为文本流程+状态卡+关键帧截图的多模态包，配合 branch-loaded 推理机制，Qwen3-VL-235B 在 OSWorld 上从 21.3% 飙到 39.2%。"
showToc: true
---

## 背景：Agent 的技能为什么需要"带图"

当前主流 Agent 系统（Voyager、AgentTrek 等）把可复用技能编码为文本 prompt、代码片段或 learned routine。纯文字交互场景下这够用，但视觉 Agent 面对的是屏幕截图——它不仅需要知道"点哪里"，还需要识别"当前是不是正确的状态"、"操作完成了没有"。

纯文本技能的困境：要么写得冗长但仍然欠缺关键视觉信息，要么直接塞 demo 截图但太长、太 instance-specific、难以泛化。

## 核心机制：三层结构的多模态技能包

MMSkills 把每个技能定义为一个四元组 (D, P, S, K)：

| 组件 | 含义 | 作用 |
|------|------|------|
| **D (Descriptor)** | 技能的紧凑描述 | 用于检索匹配 |
| **P (Procedure)** | 可复用的文本流程 | 和传统 text skill 一致 |
| **S (State Cards)** | 运行时状态卡 | 每张卡对应一个关键决策点，包含 when-to-use、when-not-to-use、visible cues、verification cue、available views |
| **K (Keyframes)** | 多视角截图包 | 全局视图 + focus crop + before/after 视图 |

纯文本技能是退化形式 (D, P, ∅, ∅)；MMSkills 把流程、决策条件和视觉证据绑定成一个可复用单元。

## 技能生成：从公开轨迹自动提炼

生成流水线分五阶段：

1. **Task embedding + clustering**：把任务指令和轨迹元数据聚类，将大领域切分成语义聚焦的小组
2. **Cluster-level skill planning**：LLM agent 为每个聚类规划原子技能，划定工作流边界和完成条件
3. **Skill merging**：跨聚类去重合并
4. **Per-skill generation**：对每个技能，从对应轨迹中提取流程文本、状态卡、关键帧截图
5. **Meta-skill-guided auditing**：用元技能做质量审核

整个流程用的都是非评测数据的公开轨迹，和测试集完全隔离。

## 推理时的 Branch-Loaded 机制

这是设计中最精巧的部分。推理时不是把技能截图全塞进 context（会导致 context 爆炸和 over-anchoring），而是用一个"分支"机制：

- Agent 先从技能库预召回候选集
- 遇到需要参考技能的决策点时，在一个临时 branch 里加载选中的状态卡和关键帧
- Branch 做技能-环境对齐，输出一个结构化 guidance tuple：适用性判断、局部子目标、技能条件下的计划、负约束、视觉验证检查
- 这个 guidance 传回主 Agent 作为决策参考，但实际动作 grounding 仍然绑定在实时观察上

好处：技能的视觉信息不污染主轨迹的 context，Agent 不会盲目复制参考截图里的坐标，而是理解"这个状态意味着什么"再做决策。

## 实验结果

在 OSWorld（桌面 GUI 任务）上的提升非常显著：

| 模型 | 基线 | +MMSkills | 提升 |
|------|------|-----------|------|
| Gemini 3.1 Pro | 44.1% | 50.1% | +6.0pp |
| Gemini 3 Flash | 36.7% | 48.0% | +11.3pp |
| Qwen3-VL-235B | 21.3% | 39.2% | +17.9pp |
| GLM-5V / Kimi-K2.6 | — | — | 一致提升 |

跨场景泛化同样成立：macOSWorld、VAB-Minecraft、Super Mario Bros 上都观察到成功率和平均分数的一致提升。文本技能（text-only skill）有帮助但跨领域不稳定，说明光有流程不够，视觉证据是关键增量。

消融实验确认：状态卡和关键帧各自贡献不可替代，branch-loading 比直接把截图塞进 context 效果好。

## 为什么重要

这篇论文回答了一个 Agent 领域的核心问题：**可复用技能的表示应该是什么形态**。结论是，对于视觉 Agent，纯文本不够，需要把"什么时候用、怎么判断状态、视觉上长什么样"一起编码进去。Branch-loaded 机制也给了一个实用的工程模式——怎么在不爆 context 的前提下让 Agent 参考多模态技能。

更大的图景：随着 GUI Agent 和具身智能成为热点赛道，"技能库"的构建范式正在从 text-only 向 multimodal 演进，MMSkills 是这个方向上一个相当完整的框架。

## 延伸思考

- 技能生成流水线本身可以看作一种"经验蒸馏"——从粗糙轨迹中提取结构化知识
- Branch-loaded 机制和 RAG 的 retrieval-then-read 模式有异曲同工之处，只是检索的不是文档而是操作手册
- 论文用的 meta-skill auditing 思路值得关注——用一个"造技能的技能"来质量控制，本质上是 self-improvement loop

## 面试关联知识点

### ReAct 框架与 Agent tool use 的局限性

ReAct 让 LLM 交替进行 reasoning 和 action，但它假设环境状态可以用文本充分描述。视觉 Agent 的核心挑战在于：observation 是高维图像，光靠文本 reasoning 无法可靠判断"对话框是否已弹出"这类视觉状态。MMSkills 的方案是把视觉判断条件显式编码到技能中，让 Agent 有"操作手册式"的视觉参考。

### 多模态模型中 visual token 如何影响 Agent 决策

LLaVA 等架构把图像编码为 visual token 序列输入 LLM。MMSkills 的 branch-loading 机制本质上是在管理 visual token 的注入方式——不是把所有参考截图的 visual token 全塞进主 context（会稀释注意力、增加推理开销），而是在隔离的分支中做视觉对齐，只把结构化文本 guidance 传回主轨迹。

### Agent 系统中"技能复用"与 RAG 的关系

MMSkills 的检索-加载-对齐-生成 pipeline 和 RAG 高度同构：descriptor 对应 embedding 索引，state card 对应 retrieved chunk，branch-loaded alignment 对应 reranking + grounding。区别是 RAG 检索的是文档段落，MMSkills 检索的是多模态操作手册。

---

- 原文链接：https://arxiv.org/abs/2605.13527
- 项目主页：https://deepexperience.github.io/MMSkills/
- 代码：https://github.com/DeepExperience/MMSkills
