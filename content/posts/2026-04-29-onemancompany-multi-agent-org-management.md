---
title: "OneManCompany: 用企业组织架构管理多 Agent 系统"
date: 2026-04-29T07:30:00-04:00
tags: [multi-agent, LLM, software-engineering]
description: "OMC 把企业管理模式（招聘-分工-考核-淘汰）搬进多 Agent 系统，在 PRDBench 上以 84.67% 成功率超越所有基线 15 个百分点。"
showToc: true
---

## 背景

当前多 Agent 框架的核心问题不在单个 Agent 能力不足，而在"组织层"缺失。现有方案要么预设固定工作流（LangGraph pipeline），要么让 Agent 自由协商（容易发散、无终止保证）。这篇来自伦敦大学学院的论文把视角拉到企业管理层面：与其让 Agent 各自为战，不如给它们一套组织制度。

## 核心机制

OMC 的设计围绕三个支柱。

### Talent-Container 架构：Agent 的"简历"与"工位"分离

每个 Agent 被拆成两部分：**Talent** 是可移植的认知身份包（prompt、角色定义、工具配置、技能），**Container** 是运行时环境（LangGraph、Claude Code、脚本等）。这种解耦意味着同一个 Talent 可以跑在不同 runtime 上，不同 runtime 的 Agent 可以在同一个项目里协作。

系统内置一个 **Talent Market**——社区驱动的 Agent 市场，HR Agent 可以在执行过程中按需"招人"补充能力缺口。

每个 OMC 实例启动时带四个默认角色：

| 角色 | 职能 |
|------|------|
| HR | 招聘管理 |
| EA | 执行助理 |
| COO | 运营 |
| CSO | 对外接口 |

CEO 是唯一的人类。

### E2R 树搜索：Explore-Execute-Review

这是论文最有意思的部分。OMC 把项目执行建模为类 MCTS 的树搜索，但区别在于：执行是真实的（Agent 产出实际代码/文档），review 信号来自显式的 supervisor 评估而非模拟回报。

三个阶段循环：

- **Explore**：分解任务、分配 Agent，面临 exploration-exploitation 权衡——用老手还是试新人
- **Execute**：Agent 在各自 Container 里执行，DAG 调度器处理依赖关系
- **Review**：reviewer 逐节点评审，质量信号自底向上传播，驱动下一轮迭代

每个任务节点有完整的有限状态机生命周期，加上三个断路器：

1. Review 轮次上限（默认 3 轮）
2. 超时（默认 3600 秒）
3. 成本预算

这保证了搜索在有界时间和成本内终止，不会出现任务永远挂起的情况。

### 自进化机制：绩效考核、PIP、自动解雇

每个 Agent 完成任务后做 post-task review，更新自己的 working principles（修改 Talent 而非重训模型）。项目结束后 COO 组织复盘，提炼出 SOP 注入后续项目。

更硬核的是：每三个项目做一次 performance review，连续三次不过进入 **PIP**（Performance Improvement Plan），PIP 期间再挂一次直接 offboard——Container 回收、工位释放、HR 从 Talent Market 重新招人。这套机制让组织能力不会被单个低效 Agent 拖垮。

## 实验结果

在 PRDBench（50 个项目级软件开发任务）上：

| 系统 | 成功率 |
|------|--------|
| OMC（Claude Code Sonnet 4.6 + Gemini 3.1 Flash Lite） | **84.67%** |
| Claude-4.5 单 Agent | 69.19% |
| GPT-5.2 单 Agent | 62.49% |
| CodeX 商业版 | 62.09% |
| Claude Code 商业版 | 56.65% |

总成本 $345.59（50 个任务），平均每个任务约 $6.9。

## 为什么值得关注

这篇论文的贡献不在于某个 Agent 更强，而在于提出了一个可操作的**组织层抽象**。当前多 Agent 系统的瓶颈确实不在单 Agent 能力（Claude Code 已经很强），而在协调、质量保证、和动态适应。OMC 用企业管理的成熟模式（招聘-分工-考核-淘汰）来解决这些问题，思路比"让 Agent 自己商量"要靠谱得多。

Talent-Container 分离的设计也值得注意——这本质上是 Agent 领域的 Docker 思想：把身份/能力与运行环境解耦，实现可移植和可组合。

**链接：**
- 论文：[arXiv:2604.22446](https://arxiv.org/abs/2604.22446)
- 项目主页：[one-man-company.com](https://one-man-company.com)
- GitHub：[1mancompany/OneManCompany](https://github.com/1mancompany/OneManCompany)

## 面试关联知识点

### Q1: 多 Agent 系统中，如何保证任务不会无限循环或死锁？

OMC 的方案是三层保护：(1) 每个任务节点有有限状态机，状态转移路径是有向无环的；(2) DAG 调度器保证依赖拓扑序执行，不允许环形依赖；(3) 三个断路器（review 轮次上限、超时、成本预算）兜底。形式化上，只要底层 executor 遵守超时约定，整个搜索一定在有界时间内终止。

### Q2: ReAct 框架和 E2R 树搜索的区别是什么？

ReAct 是单 Agent 的 Reasoning-Acting 循环，一个 Agent 在思考和工具调用之间交替。E2R 是组织级的搜索框架，操作对象是任务分解树而非单步 action，涉及多个异构 Agent 的协调。可以理解为 ReAct 解决"一个人怎么做事"，E2R 解决"一个团队怎么分工做事"。

### Q3: Agent 系统中 exploration-exploitation 权衡体现在哪？

传统理解是 decoding 层面（temperature、top-p），但在多 Agent 组织中体现在人员分配：是把任务给成功率高的 Agent（exploitation），还是试用新招的 Agent 来发现更优解（exploration）。OMC 通过 performance history 和 Talent Market 的动态招聘来平衡这个权衡。
