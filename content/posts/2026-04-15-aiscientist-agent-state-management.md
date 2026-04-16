---
title: "AiScientist：长周期 Agent 的瓶颈不是推理，是状态管理"
date: 2026-04-15T07:30:00-04:00
tags: [agent, research-engineering, multi-agent, state-management]
description: "长周期 ML research agent 的真正护城河不是单次推理更强，而是能不能把中间状态、失败证据和修复路径稳稳留下来。"
showToc: true
---

## 背景

最近大家聊 Agent，容易把重点全压在 planner、reasoning token 和模型本身上。但真正一跑到几小时甚至几天的 ML 研究任务，最先烂掉的通常不是"不会想"，而是"前面做过什么已经散了"。

[AiScientist](https://arxiv.org/abs/2604.13018v1) 瞄准的就是这个硬问题：给 agent 一篇 paper、一个干净的 Docker + GPU 环境和固定时间预算，让它从零复现实验。这个任务覆盖读论文、搭环境、拉数据、写代码、跑实验、查 bug、修结果——任何一环断档，前面努力基本白干。

作者引用的 PaperBench 结果很扎心：此前最强 agent 只有 21% 复现分，而 top ML PhD 在 48 小时预算下能到 41%。差距不是一点 prompt engineering 能糊过去的。

## 核心机制：Thin Control over Thick State

AiScientist 的核心设计可以概括成一句话：**thin control over thick state**。

顶层 Orchestrator 只保留轻量控制信息，把真正重要的项目状态外置到共享 workspace。这个 workspace 不是普通文件夹，而是系统的**唯一事实来源**：

- `paper_analysis/` 存论文理解、目标指标和歧义
- `submission/` 存可运行代码、配置和 `reproduce.sh`
- `agent/` 下存计划、实现日志、实验日志以及具体实验输出

后续 agent 不靠聊天上下文继承世界观，而是每次重新进入 workspace，按需读取工件，再写回新证据。

## 两个最聪明的设计

### File-as-Bus

作者把文件本身当成 agent 协作总线，用权限边界控制谁能改哪里，避免多个 agent 互相踩状态。

### Agent-as-Tool

顶层 Orchestrator 把 paper comprehension、prioritization、implementation、experimentation 这些 specialist 当成"可调用工具"，需要时再委派，而不是强行把所有步骤塞进一个巨大的对话链。好处很现实：顶层上下文不会越跑越臃肿，子任务也能围绕局部问题独立展开。

论文里还明确强调，真正的主循环不是单向流水线，而是 **evidence-driven loop**：先搭可运行 scaffold，再在 implement → run → diagnose → patch → re-validate 之间反复迭代，靠实验日志驱动下一步，而不是靠 agent 硬猜。

## 结果

- PaperBench 上相对 best matched baseline **平均提升 10.54 分**
- MLE-Bench Lite 上做到 **81.82% Any Medal**
- 消融实验：去掉 File-as-Bus，PaperBench 掉 6.41 分，MLE-Bench Lite 的 Any Medal **直接掉 31.82 个点**

这个数字说明一件事：长周期 agent 的真正护城河，不是一次调用时更聪明，而是能不能把中间状态、失败证据和修复路径稳稳留下来。

很多 agent demo 看着能做事，实际一长跑就失忆，像一群短期记忆只有金鱼水平的实习生。这篇论文算是把病根说透了。

## 对 Agent 工程的启发

1. **Artifact-first 的状态管理**：做 coding agent 或 research agent 时，memory 不该只理解成"把历史塞回 prompt"，而应该优先做 artifact-first 的状态管理。
2. **角色边界比数量重要**：多 agent 的价值不在"人多热闹"，而在角色边界、权限边界和 handoff 机制是否清楚。
3. **评价标准要升级**：别只看单轮成功率，真正该看的是 agent 能不能在 delayed feedback 下跨多轮稳定收敛。

换句话说，下一波更强的 agent 系统，未必先赢在更长 CoT，反而更可能赢在更像一个靠谱的软件工程系统。

## 原文

- [arXiv](https://arxiv.org/abs/2604.13018v1)
- [HTML 版](https://arxiv.org/html/2604.13018v1)
