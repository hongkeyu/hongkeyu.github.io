---
title: "Nemotron 3 Super: Mamba + MoE + FP4 预训练 + 原生投机解码，四合一"
date: 2026-04-16T07:30:00-04:00
tags: [LLM, MoE, Mamba, FP4, Speculative-Decoding, NVIDIA]
description: "NVIDIA 开源 Nemotron 3 Super：120B 总参 / 12B 活跃参的 Hybrid Mamba-Attention MoE 模型，首次 FP4 全量预训练，内置 MTP 实现原生 speculative decoding，推理吞吐比 Qwen3.5-122B 高 7.5 倍。"
showToc: true
---

NVIDIA 开源了 Nemotron 3 Super，一个 120B 总参 / 12B 活跃参的 Hybrid Mamba-Attention MoE 模型。它首次用 NVFP4 完成全量预训练，内置 MTP 层实现原生 speculative decoding，推理吞吐比同级别的 Qwen3.5-122B 高 7.5 倍。

这篇论文信息密度极高，值得逐块拆解。

## 背景

MoE 和 Mamba 是当前 LLM 推理效率优化的两条主线：

- **MoE**：通过稀疏激活让总参数量大幅膨胀而不增加每 token 计算量
- **Mamba**（状态空间模型）：用线性复杂度的序列建模替代 self-attention 的 O(n²) KV cache 开销

此前两条路各自发展，Nemotron 3 Super 是第一个把两者结合并跑通完整训练-部署流程的大规模开源模型。

## 核心架构：三根支柱

### 1. LatentMoE — 硬件感知的稀疏专家设计

传统 MoE 设计只关注 "accuracy per FLOP"，忽略了实际部署中的内存带宽和 all-to-all 通信瓶颈。LatentMoE 从硬件-软件协同设计的视角出发：

- **低延迟场景**：MoE 推理瓶颈是读取 expert 权重的内存带宽，代价正比于 d × m（隐藏维度 × expert FFN 中间维度），所以要压缩这两者
- **高吞吐场景**：瓶颈是 all-to-all routing，通信量正比于 d × K（K 为激活的 expert 数），所以也要压 d 或 K
- **方案**：用一个 latent projection（1024 维）压缩输入到 expert 的维度，然后用 512 个小 expert（中间维度仅 2688）+ top-22 routing

这使得模型同时优化了 accuracy per FLOP 和 accuracy per parameter —— 后者在实际部署中才是决定成本的关键指标。

### 2. Multi-Token Prediction (MTP) — 内建投机解码

模型训练时同时预测未来多个 token，训练好的 MTP head 在推理时直接充当 draft model，无需额外的小模型。

关键改进：多个 MTP head 共享权重，解决了传统 MTP 在自回归 draft 时的 train-inference distribution mismatch 问题。Draft 长度可以超过训练时的 head 数量，接受率更高。

### 3. Hybrid Mamba-Attention 交错架构

88 层网络以 Mamba-2 block 为主体，间歇插入少量 self-attention 层作为"全局锚点"：

- Attention 层用 GQA（32 query heads / 2 KV heads），支持 1M context length
- Mamba 层在生成阶段只维护常量大小的 state，KV cache 开销降到极低

## NVFP4 预训练

Nemotron 3 Super 是**第一个全程用 FP4 精度完成预训练的大模型**。所有线性层（除少数例外）的 fprop / dgrad / wgrad 都用 NVFP4 GEMM kernel。

具体策略：

| 组件 | 精度 | 原因 |
|------|------|------|
| 大部分线性层 | FP4 (2D block scaling) | 主力 |
| 梯度和激活 | FP4 (1D block scaling) | 量化到位 |
| 网络最后 15% 的层 | BF16 | 训练稳定性 |
| Attention QKV projection | BF16 | 训练稳定性 |
| MTP 层 | BF16 | 训练稳定性 |
| Mamba output projection | MXFP8 | FP4 下容易 underflow |

这证明了 FP4 预训练在 100B+ 规模是可行的，对未来在 Blackwell 架构上训练超大模型有直接参考价值。

## Post-Training：重注 Agent 能力

**SFT 采用两阶段 loss 设计：**
- Stage 1：token-level 平均 loss，激发强推理行为
- Stage 2：per-conversation 归一化，解决长输入短输出场景下 loss 被长 output 主导的问题

**RL 阶段**大幅扩展了 agentic 训练环境，用 SWE-Gym / R2E-Gym 的真实 GitHub issue 做软件工程任务蒸馏，teacher model 用 Qwen3-Coder-480B。模型还支持三种 reasoning 模式（off / regular / low-effort）。

## 为什么重要

- **吞吐**：8K input / 64K output 设置下，比 GPT-OSS-120B 快 2.2×，比 Qwen3.5-122B 快 7.5×
- **部署友好**：12B 活跃参数 + FP4 量化 = 更少 GPU 跑同级别能力
- **完全开源**：base / post-trained / FP4 / FP8 checkpoint 全部放出，连预训练数据集都开源了
- **架构启示**：LatentMoE 的"同时优化 accuracy per FLOP 和 per parameter"思路，对所有做 MoE 部署的人都有参考意义

## 面试关联知识点

### Speculative Decoding 原理

投机解码用一个小/快的 draft model 先生成 K 个候选 token，再用大模型一次 forward pass 并行验证。命中的 token 直接接受，不命中则从大模型的分布重新采样。

关键优势：在不改变输出分布的前提下减少大模型 forward 次数。Nemotron 3 Super 的创新是用 MTP head 替代外部 draft model，省去额外模型的显存和加载开销。

### MoE Routing 设计

传统 MoE（如 Mixtral）用 top-K gating 在 8 个大 expert 中选 2 个。Nemotron 3 Super 反其道行之：512 个小 expert + top-22，配合 latent projection 降维。

面试要点：expert 数量 vs 大小的 tradeoff 取决于部署硬件的 memory bandwidth 和 all-to-all 通信代价，不是越大越好。

### KV Cache 与 Mamba 的关系

Self-attention 的 KV cache 随序列长度线性增长，是长上下文推理的主要显存瓶颈。Mamba 用固定大小的 state（类似 RNN 的 hidden state）替代 KV cache，生成阶段内存占用恒定。但纯 Mamba 在需要全局信息检索的任务上弱于 attention，所以 Nemotron 3 用少量 attention 层做"锚点"来补偿。

---

**原文链接：** https://arxiv.org/abs/2604.12374

**HuggingFace：** https://huggingface.co/papers/2604.12374
