---
title: "OLMo Hybrid: 混合架构凭什么比纯 Transformer 强？附 Nemotron 3 Super 对比"
date: 2026-03-12T07:30:00+08:00
tags: ["hybrid-architecture", "MoE", "LLM"]
description: "Ai2 发布 OLMo Hybrid 7B，混合 GDN-Attention 架构训练效率翻倍；NVIDIA Nemotron 3 Super 用 Mamba-2 + Latent MoE 实现 120B 参数仅 12B 激活。"
showToc: true
---

## OLMo Hybrid: 混合架构凭什么比纯 Transformer 强？

"Transformer is all you need" 这句话可能要加个问号了。

上周（3月6日），Allen Institute for AI（Ai2）正式发布了 OLMo Hybrid，一个 7B 参数的全开源混合架构语言模型。模型权重、训练代码、数据配比、训练日志全部公开——这在混合架构模型中是独一份的透明度。Nathan Lambert 在 Interconnects 上写了详细的分析，Lambda AI 也发布了训练侧的技术报告。

### 背景：混合架构的第二波浪潮

2023 年底 Mamba 横空出世，提出用状态空间模型（SSM）替代 attention，引发了"是否需要全 attention"的讨论。但早期的纯 RNN/SSM 模型在 scale up 时性能下降，热度很快退去。

2025-2026 年，混合架构卷土重来，而且这次是集体行动：Qwen 3.5 用了 Gated DeltaNet，Kimi Linear 也是混合架构，NVIDIA Nemotron 3 Nano 用了 Mamba 层，IBM Granite 4 同样走了混合路线。OLMo Hybrid 是这波浪潮中开源程度最高、实验对照最严格的一个。

### 核心设计：3:1 的 GDN-Attention 交替

OLMo Hybrid 的架构很直白：保留标准 Transformer 的整体 layout，但把 75% 的 attention 层替换为 Gated DeltaNet（GDN）层，每 3 层 GDN 接 1 层标准 multi-head attention。GDN 本质上是一种线性 RNN：每个 head 维护一个线性递归状态（recurrent state），用 query/key/value 加一个学习到的 gate 来更新。它避开了 attention 的 O(n^2) 复杂度，用固定大小的隐状态来压缩历史信息。

为什么是 3:1 而不是全换？Ai2 的 scaling 实验给出了清晰的排序：

hybrid GDN (3:1) > pure GDN > standard transformer > hybrid Mamba2 > pure Mamba2

关键发现是这些差距在参数量和计算量增大时保持甚至扩大——这意味着混合 GDN 架构的 scaling law 本身就更优。

### 理论支撑：表达力 → 数据效率 → 更好的 scaling

配套论文的理论部分是最有价值的。核心论证链条是：

1. Attention 和 RNN 有互补的表达力。Transformer 擅长 recall（从上下文中精确检索信息），RNN 擅长 state tracking（跟踪和更新状态）。已有理论工作证明了这一点（Merrill et al., 2024）。

2. 混合架构的表达力严格大于两者之和。论文构造了一类和代码执行相关的形式化问题，纯 Transformer 和纯 GDN 都无法表达，但混合架构可以——不仅理论上能表达，实验上也能学到。

3. 更强的表达力为什么带来更好的 scaling？论文引用了 neural scaling 的 quantization model：语言建模本质上是一个 multi-task 目标（每个 token 位置可能需要不同的子技能），更强表达力的架构能覆盖更多子任务类型，因此在相同数据量下学到更多，体现为更陡的 scaling curve。

这套论证的核心 insight 是：架构的表达力不是锦上添花，而是直接影响 scaling law 斜率的基础性因素。

### 实际表现：预训练大幅提升，post-training 有坑

预训练阶段，OLMo Hybrid 在 3T tokens 上训练（512 张 B200 GPU，7 天完成），相对 OLMo 3 实现了约 2 倍的训练效率提升。具体 benchmark：MMLU STEM +4.5，MBPP（代码）+6.7，MedQA +7.1，长上下文能力提升更为显著。

但 post-training（SFT + RL）阶段出了问题：直接套用 OLMo 3 的 Tulu 3 配方，知识类任务有提升，但 extended reasoning（长链推理）反而下降了。Ai2 的猜测是：混合架构作为一个"不同的学生模型"，它的学习特性和纯 Transformer 不同，因此需要不同的 teacher 数据。这是一个重要的开放问题——不同架构可能需要定制化的 post-training 数据管线。

### 现实困境：推理工具链拖后腿

理论上混合架构的最大卖点之一是长上下文生成时的内存效率——RNN 层不需要 KV cache，隐状态大小固定。但现实很残酷：当前 vLLM 对 GDN 模型的 kernel 支持不成熟，存在数值稳定性问题。为了拿到正确结果，需要关掉 CUDA graphs（--enforce-eager）、关掉 cascade attention（--disable-cascade-attn）、把 RNN cache 提升到 FP32。代价是推理吞吐量暴跌，理论上的效率优势被完全抵消。Nathan Lambert 估计需要 3-6 个月才能让 OSS 推理栈真正支持好这些模型。

### 对边缘部署的启示

这对 Jetson 这类边缘设备尤其值得关注。混合架构天然适合长上下文场景（Agent、RL），因为 RNN 层的内存占用是 O(1) 而非 O(n)。一旦推理工具链成熟，7B 混合模型在边缘设备上跑长对话的可行性会比纯 Transformer 高得多。但目前 TensorRT 对 GDN 的支持还是空白，短期内想在 Jetson 上部署 OLMo Hybrid 不现实。

### 一个有趣的问题

Nathan Lambert 在文末抛了个猜测：GPT、Claude 这些闭源 frontier 模型是否已经在用混合架构？他给了大约 50% 的概率。逻辑是：如果混合架构的 scaling 优势在 frontier scale 依然成立，经济上很难忽略。但闭源模型可能有其他效率更高的架构方案。

原文链接：
- [Interconnects 分析](https://www.interconnects.ai/p/olmo-hybrid-and-future-llm-architectures)
- [Lambda 训练细节](https://lambda.ai/blog/open-model-open-metrics-how-lambda-and-the-olmo-team-trained-olmo-hybrid)
- [论文](https://allenai.org/papers/olmo-hybrid)
- [模型权重](https://huggingface.co/collections/allenai/olmo-hybrid)

---

## 进一步讨论：NVIDIA Nemotron 3 Super

OLMo Hybrid 用的是 GDN + Attention 的混合路线，而 NVIDIA 几乎同一时间（3月10日）发布的 Nemotron 3 Super 走了另一条混合路线——Mamba-2 + Transformer + MoE 三种层交替排列，且在架构创新上更加激进。

### 基本参数

120B 总参数，仅 12B 激活参数（MoE），100万 token 上下文窗口。

### 架构五大创新

**1. 混合 Mamba-Transformer 骨干**

三种层交替排列：Mamba-2 层处理大部分序列（线性复杂度，撑起 1M 上下文），Transformer attention 层穿插在关键位置（保证精确检索能力），MoE 层扩展参数容量但不增加推理成本。这和 OLMo Hybrid 是同一条路线，但 NVIDIA 用的是 Mamba-2 而非 GDN。

**2. Latent MoE（潜空间路由）**

标准 MoE 直接在完整 hidden dimension 上路由 token 到 expert。Latent MoE 先把 token 压缩到低秩潜空间，在小维度上做 expert 计算，再投影回去。好处是同样的计算成本可以调用 4 倍数量的 expert，实现更细粒度的专业化——比如 Python 语法和 SQL 逻辑可以走完全不同的 expert。这是个新概念，之前没在其他模型里见过。它本质上是在 MoE 路由之前加了一层 bottleneck（类似 autoencoder 的 encoder），让 expert 在压缩空间里工作。

**3. Multi-Token Prediction (MTP)**

训练时预测多个未来 token，不是只看下一个。两个好处：训练时迫使模型学习更长程的逻辑依赖（reasoning 更强）；推理时自带 speculative decoding，一次前向预测多个 token 再并行验证，代码/tool call 等结构化生成最高 3x 加速，不需要额外的 draft model。

**4. 原生 NVFP4 预训练**

不是训完再量化（PTQ），而是从第一个梯度更新开始就在 NVFP4（4-bit 浮点）精度下训练。模型从头学会在 4-bit 算术约束下保持准确性。在 Blackwell B200 上，推理速度比 H100 上的 FP8 快 4 倍。

**5. 多环境 RL 后训练**

用 NeMo Gym 在 21 种环境配置下做 RL，超过 120 万次 rollout。不是通常的"RLHF 对齐人类偏好"，而是在真实 agent 环境（代码执行、工具调用、多步规划）中强化。

### 性能数据

- 吞吐量：比 GPT-OSS-120B 快 2.2x，比 Qwen3.5-122B 快 7.5x（8k input / 16k output 场景）
- 精度：多数 benchmark 持平或超过 GPT-OSS-120B 和 Qwen3.5-122B
- PinchBench（Agent 能力评测）：85.6%，同级开源模型最佳
- RULER 1M 长上下文：超过上述两个模型

完全开源：权重（BF16/FP8/NVFP4）、训练数据、SFT 数据集、RL 环境配置、模型 recipe 全部公开。

### 和 OLMo Hybrid 的关联

两个模型验证了同一个趋势：混合架构正在从实验走向工业级。OLMo Hybrid 提供了最严格的消融实验证明混合架构 scaling 优势；Nemotron 3 Super 则展示了在更大规模上的工程落地。OLMo Hybrid 证明了混合架构 2x 训练效率优势，Nemotron 是工业级验证。MTP Self-Distillation 论文提出的方法，Nemotron 直接把 MTP 用在了 speculative decoding。
