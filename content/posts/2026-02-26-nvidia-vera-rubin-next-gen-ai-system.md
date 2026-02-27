---
title: "NVIDIA Vera Rubin: 130万个零件构成的下一代AI系统，推理成本降至Blackwell的十分之一"
date: 2026-02-26T07:30:00+08:00
tags: [GPU架构, 推理优化, NVIDIA]
description: "NVIDIA Vera Rubin 首批样片交付，每瓦性能 10 倍于 Blackwell，推理 token 成本降 10 倍，首个 100% 液冷 AI 系统"
showToc: true
---

TL;DR: NVIDIA 首批 Vera Rubin 样片已交付客户，每瓦性能是 Grace Blackwell 的 10 倍，推理 token 成本降低 10 倍，训练 MoE 模型所需 GPU 数量降至 Blackwell 的四分之一。预计 2026 下半年出货。

---

## 背景

Grace Blackwell 在 2024 年量产后重新定义了 rack-scale AI 计算的上限。但 AI 基础设施的核心矛盾从未改变：算力需求指数增长，而能耗和成本不可能同步线性扩张。Vera Rubin 是 NVIDIA 对这个矛盾给出的下一个答案。

CNBC 本周独家探访了 NVIDIA 圣克拉拉总部，首次公开展示了 Vera Rubin NVL72 机架的完整形态。Jensen Huang 在今年 1 月的 CES 上宣布该系统已进入全面量产。

## 核心硬件配置

一个 Vera Rubin NVL72 机架包含：
- 72 颗 Rubin GPU + 36 颗 Vera CPU（88核 ARM 架构）
- 每颗 Rubin GPU 配备 288 GB HBM4，单 GPU 内存带宽达 1.2 TB/s
- 总计约 1300 颗芯片、130 万个组件
- 18 个计算托盘（compute tray），每个托盘可热插拔——这是相比 Blackwell 的重大改进，后者的组件是焊死在主板上的
- 整机重量接近 2 吨

## NVLink 6 Spine

第六代 NVLink 互连提供每机架 260 TB/s 的聚合带宽。NVIDIA 称其支持零停机维护和机架级 RAS（可靠性、可用性、可维护性）服务。模块化设计意味着故障组件可以在不影响整体运行的情况下替换。

## 散热与能耗

Vera Rubin 是 NVIDIA 首个 100% 液冷系统。每个 SuperChip（2 颗 Rubin GPU + 1 颗 Vera CPU，共约 17000 个组件）都通过专用冷板（cold plate）进行液冷散热。NVIDIA 称这种设计比传统蒸发式冷却大幅减少了水的消耗。

整机功耗大约是 Blackwell 的两倍，但因为性能提升了 10 倍/瓦，实际的推理效率远高于前代。Mizuho 分析师 Jordan Klein 的评价很直接：关键指标是"每瓦能产出多少 token"，这个比值越高，每一美元的回报就越大。

## 竞争格局

Vera Rubin 的首批客户名单包括 Meta、OpenAI、Anthropic、Amazon、Google、Microsoft。其中 Meta 上周刚宣布将在 2027 年的数据中心中部署 Vera Rubin。

但竞争也在加剧。AMD 的首个 rack-scale 系统 Helios 预计今年出货，Meta 同时也给了 AMD 高达 6GW 算力容量的承诺。Google 的 TPU、Amazon 的 Trainium 2 也在持续扩张自研芯片的份额。Futurum Group 估计 Vera Rubin 机架售价约 350-400 万美元，比 Blackwell 贵约 25%。

NVIDIA 基础设施负责人 Dion Harris 对竞争对手的回应是："Hats off to anyone who's going to try. But this is certainly not a simple endeavor."

## 对边缘推理的启示

虽然 Vera Rubin 是数据中心级产品，但它的设计哲学对边缘推理也有参考价值：模块化、液冷、每瓦效率优先。NVIDIA 同时还发布了 Rubin CPX（配备 128 GB GDDR7 而非 HBM4），定位更偏向推理和中等规模部署，这可能是未来 Jetson 系列产品线演进的参照方向。

---

## 面试关联知识点

### 1. KV Cache 与推理优化

Vera Rubin 每颗 GPU 288 GB HBM4 的配置，直接影响的就是 KV Cache 的容量上限。KV Cache 在自回归解码时缓存之前 token 的 Key/Value 向量，避免重复计算。HBM4 的带宽提升（1.2 TB/s）意味着 decode 阶段的 memory-bound 瓶颈被进一步缓解。面试常问：Prefill 阶段是 compute-bound（大量并行矩阵乘），Decode 阶段是 memory-bound（逐 token 读取 KV Cache），两者对硬件的需求完全不同。

### 2. 模型量化与硬件适配

Rubin CPX 用 GDDR7 而非 HBM4，带宽低但容量/成本比更好，天然适合跑量化模型。量化的本质是用低精度（INT4/INT8/FP8）表示权重和激活值，换取更小的内存占用和更高的吞吐。面试考点：PTQ（Post-Training Quantization）vs QAT（Quantization-Aware Training）的区别——PTQ 不需要重新训练但精度损失更大，QAT 在训练时模拟量化误差因此精度更高。GGUF 格式支持混合精度（不同层用不同量化位数），这是 llama.cpp 生态的核心。

### 3. NVLink vs PCIe vs Ethernet 在分布式训练中的角色

NVLink 6 的 260 TB/s 机架内带宽是 All-Reduce 梯度同步的物理基础。面试常问：All-Reduce 的 Ring 算法——N 个 GPU 形成环，每个 GPU 发送 1/N 的梯度给下一个节点，经过 2(N-1) 步完成全部同步。NVLink 解决的是节点内高带宽互连，跨节点仍然依赖 InfiniBand 或 Ethernet（Vera Rubin 部署中首次出现 Co-Packaged Optics）。

---

## 进一步讨论：Qwen3.5-27B Thinking 模式实测

当天频道里还讨论了一个有趣的实测：Qwen3.5-27B 在 Vercel AI SDK 下的 Thinking ON vs OFF 对比。

**测试环境：** RTX 4090，16GB GGUF 量化，llama.cpp 推理，96K 上下文。17 项测试覆盖工具调用、结构化输出、Agent Loop、纯生成四类能力。

**关键结论：**

1. **通过率一样：15/17 vs 15/17。** Thinking 开不开，能做的事一样多。
2. **但 OFF 省了约 82% 的 token。** 平均 1845 → 321 token。省钱省时间，结果一样。
3. **速度差不多**（37.8 vs 35.6 tok/s）。
4. **两个失败项有意思：** 角色一致性（多轮保持人设）两边都挂，是 27B 的硬伤；逻辑推理真假 ON 反而挂了、OFF 过了——thinking 把自己绕晕了。

**一句话总结：** 跑 agent/工具调用场景，Qwen3.5-27B 关掉 thinking 就行——省 80% token，效果不降。Thinking 模式更适合数学、代码等需要长链推理的任务，在这类结构化工具调用场景里基本是浪费。

---

原文链接：
- [CNBC: First look at NVIDIA's Vera Rubin](https://www.cnbc.com/2026/02/25/first-look-at-nvidias-ai-system-vera-rubin-and-how-it-beats-blackwell.html)
- [Tom's Hardware: NVIDIA delivers first Vera Rubin samples](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-delivers-first-vera-rubin-ai-gpu-samples-to-customers-88-core-vera-cpu-paired-with-rubin-gpus-with-288-gb-of-hbm4-memory-apiece)
