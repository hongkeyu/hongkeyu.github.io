---
title: "pQuant: 用参数民主化解释 1-bit LLM 的瓶颈并修复它"
date: 2026-03-01T07:30:00+08:00
tags: [quantization, MoE, edge-deployment]
description: "北大团队提出 pQuant，发现 1-bit QAT 模型的核心瓶颈是参数民主化——所有权重敏感度被拉平，通过双分支结构+MoE 扩展实现 perplexity 降 32%、吞吐量超 FP16 两倍。"
showToc: true
---

TL;DR: 北大团队提出 pQuant，发现 1-bit QAT 模型的核心瓶颈是"参数民主化"——所有权重的敏感度被拉平，表达力坍缩。解法是把 linear layer 拆成 1-bit 主干 + 8-bit 高精度分支，用 feature scaling 引导敏感参数流向高精度分支，再扩展为稀疏激活的 MoE。结果：perplexity 比 SOTA 1-bit 基线降 32%，吞吐量超 FP16 模型 2 倍以上。

## 背景：sub-2-bit 量化卡在哪

极低比特量化（sub-2-bit）是边缘部署的终极方案：把浮点矩阵乘法替换成 bitwise 运算，内存和计算开销断崖式下降。BitNet 开创了从零训练 1-bit LLM（QAT-Scratch）的范式，BitNet 1.58 在 2-bit 下几乎无损。但 1-bit 模型始终有两个硬伤：精度差距不可忽略（下游任务只能恢复到 FP16 的约 80%），而且 scaling 效率极差——模型变大带来的增益远低于 FP16 的同等放大。

过去的解释多归咎于"信息容量不够"，但这篇论文给了一个更精确的诊断。

## 核心发现：Parameter Democratization

作者用 perturbation-based sensitivity metric（基于 Optimal Brain Surgeon 框架）分析了 FP16 LLaMA-3 和 1-bit BitNet 的权重敏感度分布。结果非常直观：

- FP16 模型中，少数权重的敏感度显著高于其余（符合量化理论的经典认知：存在一小撮"关键参数"对输出影响巨大）
- 1-bit BitNet 中，敏感度分布几乎完全平坦——所有参数被"民主化"了

这意味着 1-bit 训练过程中，模型丧失了区分"重要"和"不重要"参数的能力。所有权重被压到 +1/-1 两个值，梯度信号无法在参数之间建立差异化的重要性层级。这不仅降低了表达力，还解释了为什么 scaling 不好：增加参数量只是增加了更多"同质化"的权重，边际收益递减。

## pQuant 的解法：Decoupled Linear Layer

思路很清晰：既然 1-bit 会把所有参数拉平，那就结构性地"保护"一部分参数。具体做法是把 FFN 中的 linear layer 拆成两个并行分支：

1. **1-bit 主分支**：覆盖绝大多数参数，负责主要计算，保持 bitwise 运算的效率优势
2. **8-bit 高精度分支**：维度远小于主分支（r 远小于 D_model），专门承载最敏感的参数

比如 D=4096，r 可能只有 128 或 256。主分支 4096×4096 = 16M 参数用 1-bit，高精度分支 4096×128 = 512K 参数用 8-bit。总参数量只多了约 3%，但关键信息被 8-bit 分支保护住了。

```
输入 x (维度 D)
    │
    ├──→ W_1bit (D × D) ──→ y1    ← 1-bit 主分支，大矩阵
    │
    └──→ W_8bit (D × r) ──→ y2    ← 8-bit 高精度分支，小矩阵（r << D）

输出 y = y1 + y2
```

关键设计在于不是人为指定哪些参数重要，而是通过 feature scaling 机制让模型自己学：训练过程中，输入特征经过可学习的缩放因子，幅度大的特征自然被路由到高精度分支处理，幅度小的留给 1-bit 分支。这样敏感度的差异化就被重新建立起来了。

MHA 部分没有用这个双分支设计，而是全部 1-bit。理由是两个：FFN 层的敏感参数浓度更高（这是 SPQR 等工作早就验证过的），而且 FFN 的激活分布更不规则、outlier 更多，更需要高精度保护。

## Feature Scaling：让模型自己学参数重要程度

这是最巧妙的部分。不是直接标记"哪个权重重要"，而是让输入特征的幅度来自然引导。

每个输入 x 过一个可学习的缩放向量 s（和 x 同维度）：

```
x_scaled = x ⊙ s    （逐元素乘）
```

训练过程中，s 的某些维度会变大、某些会变小。幅度大的维度意味着"这个特征方向对输出影响大"——自然就是敏感的部分。然后 1-bit 主分支处理原始 x，8-bit 分支处理 x_scaled。因为 8-bit 分支输入被 s 放大了，那些 s 值大的维度在 8-bit 分支里产生更大的梯度信号，训练时 8-bit 分支的权重就会自动适应这些敏感方向。

s 就是一个普通的可学习参数，和模型权重一样通过反向传播更新：

```
前向：x_scaled = x ⊙ s → 送入 8-bit 分支 → 得到 y2 → 合并 y1+y2 → 最终 loss

反向：∂L/∂s = ∂L/∂y2 · ∂y2/∂x_scaled · ∂x_scaled/∂s
                                              ↑
                                           = x（因为 x_scaled = x⊙s，对 s 求导就是 x）
```

s 初始化为全 1（不缩放），训练过程中如果某个维度 i 对 loss 影响大，`∂L/∂s_i` 就大，s_i 会被推向更大的值；如果某个维度 j 对 loss 不重要，`∂L/∂s_j` 很小，s_j 基本保持不变或变小。

本质上 s 和 LoRA 里的 scaling factor、Batch Normalization 里的 gamma 是同一类东西——可学习的逐元素缩放，没有什么特殊的训练机制。

## 稀疏 MoE 扩展

8-bit 分支可以自然地看作一个"专家"。作者把它扩展为多个稀疏激活的 expert，配合 top-1 router，每个 token 只激活一个 expert。

```
输入 x
    │
    ├──→ W_1bit (共享) ──→ y1
    │
    ├──→ Router(x) → 选出 expert_k
    │
    ├──→ Expert_1 (8-bit, D×r)
    ├──→ Expert_2 (8-bit, D×r)
    ├──→ ...
    └──→ Expert_N (8-bit, D×r)
         只有 expert_k 被激活 ──→ y2

输出 y = y1 + y2
```

Router 就是一个小的线性层 + softmax：`scores = softmax(W_router · x)`，取分数最高的那个 expert（top-1）。每个 token 只激活一个 expert，所以推理时计算量 = 1-bit 主分支 + 1 个 8-bit expert，跟不用 MoE 时一样。但模型总容量变成了 N 个 expert 的并集。

Expert 的创建方式很直接——随机初始化，然后端到端训练：

```python
class MoEDecoupledLinear(nn.Module):
    def __init__(self, D, r, num_experts):
        self.W_1bit = BitLinear(D, D)          # 1-bit 主分支
        self.experts = nn.ModuleList([
            nn.Linear(D, r)                     # N 个 8-bit 小矩阵
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(D, num_experts) # 路由器
        self.s = nn.Parameter(torch.ones(D))    # 缩放向量

    def forward(self, x):
        y1 = self.W_1bit(x)
        scores = softmax(self.router(x))  # [batch, num_experts]
        k = scores.argmax(dim=-1)         # top-1 选择
        x_scaled = x * self.s
        y2 = self.experts[k](x_scaled)
        return y1 + y2 * scores[k]  # 乘以 router 权重
```

训练过程中 router 学会根据输入特征把不同类型的 token 分给不同 expert，每个 expert 因为"看到"的 token 分布不同，权重逐渐特化。这和 Mixtral、Qwen3.5 等 MoE 模型的 expert 创建方式完全一样——都是随机初始化 + 联合训练。

一个实际的负载均衡问题：如果不加约束，router 可能会把所有 token 都路由到同一个 expert（winner-take-all 坍缩）。标准做法是加一个 auxiliary loss：

```
L_balance = α · Σ_i (f_i · p_i)
```

f_i 是 expert_i 被选中的频率，p_i 是平均路由概率。这个 loss 惩罚"某个 expert 被选得太频繁"的情况，迫使 router 均匀分配。

## 关于 Shadow Weights 和训练机制

1-bit 训练的一个核心问题：权重只有 +1/-1 两个值，梯度没法直接更新（你不能给 +1 加一个 0.001 的梯度，结果还是 +1）。

解法是维护一份 FP16（半精度浮点数，1 bit 符号 + 5 bit 指数 + 10 bit 尾数）精度的"影子权重"。训练时的流程：

1. 保存一份 FP16 的完整权重 W_fp16
2. 前向传播时，对 W_fp16 做 sign 量化得到 W_1bit（+1/-1），用 W_1bit 算 loss
3. 反向传播时，梯度通过 STE（Straight-Through Estimator）直接传回 W_fp16
4. 用优化器更新 W_fp16（比如 Adam 把它从 0.37 更新到 0.35）
5. 下次前向传播再量化：sign(0.35) = +1

更准确的说法是：**1-bit 前向 + 1-bit 计算 loss + FP16 存储和更新 + 每步重新量化**。推理时只保留 1-bit，FP16 副本丢掉。

## 实验结果

- WikiText2 perplexity 比 SOTA 1-bit 基线降低 32%
- 精度超过 2-bit 量化模型，同时推理吞吐量高 18.2%
- 达到 FP16 模型的精度水平，吞吐量超过 FP16 两倍以上
- Scaling 行为明显改善：模型变大时，性能增益不再严重衰减

训练细节上，FP16 shadow weights 仅用于训练阶段保证梯度稳定性，推理时丢弃，只保留 1-bit 和 8-bit 参数。量化方式是标准的 sign 函数 + mean centering + AbsMax INT8 量化。

## 为什么这篇值得读

"Parameter democratization" 是一个很好的概念贡献。过去我们知道极低比特量化效果差，但解释通常停留在"精度不够"这个层面。这篇论文把问题精确定位到敏感度分布的坍缩，而且给出了可视化证据（FP16 vs 1-bit 的 sensitivity heatmap 对比非常直观）。解法也不是硬凑的——双分支 + feature scaling 引导 + MoE 扩展，每一步都有清晰的动机。

对边缘部署来说，这意味着 1-bit 模型不再是"只能勉强用"的状态。在 Jetson 这样的设备上，bitwise 运算的硬件优势是实打实的，如果精度能接近 FP16，那部署价值就完全不一样了。

原文链接：https://arxiv.org/abs/2602.22592

## 面试关联知识点

**1. 模型量化中"敏感权重"的概念和处理方式**

量化理论认为并非所有权重同等重要。敏感度可用 Hessian-based metric 衡量：s_ij = w_ij^2 / (XX^T)^{-1}，即权重大小与输入协方差逆的联合度量。处理方式包括：混合精度（SPQR 对 outlier 权重保留高精度）、分组量化（GPTQ 按列分组逐步量化并补偿误差）、以及本文的结构化分支方案。

**2. QAT vs PTQ 的区别和适用场景**

PTQ（Post-Training Quantization）在训练完成后直接量化，依赖校准数据，适合 4-bit 以上精度，代表方法 GPTQ/AWQ。QAT（Quantization-Aware Training）在训练过程中模拟量化误差，模型能"适应"低精度表示，精度更好但训练成本高。QAT-Scratch 是 QAT 的极端形式：从零开始用低精度训练，不依赖预训练权重，BitNet 系列是代表。在 sub-2-bit 场景下，PTQ 效果急剧恶化，QAT-Scratch 是目前唯一可行的路线。

**3. MoE（Mixture of Experts）的核心机制**

MoE 通过路由器（router）将不同 token 分配给不同的 expert 子网络，推理时只激活部分 expert（通常 top-1 或 top-2），实现"增加模型容量但不增加计算量"。关键挑战包括负载均衡（避免所有 token 涌向同一个 expert）和通信开销（分布式训练时 expert 分散在不同设备）。pQuant 的设计是把 MoE 思想应用到量化精度的分配上——1-bit 共享专家 + 8-bit 路由专家，是一种创新的组合。
