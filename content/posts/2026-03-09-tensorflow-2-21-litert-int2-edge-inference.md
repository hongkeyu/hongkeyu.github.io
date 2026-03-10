---
title: "Google 正式发布 TensorFlow 2.21：LiteRT 取代 TFLite，INT2 量化落地边缘推理"
date: 2026-03-09T07:30:00+08:00
tags: ["quantization", "edge-inference", "tensorflow"]
description: "Google 在 TensorFlow 2.21 中将 LiteRT 正式取代 TFLite，带来 INT2/INT4 算子支持、1.4x GPU 加速和 PyTorch/JAX 原生转换。"
showToc: true
---

TL;DR: Google 在 TensorFlow 2.21 中将 LiteRT 从 preview 升级为正式的 on-device 推理框架，全面取代 TFLite。核心亮点是 INT2/INT4 算子支持、1.4x GPU 加速、NPU 统一工作流，以及对 PyTorch/JAX 模型的原生转换支持。

---

## 背景

TFLite 一直是 Google 在移动端和边缘设备上的推理框架，但它有几个长期痛点：算子覆盖不全、量化精度选择有限、与非 TensorFlow 训练框架的互操作性差。如果你用 PyTorch 训练了一个模型想部署到手机或 Jetson 上，中间要经过 ONNX 转换或手动重写，链路很脆弱。

LiteRT 从 2025 年下半年开始作为 preview 出现，本次 TF 2.21 标志着它正式"毕业"，成为 Google 官方推荐的唯一 on-device 推理栈。

## 核心更新

### 1. GPU 与 NPU 加速

LiteRT 的 GPU delegate 相比旧版 TFLite 提升了 1.4x 推理速度。更重要的是，NPU（Neural Processing Unit）加速现在有了统一的工作流——以前在不同芯片（高通 Hexagon、联发科 APU、三星 Exynos）上跑 NPU 需要各自的 delegate 和适配逻辑，现在 LiteRT 把这些抽象成了统一接口。这对跨平台 GenAI 部署（比如 Gemma 模型跑在不同手机上）意义很大。

### 2. 极低精度量化：INT2 和 INT4

这是最值得关注的部分。tf.lite 的算子集新增了对极低精度数据类型的支持：

- tfl.fully_connected 支持 INT2（2-bit 权重）
- tfl.cast 支持 INT2 和 INT4 之间的转换
- tfl.slice 支持 INT4
- SQRT 算子支持 int8 和 int16x8
- 比较算子支持 int16x8

INT2 的 fully_connected 支持意味着你可以把模型的线性层权重压缩到 2-bit，这在边缘设备的内存约束下非常关键。一个 7B 参数的模型在 FP16 下需要约 14GB，INT4 大约 3.5GB，INT2 则可以压到约 1.75GB——这已经可以塞进很多手机和嵌入式设备的内存了。

当然，INT2 的精度损失是实打实的，不是所有层都适合压到这个程度。实践中更可能的做法是 mixed-precision：attention 层保持 INT4 或 INT8，FFN 层压到 INT2，通过逐层敏感度分析（比如用 Hessian-based 方法）来决定每层的量化位宽。

### 3. PyTorch / JAX 原生转换

这可能是对日常工作流影响最大的改动。以前从 PyTorch 到 TFLite 的路径是 PyTorch -> ONNX -> TF SavedModel -> TFLite，每一步都可能出问题（动态 shape 不支持、自定义算子丢失等）。现在 LiteRT 提供了 first-class 的 PyTorch 和 JAX 转换支持，可以直接从这两个框架的模型格式转为 LiteRT 格式，不需要先转成 TensorFlow。

这对用 PyTorch 做研究、用边缘设备做部署的人来说是真正的 quality-of-life 提升。

## 对 Jetson 生态的意义

虽然 LiteRT 主要面向移动端和 IoT，但它的 NPU 统一抽象和 INT2/INT4 算子支持对 Jetson Orin 这类平台也有参考价值。Jetson 上主流的推理栈还是 TensorRT，但 LiteRT 的跨框架转换能力可能会成为一个轻量级替代方案，特别是对于不需要极致性能、但需要快速从 PyTorch 原型到设备部署的场景。

另外，INT2 量化在 GGUF 生态中对应的是 Q2_K 格式。llama.cpp 从去年开始就支持 Q2_K，但在 Google 官方框架中看到 INT2 算子级别的支持，说明极低精度量化正在从社区实验走向工业级标准化。

## 延伸

值得关注的趋势是：量化的粒度正在从"模型级别"走向"算子级别"。以前我们说一个模型是 INT4 量化的，指的是整个模型统一量化到 4-bit。现在的方向是每个算子、每一层甚至每个 tensor 都可以有不同的精度，框架层面需要支持这种 mixed-precision 的灵活性。LiteRT 这次对不同算子分别添加不同精度支持，正是这个方向的体现。

---

**参考链接**

- [原文](https://www.marktechpost.com/2026/03/06/google-launches-tensorflow-2-21-and-litert-faster-gpu-performance-new-npu-acceleration-and-seamless-pytorch-edge-deployment-upgrades/)
- [Google 官方博客](https://developers.googleblog.com/whats-new-in-tensorflow-221/)
- [GitHub Release](https://github.com/tensorflow/tensorflow/blob/r2.21/RELEASE.md)
