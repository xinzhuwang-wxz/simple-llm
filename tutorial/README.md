# 现代大模型推理与基础设施入门教程

> **基于 SimpleLLM | 从理论到源码 | 剖析极简推理引擎**

欢迎来到 SimpleLLM 推理入门教程！如果你曾好奇 ChatGPT 或 Claude 背后的大模型是如何在 GPU 上快速吐出文字的，或者你在阅读 vLLM 等工业级推理引擎源码时感到无从下手，那么这套教程正是为你准备的。

## 为什么要写这套教程？

目前市面上的大多数教程要么只停留在宏观理论（例如“什么是 KV Cache”），要么一上来就让你看动辄十万行的工业级 C++ 代码（如 vLLM 或 TensorRT-LLM）。这在“懂理论”和“能写代码”之间留下了一道巨大的鸿沟。

**本教程试图填补这道鸿沟。** 

我们将以 **[SimpleLLM](../README.md)**（一个仅有 ~950 行代码、却包含了连续批处理、CUDA 图加速、Flash Attention 等现代核心技术的极简推理引擎）为解剖对象，带你从最基础的理论出发，一步步深入到每一行关键代码，最终理解高性能推理的本质。

---

## 📖 教程目录 (Chapters)

我们建议你按照顺序阅读，这 5 个章节将带你完成一次从宏观到微观、从理论到实践的完整旅程：

### [Chapter 1: 从零理解 LLM 推理与性能瓶颈](./01-Foundations.md)
*   **什么是自回归生成 (Autoregressive Generation)**
*   **拆解生命周期：Prefill (预填充) 与 Decode (解码)**
*   **推理的第一大痛点：为什么我们需要 KV Cache？**
*   *适合人群：完全没有大模型推理概念的新手。*

### [Chapter 2: 突破显存与吞吐极限：现代推理核心技术](./02-HighPerformanceCoreTech.md)
*   **告别木桶效应：连续批处理 (Continuous / In-flight Batching)**
*   **显存管理革命：Paged Attention 是如何借鉴操作系统内存分页的？**
*   **CPU 开销克星：理解 CUDA Graphs**
*   *适合人群：知道基础概念，但想深入理解现代推理引擎必备优化手段的开发者。*

### [Chapter 3: SimpleLLM 引擎架构与源码深度走读](./03-EngineArchitecture.md)
*   **揭秘 `llm.py`：异步请求队列与后台推理循环**
*   **手撕源码：连续批处理的优雅实现机制**
*   **在代码中观察 Prefill 与 Decode 的分流执行**
*   *适合人群：想看懂实际代码，了解如何用 Python 写出一个高并发调度的 AI 工程师。*

### [Chapter 4: 极致的算子优化：Model 与 Kernel 融合](./04-ModelAndKernelFusion.md)
*   **深入 `model/model.py`**
*   **Flash Attention 2 的接入与实战**
*   **算子融合 (Kernel Fusion) 的魔法：为什么要把操作合在一起？**
*   **(进阶) 浅窥 Triton 算子与 MoE 机制**
*   *适合人群：对底层模型计算、显存带宽瓶颈 (Memory-bound) 优化感兴趣的底层爱好者。*

### [Chapter 5: 实战与模拟器：动手验证你的理解](./05-SimulatorAndPractice.md)
*   **环境搭建与云 GPU (RunPod) 租用指南**
*   **一行代码跑通 120B 大模型**
*   **使用 Mac 模拟器：直观可视化观察 Batching 与 KV Cache 分配过程**
*   *适合人群：所有想要动手跑一跑代码的人。*

---

## 🎯 学习目标与路线

读完本教程后，你将能够：

1.  **脱口而出**：解释清楚 KV Cache、Paged Attention、Continuous Batching、Prefill/Decode 等所有核心面试考点。
2.  **读懂源码**：不仅能看懂 SimpleLLM 这 ~950 行代码，还能以此为跳板，去阅读 vLLM 等大型开源项目的源码。
3.  **定位瓶颈**：明白推理过程中的 Memory-bound（显存带宽瓶颈）和 Compute-bound（计算瓶颈）分别发生在哪里，以及业界是如何优化它们的。

### 学习建议
*   **先看图，后看代码**：教程中包含大量的时序图和架构图，请务必先理解图形化的流程，再去看对应的源码。
*   **动手运行**：理论是很枯燥的。强烈建议你在阅读完前两章后，直接跳到 [Chapter 5](./05-SimulatorAndPractice.md) 运行一下模拟器，通过可视化的日志验证你的猜想。

---

准备好了吗？让我们进入 [**Chapter 1: 从零理解 LLM 推理与性能瓶颈**](./01-Foundations.md)。