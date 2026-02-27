# 现代AI推理与基础设施入门教程

> **基于SimpleLLM项目 | 苏格拉底式学习法 | Mac可运行模拟器**

---

## 目录

### 第一部分：AI推理全景图与核心概念
- [1.1 从训练到推理：为什么需要专门的推理引擎？](./tutorial/01-intro.md)
- [1.2 推理系统全景图](./tutorial/02-overview.md)
- [1.3 核心术语体系](./tutorial/03-terminology.md)

### 第二部分：Transformer推理原理详解
- [2.1 自回归生成机制](./tutorial/04-autoregressive.md)
- [2.2 KV Cache：推理效率的关键](./tutorial/05-kv-cache.md)
- [2.3 Prefill与Decode：两个阶段的分工](./tutorial/06-prefill-decode.md)

### 第三部分：现代推理引擎核心技术
- [3.1 连续批处理（Continuous Batching）](./tutorial/07-batching.md)
- [3.2 Paged Attention与内存管理](./tutorial/08-paged-attention.md)
- [3.3 CUDA Graph优化](./tutorial/09-cuda-graphs.md)
- [3.4 内核融合与Triton](./tutorial/10-kernel-fusion.md)
- [3.5 量化与MoE优化](./tutorial/11-quantization-moe.md)

### 第四部分：代码级理解与SimpleLLM解析
- [4.1 SimpleLLM项目结构](./tutorial/12-project-structure.md)
- [4.2 推理引擎核心代码解析](./tutorial/13-engine-code.md)
- [4.3 模型定义与前向传播](./tutorial/14-model-code.md)
- [4.4 Triton内核深度解析](./tutorial/15-triton-kernels.md)

### 第五部分：GPU环境准备与租卡指南
- [5.1 云GPU服务选择](./tutorial/16-cloud-gpu.md)
- [5.2 RunPod租卡完整指南](./tutorial/17-runpod-setup.md)
- [5.3 环境配置与依赖安装](./tutorial/18-environment.md)
- [5.4 运行SimpleLLM推理](./tutorial/19-running-inference.md)

### 第六部分：Mac模拟器与实践
- [6.1 模拟器使用说明](./tutorial/20-simulator-intro.md)
- [6.2 模拟器1：Tokenization与Attention可视化](./tutorial/21-sim-tokenization.md)
- [6.3 模拟器2：批处理机制模拟](./tutorial/22-sim-batching.md)
- [6.4 模拟器3：KV Cache动态演示](./tutorial/23-sim-kv-cache.md)

### 第七部分：进阶学习路径
- [7.1 扩展学习资源](./tutorial/24-resources.md)
- [7.2 后续实践建议](./tutorial/25-next-steps.md)

---

## 学习路线图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          AI推理与Infra入门学习路线                        │
└─────────────────────────────────────────────────────────────────────────┘

第1周
├── Day 1-2: 认知准备
│   ├── AI推理全景图理解
│   ├── 核心术语体系
│   └── [可选] 运行模拟器1：Tokenization
│
├── Day 3-4: 理论基础  
│   ├── 自回归生成机制
│   ├── KV Cache原理
│   └── Prefill/Decode分工
│
└── Day 5-7: 核心技ポ理解
    ├── 连续批处理
    ├── Paged Attention
    └── [可选] 运行模拟器2：批处理

第2周
├── Day 8-10: 代码级理解
│   ├── SimpleLLM项目结构
│   ├── 推理引擎核心
│   └── Triton内核
│
├── Day 11-12: GPU环境
│   ├── 租卡指南
│   ├── 环境配置
│   └── 运行推理
│
└── Day 13-14: 实践与进阶
    ├── 运行实际推理
    ├── 模拟器3：KV Cache
    └── 扩展学习

总计：约14天
```

---

## 如何使用本教程

### 1. 学习模式
本教程采用**苏格拉底式学习法**，每个概念都通过以下结构讲解：

```
📍 问题 (Question)
   ↓ 启发思考
💡 概念解释 (Concept)
   ↓ 代码验证
🔧 代码实现 (Code)
   ↓ 延伸问题
📝 练习题 (Exercise)
```

### 2. Mac模拟器
教程中的模拟器可以在Mac上运行，帮助你直观理解核心机制：

```bash
# 进入模拟器目录
cd tutorial/simulator

# 运行模拟器
python simulator_1_tokenization.py
python simulator_2_batching.py  
python simulator_3_kv_cache.py
```

### 3. 代码阅读建议
- 核心代码都标注了行号，便于对照仓库源码
- 建议先运行模拟器理解概念，再阅读实际代码
- 遇到不理解的地方，可以画图或添加打印语句调试

---

## 预备知识

本教程假设你具备以下基础知识：

✅ **已掌握**
- 深度学习基础（神经网络、反向传播）
- Python编程能力
- 基本的线性代数和概率论

⏳ **将学会**
- 推理引擎架构设计
- GPU计算优化技术
- 生产级AI系统部署

---

## 反馈与交流

如果你在学习过程中遇到问题：
1. 先尝试运行模拟器，添加打印语句调试
2. 查看SimpleLLM源码，对照教程讲解
3. 记录问题，在社区寻求帮助

---

## 开始学习

➡️ [进入第一部分：AI推理全景图与核心概念](./tutorial/01-intro.md)

---

*教程基于SimpleLLM项目编写，该项目是一个约950行的极简可扩展LLM推理引擎，适合学习现代推理技术。*
