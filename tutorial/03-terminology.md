# 1.3 核心术语体系

> 📍 **问题**：在AI推理领域，我们经常听到各种专业术语，如KV Cache、Continuous Batching、Triton等。这些术语到底是什么意思？

---

## 1.3.1 推理流程核心术语

### 1.3.1.1 Tokenizer（分词器）

### 💡 概念解释

**Tokenizer**是将文本转换为模型可以处理的数字序列（tokens）的组件。

**类比**：
- 文本 → Token：就像把一句话分解成单词
- 模型只能处理数字，所以需要这种"翻译"过程

### 🔍 深度思考

**问题**：为什么不能直接处理字符？

**启发**：
- 英文字符集只有128个，表达能力有限
- 使用Token可以：
  - 表示常用词组（如"machine learning"作为一个token）
  - 处理不同语言
  - 压缩信息表示

**答案**：
使用Token的优势：
- **表示效率高**：常用词用一个token表示，而不是多个字符
- **语义完整**：每个token都有一定语义意义
- **vocab_size通常为30k-128k**：足够覆盖各种表达

### 📝 代码示例（SimpleLLM）

```python
# 位置：model/tokenizer.py

# Tokenizer将文本转换为token IDs
tokens = tokenizer.encode("What is AI?")
print(tokens)  # [1234, 567, 890, 123]

# 解码
text = tokenizer.decode(tokens)
print(text)  # "What is AI?"
```

---

### 1.3.1.2 Prefill（预填充）

### 💡 概念解释

**Prefill**是推理的第一阶段：处理输入的所有tokens，构建初始的KV Cache。

### 🔍 深度思考

**问题**：为什么叫"Prefill"？

**启发**：
- "Pre" = 预先
- "fill" = 填充
- 预先填充KV Cache，为后续的Decode阶段做准备

**答案**：
在Prefill阶段：
1. 输入："What is AI?" → 4个tokens
2. 模型一次性计算这4个tokens的注意力
3. 将计算得到的Key和Value存入KV Cache
4. 准备完成，开始Decode

### 📝 图示

```
输入: "What is AI?"

Prefill阶段:
┌────────────────────────────────────────────────────────────┐
│ Token 1: What                                             │
│ Token 2: is                                              │
│ Token 3: AI                                             │
│ Token 4: ?                                               │
└────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│ KV Cache构建完成                                           │
│ K[1], V[1] → Token 1的Key-Value                           │
│ K[2], V[2] → Token 2的Key-Value                           │
│ K[3], V[3] → Token 3的Key-Value                           │
│ K[4], V[4] → Token 4的Key-Value                           │
└────────────────────────────────────────────────────────────┘
```

---

### 1.3.1.3 Decode（解码）

### 💡 概念解释

**Decode**是推理的第二阶段：自回归地生成下一个token。

### 🔍 深度思考

**问题**：为什么是"自回归"？

**启发**：
- 每个新token的生成依赖于之前生成的所有tokens
- 就像写作文：一个字一个字地写，每个新字都要考虑前面的内容

**答案**：
自回归生成过程：
```
Step 1: 输入 "What is AI?" → 预测下一个token "is"
Step 2: 输入 "What is AI? is" → 预测下一个token "a"
Step 3: 输入 "What is AI? is a" → 预测下一个token "field"
... 继续直到结束
```

---

### 1.3.1.4 Sampling（采样）

### 💡 概念解释

**Sampling**是从模型输出的概率分布中选择下一个token的策略。

### 📝 常见采样策略

| 策略 | 描述 | 特点 |
|------|------|------|
| **Greedy** | 总是选择概率最高的token | 确定性强，可能重复 |
| **Temperature** | 通过调整温度参数控制分布平滑度 | 温度高→更随机 |
| **Top-K** | 只从概率最高的K个token中采样 | 平衡多样性和质量 |
| **Top-P (Nucleus)** | 从累积概率达到P的最小集合中采样 | 自适应候选数量 |

### 📝 代码示例（SimpleLLM）

```python
# 位置：llm.py - _sample_tokens方法

# 简化示例：Greedy Sampling
def greedy_sample(logits):
    return torch.argmax(logits, dim=-1)

# Temperature Sampling
def temperature_sample(logits, temperature=0.7):
    # 调整分布
    logits = logits / temperature
    # 转换为概率
    probs = F.softmax(logits, dim=-1)
    # 采样
    return torch.multinomial(probs, num_samples=1)
```

---

## 1.3.2 内存管理核心术语

### 1.3.2.1 KV Cache

### 💡 概念解释

**KV Cache**是存储注意力机制中Key和Value的缓存，避免重复计算。

### 🔍 深度思考

**问题**：没有KV Cache会怎样？

**启发**：
- 假设生成100个token
- 第50步Decode时，需要计算第1-50个token之间的注意力
- 如果每次都重新计算所有Key和Value，浪费巨大

**答案**：
KV Cache的作用：
- 存储已经计算过的Key和Value
- Decode时只需计算当前token的Query
- 从Cache读取历史的Key和Value
- 大幅减少计算量

### 📝 图示

```
没有KV Cache:
┌─────────────────────────────────────────────────────────────┐
│ Decode Step 50:                                            │
│ - 重新计算 K[1-50], V[1-50]  ← 重复计算！                    │
│ - 然后计算注意力                                           │
└─────────────────────────────────────────────────────────────┘

有KV Cache:
┌─────────────────────────────────────────────────────────────┐
│ Decode Step 50:                                            │
│ - 读取 K[1-49], V[1-49]  ← 从Cache读取                     │
│ - 只计算 K[50], V[50]     ← 新计算                          │
│ - 然后计算注意力                                           │
└─────────────────────────────────────────────────────────────┘
```

---

### 1.3.2.2 Paged Attention（分页注意力）

### 💡 概念解释

**Paged Attention**是一种内存管理技术，将KV Cache分成固定大小的"页面"进行管理。

### 📝 类比

| 传统方式 | Paged Attention |
|----------|-----------------|
| 连续的内存块 | 分页管理 |
| 需要预分配整个序列长度 | 按需分配页面 |
| 内存碎片 | 无碎片 |

### 🔍 深度思考

**问题**：Paged Attention解决了什么问题？

**启发**：
- 传统方式需要预分配最大长度（如4096）的连续内存
- 但实际请求可能只需要100 tokens
- 大量内存被浪费

**答案**：
Paged Attention的优势：
- 按需分配内存页
- 序列可以动态增长
- 完成序列的页面立即释放
- 内存利用率大幅提升

---

### 1.3.2.3 Slot-based KV Cache（槽位式KV Cache）

### 💡 概念解释

**Slot-based KV Cache**是SimpleLLM使用的内存管理方式：将GPU内存预分成固定数量的槽位。

### 📝 SimpleLLM实现

```python
# 位置：llm.py - 初始化阶段

# 预分配槽位
self.max_num_seqs = 10  # 最大并发序列数
self.max_seq_len = 1024  # 每个序列最大长度

# KV Cache形状: [num_slots, seq_len, num_heads, head_dim]
# 这是一个固定大小的连续内存块
```

---

## 1.3.3 批处理核心术语

### 1.3.3.1 Static Batching（静态批处理）

### 💡 概念解释

**Static Batching**是传统的批处理方式：等待所有请求到齐后一起处理。

### 📝 图示

```
时间 →
─────────────────────────────────────────────────────────────────→

请求A: [=====处理=====] ████████████
请求B:      [=====等待=====] ████████████
请求C:           [=====等待=====] ████████████
                ↑
           必须等待最长的请求完成

问题: 请求B和C需要等待A完成才能返回
```

---

### 1.3.3.2 Continuous Batching（连续批处理）

### 💡 概念解释

**Continuous Batching**允许新请求随时加入正在执行的批次。

### 📝 图示

```
时间 →
─────────────────────────────────────────────────────────────────→

批次1: [A][B][C][D][E] → [A][B][C][D] → [B][C][D] → ...
          ↑  新请求加入      ↑ A完成      ↑ B完成

优势: 
- A完成后立即返回
- B、C、D继续处理
- 新请求可以立即加入
```

### 🔍 深度思考

**问题**：连续批处理如何判断请求是否完成？

**启发**：
- 每个Decode步骤后检查是否生成结束符（EOS）
- 或者检查是否达到最大token数

**答案**：
SimpleLLM的实现：
```python
# 位置：llm.py - _inference_loop

# 检查每个序列是否完成
for seq_idx in range(batch_size):
    if finished[seq_idx]:
        # 完成，移除该序列
        # 槽位立即释放给新请求使用
```

---

## 1.3.4 性能优化核心术语

### 1.3.4.1 CUDA Graph

### 💡 概念解释

**CUDA Graph**是NVIDIA的技术，通过预捕获计算图来减少内核启动开销。

### 🔍 深度思考

**问题**：内核启动有什么开销？

**启发**：
- 每次GPU操作都需要CPU提交到GPU
- 这个提交过程有开销
- 如果每次Decode有100个小操作，累积开销很大

**答案**：
CUDA Graph的工作方式：
1. **捕获**：运行一次完整的Decode，记录所有操作
2. **重放**：后续Decode直接重放这个图
3. **优势**：消除内核启动开销，显著提升性能

### 📝 SimpleLLM实现

```python
# 位置：llm.py - _capture_cuda_graph

# 捕获CUDA Graph
self.graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(self.graph):
    # 这里是完整的Decode步骤
    # 第一次运行会执行实际计算
    self._decode_step_captured(...)

# 后续直接重放
self.graph.replay()  # 比普通调用快
```

---

### 1.3.4.2 Kernel Fusion（内核融合）

### 💡 概念解释

**Kernel Fusion**是将多个GPU操作合并为一个内核的技术。

### 📝 示例

| 融合前 | 融合后 |
|--------|--------|
| MatMul → ReLU → MatMul | 一个融合内核 |
| RMSNorm → 残差相加 | 一个融合内核 |
| RoPE位置编码 | 融合到注意力计算中 |

### 🔍 深度思考

**问题**：融合为什么能提升性能？

**启发**：
- 每次GPU操作都需要内存读写
- 融合减少内存访问次数
- 同时避免CPU-GPU通信开销

**答案**：
融合的优势：
- 减少内存带宽使用
- 消除内核启动开销
- 提高计算密度

---

### 1.3.4.3 Triton

### 💡 概念解释

**Triton**是一种用于编写高效GPU内核的编程语言/编译器。

### 📝 为什么需要Triton？

| 方案 | 优点 | 缺点 |
|------|------|------|
| CUDA C++ | 完全控制 | 编写困难 |
| PyTorch | 容易 | 不够高效 |
| Triton | 容易+高效 | 相对新 |

### 📝 SimpleLLM中的Triton内核

```python
# 位置：kernels/norm.py

import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    input_ptr, weight_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_cols, eps, BLOCK_SIZE: tl.constexpr
):
    # Triton内核实现
    # 比PyTorch实现更高效
    ...
```

---

## 1.3.5 模型架构核心术语

### 1.3.5.1 MoE (Mixture of Experts)

### 💡 概念解释

**MoE**是一种稀疏激活的模型架构：多个"专家"网络，但只激活部分专家处理每个token。

### 📝 SimpleLLM使用的MoE

```python
# 模型有128个专家
# 每次只激活top-4专家
# 大幅减少计算量，同时保持模型容量
```

### 🔍 深度思考

**问题**：MoE如何做到"稀疏"？

**启发**：
- 128个专家不是都参与计算
- 路由器（Router）决定每个token使用哪些专家
- 只计算激活的4个专家

**答案**：
MoE的优势：
- 推理时只计算部分专家 → 快
- 模型参数很多 → 能力强
- 平衡了效率和性能

---

### 1.3.5.2 GQA (Grouped Query Attention)

### 💡 概念解释

**GQA**是一种注意力机制优化：多个Query头共享一组Key-Value头。

### 📝 对比

| 机制 | Q头数 | KV头数 | 内存 |
|------|-------|--------|------|
| MHA | 64 | 64 | 多 |
| GQA | 64 | 8 | 少 |
| MQA | 64 | 1 | 最少 |

### 🔍 深度思考

**问题**：减少KV头数会影响效果吗？

**启发**：
- KV头数影响表达能力
- 但共享的KV可以服务于多个Q
- 适当平衡即可

**答案**：
SimpleLLM使用GQA：
- 64个Query头
- 8个KV头
- 在减少内存的同时保持较好效果

---

## 1.3.6 完整术语表

| 术语 | 英文 | 章节 | 简述 |
|------|------|------|------|
| Tokenizer | Tokenizer | 1.3.1.1 | 文本→数字转换 |
| Prefill | Prefill | 1.3.1.2 | 处理输入，构建KV Cache |
| Decode | Decode | 1.3.1.3 | 自回归生成token |
| Sampling | Sampling | 1.3.1.4 | 从概率分布选token |
| KV Cache | KV Cache | 1.3.2.1 | 存储Key-Value |
| Paged Attention | Paged Attention | 1.3.2.2 | 分页内存管理 |
| Slot-based Cache | Slot-based Cache | 1.3.2.3 | 固定槽位管理 |
| Static Batching | Static Batching | 1.3.3.1 | 传统批处理 |
| Continuous Batching | Continuous Batching | 1.3.3.2 | 动态批处理 |
| CUDA Graph | CUDA Graph | 1.3.4.1 | 计算图捕获 |
| Kernel Fusion | Kernel Fusion | 1.3.4.2 | 内核融合 |
| Triton | Triton | 1.3.4.3 | GPU内核编程 |
| MoE | Mixture of Experts | 1.3.5.1 | 稀疏专家网络 |
| GQA | Grouped Query Attention | 1.3.5.2 | 分组查询注意力 |
| RoPE | Rotary Position Embedding | - | 旋转位置编码 |
| Flash Attention | Flash Attention | - | 高效注意力 |

---

## 1.3.7 本节小结

### ✅ 关键要点

1. **推理流程术语**：Tokenizer → Prefill → Decode → Sampling
2. **内存管理**：KV Cache是核心，避免重复计算
3. **批处理**：连续批处理优于静态批处理
4. **优化技术**：CUDA Graph、内核融合、Triton
5. **模型架构**：MoE、GQA等优化技术

---

## 下节预告

> [2.1 自回归生成机制](./04-autoregressive.md) 将深入解释Transformer的自回归生成原理。
