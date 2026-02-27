# 2.1 自回归生成机制

> 📍 **问题**：Transformer模型是如何逐个生成token的？为什么不能一次生成整个句子？

---

## 2.1.1 从Transformer到自回归生成

### 💡 概念解释

**自回归生成**是大型语言模型的核心工作方式：每次预测下一个token，然后把这个新token加入输入，继续预测下一个。

### 🔍 深度思考

**问题**：为什么不能一次生成整个句子？

**启发**：
- 假设要生成"Artificial intelligence is great"
- 模型在生成第1个token"Artificial"时，并不知道后面的内容
- 必须依次生成，每个token都依赖于之前的tokens

**答案**：
自回归的**本质原因**：
1. 模型在训练时就是用"前i-1个token预测第i个token"的方式
2. 推理时必须复现这个过程
3. 第t个token的生成依赖于t-1个已经生成的tokens

---

## 2.1.2 生成过程详解

### 📝 问题

**问题**：具体的生成过程是怎样的？

**答案**：

```
输入: "What is"
Step 1: 预测 → "AI"
输入: "What is AI"
Step 2: 预测 → "?"
输入: "What is AI?"
Step 3: 预测 → [EOS] (结束符)
生成完成: "What is AI?"
```

### 📝 图示

```
时间 →
─────────────────────────────────────────────────────────────────────────→

Step 1: [BOS] What is          → [BOS] What is AI
Step 2: [BOS] What is AI       → [BOS] What is AI ?
Step 3: [BOS] What is AI ?     → [BOS] What is AI ? [EOS]
                                    ↑
                              每步都把新token加入输入
```

---

## 2.1.3 数学原理

### 💡 概念解释

从数学角度，自回归生成可以表示为：

$$P(x_{1:T}) = \prod_{t=1}^{T} P(x_t | x_{1:t-1})$$

### 📝 解释

| 符号 | 含义 |
|------|------|
| $x_{1:T}$ | 生成的完整序列 $x_1, x_2, ..., x_T$ |
| $x_t$ | 第t个token |
| $x_{1:t-1}$ | 前面所有tokens |

### 🔍 深度思考

**问题**：这个公式说明了什么？

**启发**：
- 整个序列的概率 = 每个位置条件概率的乘积
- 每个位置的概率取决于前面的所有tokens

**答案**：
这就是为什么必须**逐个生成**的原因：
- 无法同时计算所有位置的概率
- 必须按顺序，从左到右生成

---

## 2.1.4 SimpleLLM中的实现

### 📝 代码位置

`llm.py` - 核心生成逻辑

### 🔧 代码解析

```python
# 简化版的自回归生成逻辑
# 位置：llm.py - _decode_step (简化版)

def _decode_step(self, input_ids, max_tokens):
    """自回归Decode步骤"""
    
    # 循环生成，直到达到最大token数或遇到结束符
    for step in range(max_tokens):
        # 1. 前向传播，获取logits
        # input_ids包含当前所有已生成的tokens
        logits = self.model(input_ids)
        
        # 2. 只取最后一个token的logits来预测下一个
        next_token_logits = logits[:, -1, :]
        
        # 3. 采样下一个token
        next_token = self._sample_tokens(next_token_logits)
        
        # 4. 加入已生成的序列
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # 5. 检查是否结束
        if next_token == self.eos_token_id:
            break
    
    return input_ids
```

---

## 2.1.5 两种生成策略

### 💡 概念解释

自回归生成有两种主要策略：

| 策略 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **Greedy** | 始终选择概率最高的token | 简单 | 可能陷入重复 |
| **Sampling** | 根据概率分布随机采样 | 更多样性 | 不确定性 |

### 📝 Temperature采样

```python
# Temperature控制分布的平滑程度
def sample_with_temperature(logits, temperature=0.7):
    # 1. 调整logits（除以temperature）
    adjusted_logits = logits / temperature
    
    # 2. 转为概率分布
    probs = F.softmax(adjusted_logits, dim=-1)
    
    # 3. 采样
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token
```

### 📝 Top-K采样

```python
def sample_top_k(logits, k=50):
    # 1. 只保留概率最高的k个token
    top_k_logits, top_k_indices = torch.topk(logits, k)
    
    # 2. 其余设为-∞（概率为0）
    masked_logits = torch.full_like(logits, float('-inf'))
    masked_logits.scatter_(1, top_k_indices, top_k_logits)
    
    # 3. 从k个中选择
    probs = F.softmax(masked_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## 2.1.6 生成效率问题

### 📝 问题

**问题**：自回归生成有什么效率问题？

**答案**：

| 问题 | 描述 | 影响 |
|------|------|------|
| **顺序计算** | 必须一个个生成，无法并行 | 延迟高 |
| **重复计算** | 每步都处理所有历史tokens | 计算浪费 |
| **内存访问** | 每步需要读取完整KV Cache | 带宽受限 |

### 🔍 深度思考

**问题**：这些问题的解决方案是什么？

**答案**：
1. **KV Cache**：缓存已计算的Key和Value，避免重复计算
2. **批处理**：同时处理多个请求
3. **CUDA Graph**：减少内核启动开销

这些正是后续章节要详细讲解的内容！

---

## 2.1.7 本节小结

### ✅ 关键要点

1. **自回归生成**：每次预测下一个token，然后把新token加入输入
2. **数学原理**：$P(x_{1:T}) = \prod_{t=1}^{T} P(x_t | x_{1:t-1})$
3. **生成策略**：Greedy vs Sampling，Temperature，Top-K
4. **效率问题**：顺序计算、重复计算、内存访问

---

### 📝 练习题

1. **计算**：如果生成100个token，每个token平均有10000个候选，需要多少次计算？
2. **思考**：为什么自回归是必须的，而不是并行的？

---

## 下节预告

> [2.2 KV Cache：推理效率的关键](./05-kv-cache.md) 将深入解释KV Cache如何解决自回归的效率问题。
