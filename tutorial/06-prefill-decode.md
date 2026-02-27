# 2.3 Prefill与Decode：两个阶段的分工

> 📍 **问题**：为什么推理要分成Prefill和Decode两个阶段？它们有什么不同？

---

## 2.3.1 两个阶段的本质差异

### 💡 概念解释

Prefill和Decode是推理的两个**计算模式完全不同**的阶段：

| 阶段 | 输入 | 输出 | 计算特点 |
|------|------|------|----------|
| **Prefill** | 所有输入tokens | KV Cache构建完成 | 可并行，计算密集 |
| **Decode** | 单个新token + Cache | 下一个token | 自回归，内存带宽受限 |

### 🔍 深度思考

**问题**：为什么计算特点不同？

**答案**：
- **Prefill**：一次性处理N个tokens，注意力矩阵是N×N，可以高度并行
- **Decode**：每次只处理1个token，Query长度=1，但需要读取全部历史KV

---

## 2.3.2 Prefill阶段详解

### 📝 问题

**问题**：Prefill阶段做了什么？

**答案**：

```
输入: "What is AI?"
Tokens: [W, h, a, t,  , i, s,  , A, I, ?]

Prefill阶段:
┌─────────────────────────────────────────────────────────────┐
│ 一次性处理所有11个tokens                                     │
│                                                             │
│ Attention计算:                                              │
│  - Q[1]:What 关注 [What]                                   │
│  - Q[2]:is   关注 [What, is]                               │
│  - Q[3]:AI   关注 [What, is, AI]                          │
│  - ...                                                      │
│  - Q[11]:?   关注 [What, is, AI, ?]                        │
│                                                             │
│ 输出: 每个token的KV值 → 存入Cache                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 2.3.3 Decode阶段详解

### 📝 问题

**问题**：Decode阶段如何工作？

**答案**：

```
Prefill完成后，开始Decode...

Step 1:
┌─────────────────────────────────────────────────────────────┐
│ 输入: 最后一个token "?"的表示                                │
│ Cache: [What, is, AI, ?]的KV                               │
│ 计算: Q × Cache → 预测下一个token → "is"                   │
│ 更新: Cache添加"is"的KV                                     │
└─────────────────────────────────────────────────────────────┘

Step 2:
┌─────────────────────────────────────────────────────────────┐
│ 输入: "is"的表示                                           │
│ Cache: [What, is, AI, ?, is]的KV                          │
│ 计算: Q × Cache → 预测下一个token → "a"                     │
│ 更新: Cache添加"a"的KV                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 2.3.4 为什么需要分开处理？

### 💡 概念解释

Prefill和Decode分开处理的主要原因：

1. **优化策略不同**
2. **调度策略不同**
3. **内存管理不同**

### 📝 详细说明

| 方面 | Prefill | Decode |
|------|---------|--------|
| **计算密度** | 高（并行计算N个token） | 低（计算单个token） |
| **瓶颈** | 计算能力（GPU算力） | 内存带宽 |
| **优化** | 增大批量、提高算力利用率 | 减少内存访问、优化Cache |
| **延迟** | 较长但只发生一次 | 短但重复多次 |

---

## 2.3.5 Prefill-Decode (PD) 分离的优化

### 🔍 深度思考

**问题**：有什么进一步的优化方式？

**答案**：
**PD分离架构**：
- Prefill和Decode在不同GPU上处理
- Prefill GPU：处理输入，大批量
- Decode GPU：处理生成，小批量
- 适合超大模型

---

## 2.3.6 SimpleLLM中的实现

### 📝 代码位置

`llm.py` - `_prefill` 和 `_decode_step` 方法

### 🔧 代码解析

```python
# 位置：llm.py - _prefill方法

def _prefill(self, input_ids, slot_ids):
    """
    Prefill阶段处理
    
    特点：
    - 一次处理所有输入tokens
    - 构建完整的KV Cache
    - 使用Flash Attention优化
    """
    # 1. 编码输入
    hidden_states = self.model(input_ids)
    
    # 2. 提取K和V
    key_states = self.model.get_key(hidden_states)
    value_states = self.model.get_value(hidden_states)
    
    # 3. 写入KV Cache
    for i, slot_id in enumerate(slot_ids):
        seq_len = input_ids.shape[1]
        self.kv_cache[slot_id, :, :seq_len] = torch.stack([key_states[i], value_states[i]])
        self.cache_seqlens[slot_id] = seq_len
    
    return hidden_states
```

```python
# 位置：llm.py - _decode_step方法

def _decode_step(self, slot_ids):
    """
    Decode阶段处理
    
    特点：
    - 每次只处理1个新token
    - 读取已有Cache
    - 添加新token的KV到Cache
    """
    # 1. 读取当前Cache长度
    current_lens = self.cache_seqlens[slot_ids]
    
    # 2. 计算新token的Q、K、V
    query = self.model.get_query(self.current_hidden)
    key = self.model.get_key(self.current_hidden)
    value = self.model.get_value(self.current_hidden)
    
    # 3. 写入Cache（新位置）
    new_positions = current_lens
    self.kv_cache[slot_ids, 0, new_positions] = key
    self.kv_cache[slot_ids, 1, new_positions] = value
    
    # 4. 更新Cache长度
    self.cache_seqlens[slot_ids] = current_lens + 1
    
    # 5. 注意力计算（使用完整Cache）
    # Q: [batch, 1, num_heads, head_dim]
    # K, V: [batch, seq_len, num_kv_heads, head_dim]
    output = self.attention(query, self.kv_cache[slot_ids])
    
    return output
```

---

## 2.3.7 本节小结

### ✅ 关键要点

1. **Prefill**：一次性处理所有输入tokens，构建KV Cache
2. **Decode**：自回归生成，每次处理1个新token
3. **计算模式不同**：Prefill计算密集，Decode内存带宽受限
4. **分开优化**：两个阶段需要不同的优化策略

---

## 下节预告

> [3.1 连续批处理（Continuous Batching）](./07-batching.md) 将解释现代推理引擎如何高效处理多个并发请求。
