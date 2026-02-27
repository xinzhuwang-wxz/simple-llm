# 2.2 KV Cache：推理效率的关键

> 📍 **问题**：在自回归生成中，每一步都需要访问之前所有token的信息。如果不保存这些信息，会发生什么？

---

## 2.2.1 没有KV Cache的问题

### 📝 问题

**问题**：假设没有KV Cache，生成100个token需要多少计算？

**启发**：
- 注意力机制的计算复杂度是 $O(n^2)$，其中n是序列长度
- 第1步处理1个token
- 第2步处理2个tokens
- 第100步处理100个tokens

**答案**：
总计算量 = $1^2 + 2^2 + 3^2 + ... + 100^2 = 338,350$

### 📝 数量级对比

| 序列长度 | 注意力计算次数 | 相对复杂度 |
|----------|----------------|------------|
| 4 | 16 | 1x |
| 100 | 10,000 | 625x |
| 1000 | 1,000,000 | 62,500x |

---

## 2.2.2 KV Cache的核心思想

### 💡 概念解释

**KV Cache**的核心思想：**空间换时间**
- 用额外内存存储已经计算过的Key和Value
- 避免重复计算

### 📝 图示

```
没有KV Cache:
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Q[1] × K[1],V[1]                                        │
│ Step 2: Q[2] × (K[1],K[2]), (V[1],V[2])  ← 重新计算K[1],V[1]  │
│ Step 3: Q[3] × (K[1],K[2],K[3]), ...   ← 重新计算K[1],K[2]    │
│ ...                                                              │
└─────────────────────────────────────────────────────────────────┘

有KV Cache:
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: 计算K[1],V[1]，存入Cache                                │
│ Step 2: 读取Cache + 计算K[2],V[2] → 写入Cache                  │
│ Step 3: 读取Cache + 计算K[3],V[3] → 写入Cache                  │
│ ...                                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2.2.3 KV Cache的数学原理

### 💡 概念解释

标准注意力机制：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 🔍 深度思考

**问题**：attention计算中，哪些部分可以缓存？

**答案**：
- **K和V**：来自之前所有tokens，可以复用
- **Q**：只来自当前token，必须新计算

### 📝 分步计算

```
Prefill阶段（处理输入）:
- 输入: [What, is, AI]
- 输出: 
  * K[1],V[1] for "What"
  * K[2],V[2] for "is"  
  * K[3],V[3] for "AI"
- Cache包含: K[1,2,3], V[1,2,3]

Decode阶段（生成新token）:
- 第4步:
  * Q[4] for new token (新计算)
  * 读取 K[1,2,3], V[1,2,3] from Cache
  * 计算 attention(Q[4], K[1-4], V[1-4])
```

---

## 2.2.4 SimpleLLM中的KV Cache实现

### 📝 代码位置

`llm.py` - KV Cache管理

### 🔧 代码解析

```python
# 位置：llm.py - 初始化KV Cache

def __init__(self, model_path, max_num_seqs=10, max_seq_len=1024):
    # KV Cache形状: [max_num_seqs, 2, max_seq_len, num_kv_heads, head_dim]
    # 2表示K和V两个矩阵
    self._kv_cache = torch.zeros(
        (max_num_seqs, 2, max_seq_len, self.num_kv_heads, self.head_dim),
        device=self.device,
        dtype=self.dtype
    )
    
    # 跟踪每个序列的当前长度
    self._cache_seqlens = torch.zeros(max_num_seqs, dtype=torch.int32)
```

---

### 📝 Prefill阶段Cache写入

```python
# 位置：llm.py - _prefill方法

def _prefill(self, input_ids, slot_ids):
    """Prefill阶段：处理输入，构建KV Cache"""
    
    # 前向传播，得到隐藏状态
    hidden_states = self.model(input_ids)
    
    # 提取K和V
    # 形状: [batch, seq_len, num_kv_heads, head_dim]
    key_states, value_states = hidden_states_to_kv(hidden_states)
    
    # 写入KV Cache
    slot_ids是每个序列对应的槽位ID
    for i, slot_id in enumerate(slot_ids):
        self._kv_cache[slot_id, 0, :len, :, :] = key_states[i]
        self._kv_cache[slot_id, 1, :len, :, :] = value_states[i]
        self._cache_seqlens[slot_id] = len
```

---

### 📝 Decode阶段Cache读取

```python
# 位置：llm.py - _decode_step方法

def _decode_step(self, slot_ids):
    """Decode阶段：读取Cache，生成新token"""
    
    # 1. 读取已有的KV Cache
    # 形状: [batch, seqlen, num_kv_heads, head_dim]
    cached_keys = self._kv_cache[slot_ids, 0, :self._cache_seqlens[slot_ids]]
    cached_values = self._kv_cache[slot_ids, 1, :self._cache_seqlens[slot_ids]]
    
    # 2. 计算当前token的Q、K、V
    # 只处理最后一个token
    query = self.model.get_query(last_hidden_states)
    key = self.model.get_key(last_hidden_states)
    value = self.model.get_value(last_hidden_states)
    
    # 3. 写入Cache（为下一步准备）
    new_pos = self._cache_seqlens[slot_ids]
    self._kv_cache[slot_ids, 0, new_pos] = key
    self._kv_cache[slot_ids, 1, new_pos] = value
    self._cache_seqlens[slot_ids] += 1
    
    # 4. 计算注意力（使用Cache中的历史KVs）
    attn_output = attention(query, cached_keys + key, cached_values + value)
```

---

## 2.2.5 KV Cache的内存占用

### 📝 问题

**问题**：KV Cache需要多少内存？

**答案**：

| 模型 | 序列长度 | KV Cache大小 |
|------|----------|--------------|
| LLaMA-7B | 4096 | ~384 MB |
| LLaMA-70B | 4096 | ~3.8 GB |
| GPT-OSS-120B (本项目) | 1024 | ~2 GB |

### 🔍 深度思考

**问题**：如何减少KV Cache的内存占用？

**答案**：
1. **量化**：用FP16/BF16 → INT8/INT4
2. **Paged Attention**：按需分配，避免预分配
3. **窗口注意力**：只保留最近N个token的KV

---

## 2.2.6 本节小结

### ✅ 关键要点

1. **问题**：没有KV Cache会导致$O(n^2)$重复计算
2. **解决**：用空间换时间，缓存K和V
3. **效果**：Decode步骤只需要$O(n)$计算
4. **实现**：Prefill写入，Decode读写

---

## 下节预告

> [2.3 Prefill与Decode：两个阶段的分工](./06-prefill-decode.md) 将解释为什么推理分成这两个阶段，以及它们如何协同工作。
