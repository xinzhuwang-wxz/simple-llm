# 4.3 模型定义与前向传播

> 📍 **问题**：model/model.py中的模型架构是怎样的？

---

## 4.3.1 模型组件

| 类 | 功能 |
|---|------|
| `GptOssConfig` | 模型配置 |
| `RMSNorm` | 归一化层 |
| `RotaryEmbedding` | 旋转位置编码 |
| `Attention` | 注意力机制 (GQA) |
| `FusedMoE` | 混合专家层 |
| `TransformerBlock` | Transformer层 |
| `GptOssModel` | 完整骨干网络 |
| `GptOssForCausalLM` | 完整语言模型 |

---

## 4.3.2 GQA注意力

```python
# 位置：model/model.py - Attention类

# 64个Query头，8个KV头
# 多个Query头共享同一个KV头
self.num_heads = 64
self.num_kv_heads = 8
```

---

## 4.3.3 MoE层

```python
# 位置：model/model.py - FusedMoE

# 128个专家，每次激活top-4
self.num_experts = 128
self.top_k = 4
```

---

## 4.3.4 本节小结

### ✅ 关键要点

1. **GQA**：64 Q头 / 8 KV头
2. **MoE**：128专家 / 激活4个
3. **完整模型**：GptOssForCausalLM
