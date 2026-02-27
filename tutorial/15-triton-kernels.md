# 4.4 Triton内核深度解析

> 📍 **问题**：SimpleLLM使用了哪些Triton内核？

---

## 4.4.1 内核列表

| 文件 | 功能 |
|------|------|
| `kernels/norm.py` | RMSNorm |
| `kernels/rope.py` | RoPE位置编码 |
| `kernels/moe.py` | MoE前向传播 |

---

## 4.4.2 RMSNorm内核

```python
# 位置：kernels/norm.py

@triton.jit
def rmsnorm_kernel(
    input_ptr, weight_ptr, output_ptr,
    n_cols, eps, BLOCK_SIZE: tl.constexpr
):
    # RMSNorm计算
    # 比PyTorch实现更高效
```

---

## 4.4.3 RoPE内核

```python
# 位置：kernels/rope.py

@triton.jit
def rope_kernel(
    query_ptr, key_ptr, cos_ptr, sin_ptr,
    positions_ptr, ...
):
    # 融合RoPE位置编码
    # Decode阶段优化
```

---

## 4.4.4 本节小结

### ✅ 关键要点

1. **Triton**：高效GPU编程
2. **内核融合**：减少内存访问
3. **应用广泛**：Norm、RoPE、MoE
