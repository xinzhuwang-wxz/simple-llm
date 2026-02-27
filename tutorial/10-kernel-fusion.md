# 3.4 内核融合与Triton

> 📍 **问题**：为什么需要融合多个GPU操作？Triton是什么？

---

## 3.4.1 内核融合的好处

### 💡 概念解释

每个GPU操作都需要：
- 内存读写
- 内核启动
- CPU-GPU通信

**融合**可以将多个操作合并为一个内核，减少开销。

### 📝 示例

| 融合前 | 融合后 |
|--------|--------|
| MatMul → ReLU → MatMul | 一个融合内核 |
| RMSNorm → 残差相加 | 一个融合内核 |

---

## 3.4.2 Triton简介

### 💡 概念解释

**Triton**是一种高效的GPU内核编程语言：

| 方案 | 难度 | 效率 |
|------|------|------|
| CUDA C++ | 难 | 高 |
| PyTorch | 易 | 中 |
| Triton | 中 | 高 |

---

## 3.4.3 SimpleLLM中的Triton

### 📝 代码位置

`kernels/norm.py` - RMSNorm内核

```python
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    input_ptr, weight_ptr, output_ptr,
    n_cols, eps, BLOCK_SIZE: tl.constexpr
):
    # Triton内核实现
    # 比PyTorch实现更高效
    ...
```

---

## 3.4.4 本节小结

### ✅ 关键要点

1. **融合**：减少内存访问和内核启动
2. **Triton**：易用且高效的GPU编程
3. **应用**：RMSNorm、RoPE、MoE等
