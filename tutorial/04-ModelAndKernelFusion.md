# Chapter 4: 极致的算子优化：Model 与 Kernel 融合 (Model & Kernel Fusion)

我们已经在上层搞定了并发调度。但当数据被喂到 GPU 嘴边时，如果计算方式不对，一切努力都是徒劳。

本章我们将深入 `model/model.py`。你会发现，原生的纯净版 PyTorch 代码在这里变得面目全非，取而代之的是一系列经过极限压榨的**底层算子 (Kernels)**。

这其中的核心理念只有一个词：**减少访存 (Reduce Memory Access)**。

---

## 4.1 算子融合 (Kernel Fusion) 的炼金术

要理解算子融合，必须先清楚 GPU 的脾气：**它算得太快，但吃得太慢。**

GPU 的计算单元极多，但它的显存 (HBM, High Bandwidth Memory) 读写速度相对缓慢。
当你用原生 PyTorch 写下 `Y = A * B + C` 时：
1. **启动乘法 Kernel**：从显存把 A 和 B 搬进 SRAM，算出中间结果 `T = A * B`，把 `T` 写回显存。
2. **启动加法 Kernel**：从显存把 `T` 和 `C` 搬进 SRAM，算出 `Y = T + C`，把 `Y` 写回显存。

那个可怜的中间张量 `T` 被写回显存，又马上被读出来。在生成几千个 Token 的漫长 Decode 中，这种毫无意义的搬运行为会“饿死”计算单元。

**算子融合 (Kernel Fusion)** 就是：用底层语言 (如 CUDA C++ 或 Triton) 写一段专门的代码，在这个 Kernel 内部一口气把乘法和加法算完。数据读进寄存器后，绝不写回显存，直到最终结果 `Y` 诞生！

### 实例 1: Fused QKV Projection
在标准的 Transformer 中，输入数据要分别乘以三个权重矩阵 $W_Q, W_K, W_V$。这需要 3 次 Kernel 启动。

看看 `model.py` 里是怎么做的：
```python
# model/model.py (第 206-209 行，Decode 前向传播片段)
# 一枪打爆：用一个融合的大权重矩阵，算出三个向量的拼接体
qkv = attn._qkv_proj(hidden)

# 纯粹在“视角 (View)”层面上切分，不发生任何物理显存数据的搬运！
query = qkv[..., :attn._q_size].view(batch_size, 1, attn.num_heads, attn.head_dim)
key = qkv[..., attn._q_size:attn._q_size+attn._kv_size].view(batch_size, 1, attn.num_kv_heads, attn.head_dim)
value = qkv[..., attn._q_size+attn._kv_size:].view(batch_size, 1, attn.num_kv_heads, attn.head_dim)
```

### 实例 2: Fused RoPE (旋转位置编码)
对于每一个词，我们要给它的 Query 和 Key 注入位置信息（比如“我是句子中的第5个词”）。原生的公式涉及复数乘法和大量的三角函数调用。
```python
# model/model.py (第 210 行)
# 直接调用底层 Triton 算子，在原地 (In-place) 暴力修改 query 和 key 的值
# 避免了创建位置编码张量的内存开销
query, key = attn.rotary_emb(positions, query, key)
```

### 实例 3: Fused RMSNorm + Residual
每一层都有残差连接和归一化 `norm(x + residual)`。
```python
# model/model.py (第 225 行)
# 将 Add 和 Norm 揉捏在一起，只需一次内存遍历
hidden, residual = layer.post_attention_layernorm(hidden, residual)
```

正是这些“抠门”到极致的优化，堆叠 36 层后，才换来了毫秒级的延迟降低。

---

## 4.2 终极核武：Flash Attention 2 的精妙调用

在没有 Flash Attention 的时代，标准 Attention 的公式 $Softmax(QK^T)V$ 会产生一个巨大的中间注意力矩阵（大小为 $N \times N$）。仅仅是将这个矩阵写进显存，就能让大模型直接 OOM。

Flash Attention 的核心魔法是：**切块 (Tiling) 计算，在线更新 Softmax 分母，在整个计算过程中不产生任何 $N \times N$ 的中间张量。**

在 SimpleLLM 中，你会看到引擎在不同阶段调用了不同的 Flash Attention 接口：

### Decode 阶段：带着 Cache 的闪电 (`flash_attn_with_kvcache`)

在 `model.py` 的 Decode 函数中，输入的新 Token 只有一个，但它要和缓存中的海量历史记录打交道。

```python
# model/model.py (第 215-217 行，Decode 核心计算)
attn_out, lse = flash_attn_with_kvcache(
    q=query,                   # [batch_size, 1, num_heads, head_dim] 新生成的 1 个词
    k_cache=attn._kv_cache,    # 巨大的静态槽位缓存 (K)
    v_cache=attn._v_cache,     # 巨大的静态槽位缓存 (V)
    k=key,                     # 刚算出来的新词的 K，需要塞进缓存里
    v=value,                   # 刚算出来的新词的 V，需要塞进缓存里
    cache_seqlens=attn._cache_seqlens[slot_indices].int(), # 这个槽位当前存了多少个词？
    cache_batch_idx=slot_indices.int(),                    # 当前请求占用了哪张桌子 (Slot)？
    ...
)
```

这个接口非常强悍。它不仅瞬间算完了注意力结果 `attn_out`，它还**顺手把新传入的 `k` 和 `v` 直接追加写进了 `attn._kv_cache` 的正确偏移位置上**！这把原来需要的内存追加拷贝操作也给融并了。

### Prefill 阶段：长短不一的大乱炖 (`flash_attn_varlen_func`)

回到 `llm.py` 中的 `_prefill` 函数。我们之前说过，传入的是一个拼在一起的长长的一维“香肠”。

```python
# llm.py (第 219-223 行，Prefill 核心计算)
attn_out, lse, _ = flash_attn_varlen_func(
    q, k, v, 
    cu_seqlens_q=cu_seqlens,  # [0, 10, 60, 160] 告诉 CUDA 句子的分割线在哪里！
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seq_len, max_seqlen_k=max_seq_len,
    ...
)
```

`flash_attn_varlen_func` (varlen 意为 Variable Length) 极其聪明。它根据 `cu_seqlens` 提供的断点，在内部将并行计算切割成互不干扰的小块。这样即使 Batch 中有一个 10 个词的句子和一个 1000 个词的句子，它们也不会互相干扰，更不需要插入毫无意义的 Padding 0 去浪费算力。

---

## 4.3 (进阶) Triton 与 4-bit 量化 MoE

如果你翻开代码看到：
```python
from kernels.moe import swizzle_mxfp4, moe_forward
```
恭喜你，你摸到了 2024 年大模型 Infra 最前沿的领域。

`gpt-oss-120b` 是一个包含 1200 亿参数的 **混合专家 (MoE) 模型**。为了把它塞进单张 80GB 的显卡，SimpleLLM 动用了极致的黑魔法：
1.  **MXFP4 超低比特量化**：模型的很大一部分权重被压缩到了 4-bit (半个字节)。但标准的 CUDA 核函数根本不认识 4-bit 数据。
2.  **稀疏路由 (Sparse Routing)**：每处理一个词，路由器会在 128 个“专家”网络中，只挑出最合适的 4 个进行计算。

怎么实现？只能靠手写底层算子。
在项目根目录的 `kernels/triton_kernels/` 下，你会找到用 Triton 语言编写的专用代码。Triton 是 OpenAI 开发的利器，它允许工程师用类似 Python 的高层语法，控制 GPU 最底层的流多处理器 (SM) 中的寄存器和共享内存 (SRAM) 行为。

比如，在推理时，Triton 算子会在极快的缓存 (SRAM) 中，实时将 4-bit 权重解压回浮点数，并在纳秒内完成矩阵乘法，最后再丢弃解压后的数据，绝不让它污染主显存！

---

至此，从高维度的状态机调度（`llm.py`），到最底层的指针偏移和算子重铸（`model.py`），你已经彻底打通了 SimpleLLM 的任督二脉。

理论与源码的探索到此结束。下一章，我们将撸起袖子，教你如何在真实的 H100 GPU 上一键启动这个猛兽，并在本地 Mac 上用模拟器亲眼见证代码中的设计在终端里翩翩起舞。

➡️ **[前往 Chapter 5: 实战与模拟器：动手验证你的理解](./05-SimulatorAndPractice.md)**