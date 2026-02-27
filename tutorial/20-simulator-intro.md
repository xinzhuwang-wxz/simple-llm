# 6.1 模拟器使用说明

> 📍 **问题**：这些模拟器能在Mac上运行吗？如何使用？

---

## 6.1.1 模拟器概述

本教程提供了3个可以在Mac上运行的模拟器，帮助你理解推理引擎的核心概念：

| 模拟器 | 主题 | 运行命令 |
|--------|------|----------|
| 模拟器1 | Tokenization与Attention | `python simulator_1_tokenization.py` |
| 模拟器2 | 批处理机制对比 | `python simulator_2_batching.py` |
| 模拟器3 | KV Cache原理演示 | `python simulator_3_kv_cache.py` |

---

## 6.1.2 运行要求

- **操作系统**: macOS / Linux / Windows
- **Python版本**: 3.8+
- **依赖**: 无额外依赖（纯Python标准库）

---

## 6.1.3 运行方法

```bash
# 进入教程目录
cd tutorial

# 运行模拟器1
python simulator/simulator_1_tokenization.py

# 运行模拟器2
python simulator/simulator_2_batching.py

# 运行模拟器3
python simulator/simulator_3_kv_cache.py
```

---

## 6.1.4 模拟器设计理念

这些模拟器使用**纯Python标准库**实现，目的是：

1. **无需GPU**：在普通电脑上即可运行
2. **直观理解**：通过可视化展示核心概念
3. **对比学习**：通过对比加深理解
4. **实践验证**：先模拟理解，再阅读真实代码

---

## 6.1.5 建议学习顺序

```
第1步: 运行模拟器1 → 理解Tokenization和Attention
         ↓
第2步: 运行模拟器2 → 理解批处理机制
         ↓
第3步: 运行模拟器3 → 理解KV Cache
         ↓
第4步: 阅读SimpleLLM源码 → 理解真实实现
```

---

## 下节预告

> [6.2 模拟器1：Tokenization与Attention可视化](./21-sim-tokenization.md)
