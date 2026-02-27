# 4.1 SimpleLLM项目结构

> 📍 **问题**：SimpleLLM项目的代码是如何组织的？

---

## 4.1.1 整体目录结构

```
simple-llm/
├── llm.py                    # 推理引擎核心 (563行)
├── README.md                 # 项目说明
├── VERSION                   # 版本号
├── setup.sh                  # 安装脚本
│
├── model/                    # 模型定义
│   ├── __init__.py
│   ├── model.py              # GPT-OSS模型定义 (324行)
│   └── tokenizer.py          # 分词器 (92行)
│
├── kernels/                  # Triton内核
│   ├── __init__.py
│   ├── norm.py               # RMSNorm
│   ├── rope.py               # RoPE位置编码
│   ├── moe.py                # MoE前向传播
│   └── triton_kernels/       # 社区内核
│       ├── routing.py
│       ├── matmul_ogs.py
│       ├── swiglu.py
│       └── ...
│
└── cookbook/                 # 使用示例
    ├── simple.py             # 批量生成
    └── chat.py               # 交互式对话
```

---

## 4.1.2 各模块功能

| 模块 | 行数 | 功能 |
|------|------|------|
| `llm.py` | 563 | 推理引擎核心 |
| `model/model.py` | 324 | 模型架构定义 |
| `model/tokenizer.py` | 92 | 分词器 |

---

## 4.1.3 数据流

```
用户请求
   ↓
llm.py: generate()
   ↓
model/tokenizer.py: encode()
   ↓
model/model.py: forward()
   ↓
kernels/: Triton优化内核
   ↓
llm.py: decode()
   ↓
model/tokenizer.py: decode()
   ↓
用户响应
```

---

## 4.1.4 本节小结

### ✅ 关键要点

1. **核心文件**：llm.py（引擎）+ model/（模型）+ kernels/（内核）
2. **代码量适中**：约1000行，适合学习
3. **分层清晰**：易于理解和扩展
