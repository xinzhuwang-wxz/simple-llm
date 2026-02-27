# 5.4 运行SimpleLLM推理

> 📍 **问题**：环境安装好了，现在应该怎么运行第一次推理？

---

## 5.4.1 简单批量生成

### 📝 代码示例

SimpleLLM提供了简单易用的API。首先创建一个测试脚本：

```python
# 文件：test_generation.py

from llm import LLM

# 初始化推理引擎
# 参数：模型路径、最大并发序列数、最大序列长度
engine = LLM("./gpt-oss-120b", max_num_seqs=10, max_seq_len=1024)

# 准备要生成的提示词
prompts = [
    "What is the meaning of life?",
    "Explain machine learning in simple terms.",
    "What is a neural network?",
    "How does photosynthesis work?",
    "What is quantum computing?"
]

# 异步生成
# 返回一个Future对象
request = engine.generate(prompts, max_tokens=200)

# 等待结果
results = request.result()

# 打印结果
for i, result in enumerate(results):
    print(f"\n=== Prompt {i+1} ===")
    print(f"Prompt: {prompts[i]}")
    print(f"Response: {result.text}")
    print(f"Reasoning: {result.reasoning}")

# 清理资源
engine.stop()
```

### 📝 运行命令

```bash
# 激活环境
source .venv/bin/activate

# 运行测试
python test_generation.py
```

---

## 5.4.2 交互式对话

### 📝 使用chat API

```python
# 文件：test_chat.py

from llm import LLM

# 初始化引擎
engine = LLM("./gpt-oss-120b", max_num_seqs=10, max_seq_len=1024)

# 多轮对话
messages = [
    {"role": "user", "content": "What is Python?"}
]

# 第一轮对话
response = engine.chat(messages, max_tokens=200).result()
print(f"Assistant: {response[0].text}")

# 添加助手回复，继续对话
messages.append({"role": "assistant", "content": response[0].text})
messages.append({"role": "user", "content": "How do I install it?"})

# 第二轮对话
response = engine.chat(messages, max_tokens=200).result()
print(f"Assistant: {response[0].text}")

# 清理
engine.stop()
```

### 📝 运行命令

```bash
python test_chat.py
```

---

## 5.4.3 使用示例脚本

### 📝 SimpleLLM自带的示例

SimpleLLM项目在 `cookbook/` 目录下提供了示例脚本：

```bash
# 进入示例目录
cd cookbook

# 运行简单示例
python simple.py

# 或运行交互式聊天
python chat.py ./gpt-oss-120b
```

---

## 5.4.4 性能基准测试

### 📝 测试吞吐量

```python
# 文件：benchmark.py

import time
from llm import LLM

engine = LLM("./gpt-oss-120b", max_num_seqs=64, max_seq_len=1024)

# 准备多个相同提示词
prompts = ["What is artificial intelligence?"] * 64

# 预热
_ = engine.generate(prompts[:1], max_tokens=10).result()

# 基准测试
start = time.time()
results = engine.generate(prompts, max_tokens=1000).result()
end = time.time()

elapsed = end - start
total_tokens = sum(len(r.text.split()) for r in results)

print(f"Total time: {elapsed:.2f}s")
print(f"Total tokens: {total_tokens}")
print(f"Tokens per second: {total_tokens/elapsed:.2f}")

engine.stop()
```

---

## 5.4.5 常见问题

### 📝 问题1：内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案**：
- 减少 `max_num_seqs`
- 减少 `max_seq_len`
- 使用更小的batch size

### 📝 问题2：模型加载失败

```
FileNotFoundError: Model not found
```

**解决方案**：
- 检查模型路径是否正确
- 确认模型文件已下载完成

### 📝 问题3：性能不稳定

**可能原因**：
- 首次运行需要编译Triton内核（正常，第二次会更快）
- GPU被其他进程占用

---

## 5.4.6 进阶使用

### 📝 自定义采样策略

```python
# 使用不同的采样参数
request = engine.generate(
    prompts, 
    max_tokens=200,
    temperature=0.7,      # 控制随机性
    top_p=0.9,           # Nucleus采样
    top_k=50,            # Top-K采样
    ignore_eos=False      # 是否忽略结束符
)
```

---

## 5.4.7 本节小结

### ✅ 关键要点

1. **初始化引擎**：`LLM(model_path)`
2. **批量生成**：`engine.generate(prompts, max_tokens)`
3. **对话模式**：`engine.chat(messages, max_tokens)`
4. **清理资源**：`engine.stop()`

---

## 恭喜完成入门！

现在你已经：
1. ✅ 理解了AI推理的核心概念
2. ✅ 学会了KV Cache、批处理等关键技术
3. ✅ 掌握了GPU环境配置
4. ✅ 能够运行SimpleLLM推理

### 后续学习建议

1. 阅读SimpleLLM源码，深入理解实现细节
2. 尝试修改代码，添加新功能
3. 学习vLLM等其他推理引擎
4. 尝试优化性能

---

## 附录：完整命令汇总

```bash
# 1. 连接GPU实例
ssh root@<your-ip>

# 2. 进入工作目录
cd /workspace/simple-llm

# 3. 激活环境
source .venv/bin/activate

# 4. 运行示例
python cookbook/simple.py
```
