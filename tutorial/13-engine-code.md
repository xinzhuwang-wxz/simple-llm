# 4.2 推理引擎核心代码解析

> 📍 **问题**：llm.py中的核心逻辑是什么？

---

## 4.2.1 LLM类结构

```python
# 位置：llm.py - LLM类

class LLM:
    def __init__(self, model_path, max_num_seqs=10, max_seq_len=1024):
        """初始化推理引擎"""
        # 1. 加载模型
        self.model = GptOssForCausalLM.from_pretrained(model_path)
        
        # 2. 加载tokenizer
        self.tokenizer = Tokenizer(model_path)
        
        # 3. 初始化KV Cache
        self._kv_cache = torch.zeros(...)
        
        # 4. 捕获CUDA Graph
        self._capture_cuda_graph()
```

---

## 4.2.2 核心方法

| 方法 | 功能 |
|------|------|
| `generate()` | 批量生成API |
| `chat()` | 对话API |
| `_prefill()` | Prefill阶段 |
| `_decode_step()` | Decode阶段 |
| `_sample_tokens()` | 采样策略 |

---

## 4.2.3 推理循环

```python
# 位置：llm.py - _inference_loop

def _inference_loop(self):
    """后台推理循环，实现连续批处理"""
    while True:
        # 1. 获取新请求
        new_requests = self.request_queue.get()
        
        # 2. 新请求进行Prefill
        for req in new_requests:
            self._prefill(req.input_ids, req.slot_id)
        
        # 3. 活跃请求执行Decode
        for req in self.active_requests:
            next_token = self._decode_step(req.slot_id)
            
            # 4. 检查是否完成
            if next_token == EOS:
                req.complete()
                self.free_slot(req.slot_id)
```

---

## 4.2.4 本节小结

### ✅ 关键要点

1. **异步默认**：推理在后台线程进行
2. **连续批处理**：新请求随时加入
3. **CUDA Graph**：Decode步骤加速
