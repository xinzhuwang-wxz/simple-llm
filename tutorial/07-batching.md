# 3.1 连续批处理（Continuous Batching）

> 📍 **问题**：如果同时有多个用户请求，推理引擎应该如何处理？等所有请求到齐再一起处理，还是一个一个处理？

---

## 3.1.1 批处理的基本概念

### 💡 概念解释

**批处理**是同时处理多个请求以提高GPU利用率的技术。

### 📝 问题

**问题**：为什么需要批处理？

**启发**：
- GPU并行计算能力强
- 处理一个请求可能只用了很小一部分GPU能力
- 同时处理多个请求可以"填满"GPU

**答案**：
批处理的优势：
- **提高GPU利用率**：一次用满GPU
- **分摊固定开销**：内核启动等开销被多个请求分担
- **增加吞吐量**：单位时间处理更多请求

---

## 3.1.2 静态批处理的问题

### 📝 问题

**问题**：传统的静态批处理有什么问题？

**答案**：

```
时间 →
──────────────────────────────────────────────────────────────────────────────→

静态批处理:
请求A (需要100 tokens):  [======Prefill======][======Decode×100======]
请求B (需要10 tokens):        [等待...][======Prefill======][==Decode×10==]
请求C (需要50 tokens):                [等待...][======Prefill======][===Decode×50===]

时间点:
t=0:  A开始Prefill
t=1:  A完成Prefill，开始Decode；B在等待
t=2:  A Decode完成×1，B完成Prefill
... 

问题: B和C必须等待A Prefill完成才能开始！
     B只需要10 tokens，却要等待A的100 tokens全部完成！
```

### 🔍 深度思考

**问题**：这种问题会导致什么后果？

**答案**：
- **尾延迟高**：某些短请求需要等待长请求
- **用户体验差**：等待时间不可预测
- **资源浪费**：GPU可能处于等待状态

---

## 3.1.3 连续批处理的解决方案

### 💡 概念解释

**连续批处理**的核心思想：**允许新请求随时加入当前批次**

### 📝 工作流程

```
时间 →
──────────────────────────────────────────────────────────────────────────────→

连续批处理:
t=0:  [A] → Prefill
t=1:  [A] → Decode-1 [B加入] → Prefill
t=2:  [A] → Decode-2 [B] → Decode-1 [C加入] → Prefill
t=3:  [A完成!返回] [B] → Decode-2 [C] → Decode-1 [D加入] → Prefill
         ↑             ↑              ↑
       A的槽位       B的槽位         C的槽位
       立即释放      继续用          继续用

关键: 一个Decode步骤完成后，立即检查是否有请求完成
     有则移除并返回，腾出的槽位给新请求使用
```

---

### 📝 关键特点

| 特点 | 说明 |
|------|------|
| **动态加入** | 新请求可以随时加入当前批次 |
| **即时释放** | 请求完成后立即释放资源 |
| **GPU饱和** | GPU始终有工作可做 |
| **低延迟** | 短请求不需要等待长请求 |

---

## 3.1.4 连续批处理的实现细节

### 📝 代码位置

`llm.py` - `_inference_loop` 方法

### 🔧 代码解析

```python
# 位置：llm.py - _inference_loop (核心逻辑)

def _inference_loop(self):
    """
    连续批处理的核心循环
    """
    while True:
        # 1. 尝试从队列中获取新请求
        new_requests = self.request_queue.get_pending_requests()
        
        # 2. 如果有可用槽位，加入新请求进行Prefill
        available_slots = self.get_available_slots()
        for req, slot in zip(new_requests, available_slots):
            self._prefill(req.input_ids, slot)
            req.state = "decoding"
        
        # 3. 对所有正在Decode的请求执行一个Decode步骤
        active_requests = self.get_active_requests()
        for req in active_requests:
            next_token = self._decode_step(req.slot_id)
            req.generated_tokens.append(next_token)
            
            # 4. 检查是否完成
            if next_token == EOS or len(req.generated_tokens) >= req.max_tokens:
                # 完成！
                req.output = self.detokenize(req.generated_tokens)
                req.future.set_result(req.output)
                # 释放槽位
                self.free_slot(req.slot_id)
```

---

### 📝 槽位管理

```python
# 位置：llm.py - 槽位管理

class SlotManager:
    """管理GPU内存槽位"""
    
    def __init__(self, num_slots):
        self.num_slots = num_slots
        self.slot_states = ["free"] * num_slots  # free或occupied
    
    def allocate(self):
        """分配一个空闲槽位"""
        for i, state in enumerate(self.slot_states):
            if state == "free":
                self.slot_states[i] = "occupied"
                return i
        return None  # 没有可用槽位
    
    def free(self, slot_id):
        """释放槽位"""
        self.slot_states[slot_id] = "free"
```

---

## 3.1.5 连续批处理的性能优势

### 💡 概念解释

根据SimpleLLM的性能数据：

| 配置 | SimpleLLM | vLLM |
|------|-----------|------|
| batch_size=1 | 135 tok/s | 138 tok/s |
| batch_size=64 | 4,041 tok/s | 3,846 tok/s |

### 📊 分析

- **高并发时性能更好**：batch_size=64时SimpleLLM反而更快
- **延迟更稳定**：请求延迟更可预测
- **吞吐量更高**：单位时间处理更多tokens

---

## 3.1.6 连续批处理的挑战

### 🔍 深度思考

**问题**：连续批处理有什么挑战？

**答案**：

1. **槽位管理复杂**
   - 需要动态跟踪每个槽位的状态
   - 需要处理槽位不足的情况

2. **请求长度差异**
   - 不同请求的输出长度差异很大
   - 需要合理调度

3. **Prefill/Decode混合**
   - 新加入的请求需要Prefill
   - 正在处理的请求需要Decode
   - 两者计算模式不同

---

## 3.1.7 本节小结

### ✅ 关键要点

1. **静态批处理问题**：短请求需要等待长请求
2. **连续批处理解决**：动态加入，即时释放
3. **核心机制**：槽位管理 + 检查完成 + 资源释放
4. **性能优势**：高吞吐，低延迟

---

## 下节预告

> [3.2 Paged Attention与内存管理](./08-paged-attention.md) 将解释如何更高效地管理推理内存。
