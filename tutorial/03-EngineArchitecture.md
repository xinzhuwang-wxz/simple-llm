# Chapter 3: SimpleLLM 引擎架构与源码深度走读 (Engine Architecture)

在看懂了前两章的理论后，现在我们要刺穿概念的迷雾，直面代码的真相。

工业级的推理引擎（如 vLLM, TensorRT-LLM）由于背负了过多的历史包袱和跨平台兼容性，往往动辄几十万行代码，初学者根本无从下口。而 SimpleLLM 的设计哲学是：**用最少的代码（~950行），展示最核心、最原汁原味的现代推理技术**。

本章，我们将逐行剖析引擎的中枢神经：`llm.py`。这是将并发请求转化为硬件算力的“大管家”。

---

## 3.1 异步引擎架构与请求队列 (The Async Heartbeat)

如果推理引擎像一个普通的 Python 脚本一样，调用 `engine.generate()` 就开始阻塞算前向传播，那它就变成了单线程应用：算完 User 1 的句子，才能接 User 2。这叫同步 (Synchronous)。

要实现“连续批处理”，推理引擎**必须是异步的 (Asynchronous)**：接客和炒菜必须分开。

### `_Request`：工作的基本单元
在 `llm.py` 的开头，定义了引擎调度流转的最基本包裹：

```python
# llm.py (第 24-38 行)
@dataclass
class _Request:
    token_ids: list[list[int]]             # 用户的输入 Prompt 转换成的 Token IDs
    max_tokens: int                        # 生成长度限制
    temperature: float                     # 采样温度
    ignore_eos: bool                       # 是否忽略结束符
    future: concurrent.futures.Future      # 💡 核心！用于异步返回结果的句柄
    results: list = field(default_factory=list)          
    pending_indices: list = field(default_factory=list)  # 尚未分配到物理槽位的句子索引
```

这里的核心是 `future` 字段。
当你作为用户调用公有 API 提交请求时：

```python
# llm.py (第 266 行附近)
def generate(self, prompts: list[str], ...) -> concurrent.futures.Future:
    ...
    future = concurrent.futures.Future()
    # 把用户输入打包
    request = _Request(token_ids=token_ids, future=future, ...)
    # 丢进线程安全的队列
    self._request_queue.put(request)
    # 函数立刻返回，完全不阻塞主线程！
    return future
```

用户拿着 `future` 在自己的线程里喝茶等待（调用 `future.result()` 陷入阻塞），而引擎的“后厨”则在另一个线程里疯狂炒菜。

---

## 3.2 连续批处理的心脏：`_inference_loop`

真正的魔法发生在 `__init__` 中启动的一个永不停止的后台线程：`_inference_loop`。它是引擎的心脏，心脏每跳动一次（执行一次 `while` 循环），被称为一个 **Step (步)**。

让我们剥开它那几百行的外衣，看看最核心的骨架逻辑：

```python
# llm.py (第 451-500 行，核心骨架解析)
def _inference_loop(self):
    # 维护三组核心状态：
    free_slots = list(range(self.max_num_seqs))  # 当前空闲的桌子 (KV Cache Slots)
    active_generations = {}                      # 正在吃饭的客人 (正在生成的请求)
    pending_requests = []                        # 拿了号还在等位的客人

    # 只要引擎没关，就死循环执行流水线
    while self._loop_running or pending_requests or active_generations:
        
        # Phase 1: 从 _request_queue 捞取新客人，加入等位区 pending_requests
        self._drain_queue(pending_requests, ...)

        # 如果彻底没活干，歇1毫秒防死锁
        if not pending_requests and not active_generations:
            time.sleep(0.001); continue

        # Phase 2: 迎客进门！尽可能把排队的人塞进空闲的桌子里
        new_work = self._assign_slots_to_pending(pending_requests, free_slots)

        # Phase 3: 先上凉菜。对刚坐下的新客人执行大矩阵乘法（Prefill），填充 KV Cache
        if new_work: 
            self._run_prefill(new_work, free_slots, active_generations, ...)
        if not active_generations: continue

        # Phase 4: 上热菜。对所有吃着饭的人（包括刚上完凉菜的），算下个词（Decode）
        # 里面包含了 CUDA Graph 的重播逻辑！
        self._run_decode_step(active_generations, free_slots, ...)
```

这就是连续批处理的完整生命周期。它的代码之清晰，几乎可以作为操作系统的教科书。

---

## 3.3 迎客与送客：槽位生命周期管理

连续批处理之所以“连续”，关键在于 **“能上能下”**。
我们先看看在 Phase 2，引擎是如何把排队的请求分配到物理显存槽位上的：

```python
# llm.py (第 361-378 行)
def _assign_slots_to_pending(self, pending_requests: list, free_slots: list):
    new_work = []
    # 遍历所有排队的人
    for req in list(pending_requests):
        # 只要这个人还有句子没分配，而且店里还有空桌子
        while req.pending_indices and free_slots:
            idx = req.pending_indices.pop(0)
            slot = free_slots.pop()  # 💡 物理占座：从空闲池拿走一个 Slot ID
            new_work.append((req, idx, slot))
            
        # 如果这个人的请求全部分配到了物理槽位，他就不用排队了
        if not req.pending_indices:
            pending_requests.remove(req)
            
        # 如果桌子发完了，直接跳出循环去算力端
        if not free_slots: break
        
    return new_work
```

那当一个句子的 `<EOS>` 被生成出来，或者达到最大长度后，系统是如何“送客”的呢？在 `_run_decode_step` 中调用了 `_handle_decode_completion`：

```python
# llm.py (第 414-445 行，精简版)
def _handle_decode_completion(self, slot, token, ...):
    req, idx, tokens, ... = active_generations[slot]
    
    # 检查终止条件
    is_eos = (token == end_token_id and not req.ignore_eos)
    is_max_len = (len(tokens) >= req.max_tokens)
    
    if is_eos or is_max_len:
        # 💡 结账走人！把桌子擦干净（回收 Slot）
        free_slots.append(slot)           # Slot 重新回到空闲池！
        del active_generations[slot]      # 从活跃字典中除名
        
        # 记录最终结果
        req.results[idx] = GenerationOutput(...)
        
        # 如果这个 Request 的所有句子都生成完了，唤醒等待的用户线程！
        if all(r is not None for r in req.results):
            req.future.set_result(req.results)
```

看到这段代码，你是否恍然大悟？
`free_slots` 就像一个蓄水池。生成结束的请求把 `slot` 丢进池子；下一个 Step 的 `_assign_slots_to_pending` 瞬间把这个热乎的 `slot` 捞出来分配给新请求。**算力在这一吞一吐之间，没有哪怕 1 毫秒的浪费。**

---

## 3.4 Prefill 与 Decode 的分流执行

在引擎层面，`_run_prefill` 和 `_run_decode_step` 的职责分明。这里隐藏着极其精妙的批量构建逻辑。

### 变长序列处理与 Padding 消除 (Varlen Processing)

在传统的深度学习批处理中，不同长度的序列必须通过填充（Padding）对齐成规则的矩形张量。例如，将长度为 3 和 5 的两个序列打包，通常需要补 2 个无效的 Padding Token。在自注意力计算中，$O(N^2)$ 的复杂度使得计算这些 Padding Token 会造成极大的算力浪费。

为了消除 Padding，`_run_prefill` 采用了 **1D 扁平化拼接 (Flattening)** 结合 **累加序列长度 (Cumulative Sequence Lengths, cu_seqlens)** 的设计。这种设计是为了完美适配底层 `flash_attn_varlen_func` 算子的内存要求。

**图解：从二维矩阵到一维无缝张量的转换**
```text
原始输入 (2 个不同长度的序列):
Seq 0: [ A, B, C ]          (长度 = 3)
Seq 1: [ D, E, F, G, H ]    (长度 = 5)

传统 Padding 方式 (存在计算浪费):
[ [ A, B, C, 0, 0 ],
  [ D, E, F, G, H ] ]

SimpleLLM / FlashAttention 的 Varlen 扁平化方式:
flat_tokens = [ A, B, C, D, E, F, G, H ]
cu_seqlens  = [ 0,       3,             8 ] 
               ^        ^              ^
             起始索引  Seq 0 结束/Seq 1 起始   Seq 1 结束
```

我们来看代码中是如何实现这种数据结构转换的：

```python
# llm.py (第 157-179 行)
def _prefill(self, sequences: list[list[int]], slot_indices: list[int]) -> torch.Tensor:
    num_seqs, seq_lens = len(sequences), [len(s) for s in sequences]
    
    # 1. 扁平化拼接 (Flattening)
    # 通过列表推导式将嵌套列表打平。例如 [[1,2,3], [4,5,6,7,8]] 变为 [1,2,3,4,5,6,7,8]
    flat_tokens = torch.tensor([t for s in sequences for t in s], dtype=torch.long, ...)
    
    # 2. 构建累加序列长度数组 (cu_seqlens)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, ...)
    
    # 初始化长度为 N+1 的全零数组
    cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, ...)
    
    # 利用 cumsum(0) 计算前缀和。
    # 若 seq_lens 为 [3, 5]，cumsum 后为 [3, 8]。
    # 填入 cu_seqlens[1:] 后，数组变为 [0, 3, 8]。
    cu_seqlens[1:] = seq_lens_t.cumsum(0)
    ...
```

底层的 CUDA 算子接收到 `flat_tokens` 和 `cu_seqlens=[0, 3, 8]` 后，便能精确知道索引 `0~2` 属于序列 0，索引 `3~7` 属于序列 1，并在计算注意力矩阵时严格进行块对角化隔离（Block-Diagonal Masking），从而实现了零 Padding 浪费。

### CUDA Graph 的内存拓扑与回放路径分支 (Replay Paths)

如前文所述，CUDA Graph 的捕获（Capture）机制对显存地址的静态性有着苛刻的要求。引擎在初始化阶段，针对每一个可能的批量大小（Batch Size），预先分配了静态的输入张量（`_graph_input_ids`）和连续的槽位索引（`[0, 1, 2, ... N-1]`）。

在每次执行 `_decode_step` 时，活跃序列在显存中的物理排列可能出现**连续**或**非连续（碎片化）**两种拓扑状态，这就要求引擎必须进行运行时的路径分流。

**图解：槽位连续性对执行路径的影响**
```text
状态 A: 槽位连续 (命中 Fast Path)
活跃请求分配情况: [Req A] [Req B] [Req C]
槽位索引 (Slot Indices): [ 0, 1, 2 ]
匹配预设的静态内存地址 ---> 调用 CUDA Graph Replay

状态 B: 槽位空洞 (回退 Slow Path)
假设 Req B 刚刚生成了 <EOS> 被回收，此时的内存状态：
活跃请求分配情况: [Req A] [ 空闲 ] [Req C]
槽位索引 (Slot Indices): [ 0, 2 ] (出现断层)
内存地址与预设的连续空间不匹配 ---> 触发降级，调用原生 model.decode
```

以下是这套路由逻辑在源码中的严谨实现：

```python
# llm.py (第 237-255 行)
def _decode_step(self, input_ids: torch.Tensor, positions: torch.Tensor, slot_indices: list[int]):
    batch_size = len(slot_indices)

    # 分支 1: CUDA Graph 加速路径 (Fast Path)
    # 严格验证连续性：slot_indices 必须严格等于 [0, 1, ..., batch_size-1]
    # 并且针对该 batch_size 的图已完成捕获。
    if slot_indices == list(range(batch_size)) and batch_size in self._cuda_graphs:
        
        # 将动态输入张量拷贝至 Graph 绑定的静态内存地址
        self._graph_input_ids[batch_size].copy_(input_ids)
        self._graph_positions[batch_size].copy_(positions)
        
        # 触发 GPU 端的回放调度，彻底绕开 CPU 内核启动开销
        self._cuda_graphs[batch_size].replay()
        return self._graph_outputs[batch_size]

    # 分支 2: Python 原生分发路径 (Slow Path)
    # 当 slot_indices 存在碎片（如 [0, 2]），静态内存映射失效。
    # 系统降级为标准的 PyTorch 前向传播，此时 CPU 必须介入并逐一发射 Kernel Launch 指令。
    return self.model.decode(
        input_ids, 
        positions, 
        torch.tensor(slot_indices, device=self.device, dtype=torch.long)
    )
```

**动态系统下的工程折中：** 
在连续批处理（Continuous Batching）场景中，序列频繁的进入与退出必然导致显存槽位的非连续状态。引擎的设计选择是：容忍短暂的 `Slow Path` 降级开销，在随后的 Step 中，一旦新的排队请求被填入这些空洞槽位，使 `slot_indices` 再次恢复连续，系统将自动切回零 CPU 负载的 `Fast Path` 状态。

为了让这台 V8 引擎能经常跑在“高铁通道”上，我们需要在底层计算上将 GPU 的访存瓶颈挤到极限。
接下来，让我们打开 `model/model.py`，看看什么是真正的算子炼金术。

➡️ **[前往 Chapter 4: 极致的算子优化：Model 与 Kernel 融合](./04-ModelAndKernelFusion.md)**