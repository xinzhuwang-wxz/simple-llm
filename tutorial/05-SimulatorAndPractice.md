# Chapter 5: 实战与模拟器：动手验证你的理解 (Hands-on & Simulator)

纸上得来终觉浅，绝知此事要躬行。理论看再多，也不如亲眼看着一行行 Token 在屏幕上蹦出来震撼。

本章分为两个部分：
1. **云端实战**：教你如何花几美元租用一台顶级的 H100 显卡，把 SimpleLLM 和 120B 大模型真正跑起来。
2. **Mac 本地模拟器**：如果你暂时不想花钱租卡，没关系！我们为你准备了一个纯 Python 写的文字版模拟器，直接在你的笔记本上可视化运行，亲眼见证前面讲过的 Batching 和 KV Cache 到底是怎么回事。

---

## 5.1 云端环境准备与实战运行 (H100 部署指南)

由于我们要运行的是一个量化后仍需 ~66.5GB 显存的 120B 巨兽模型，并且使用了一些目前只有 Hopper 架构才支持的极新特性，因此你**必须**使用一张具有 80GB 显存的 NVIDIA H100 PCIe 或 SXM 显卡，且驱动版本支持 CUDA 12.8。

如果你手头没有这种怪兽级硬件，最经济的方式是使用云服务（如 RunPod、Lambda Labs、Vast.ai 等）。

### 第一步：在 RunPod 上租用 H100

1. 注册并登录 [RunPod](https://www.runpod.io/)。
2. 充值大约 5-10 美元（H100 的按需价格大约在 $2.5 到 $4.5 每小时之间）。
3. 进入 **Pods** 页面，点击 **Deploy**。
4. 筛选 GPU 为 **1x H100 PCIe (或 SXM)**。
5. **极其重要的一步**：在选择镜像 (Template) 时，选择或搜索基于 `ubuntu22.04` 且预装了 `CUDA 12.8.x` 的基础镜像（或者使用 RunPod PyTorch 2.x 的官方镜像，并在启动后手动升级 CUDA 驱动）。
6. 配置存储：由于模型下载需要空间，请确保 Container Disk 分配了至少 150GB。
7. 点击 Deploy 启动实例，并使用 Web Terminal 或 SSH 连入。

### 第二步：环境配置与依赖安装

进入 H100 实例的终端后，按顺序执行以下命令：

```bash
# 1. 克隆 SimpleLLM 仓库
git clone https://github.com/your-username/simple-llm.git
cd simple-llm

# 2. （可选）如果你更喜欢用虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3. 运行一键安装脚本（此脚本将安装依赖并编译 Triton Kernels）
./setup.sh
```

**⚠️ 注意：下载 120B 模型权重**
`gpt-oss-120b` 的权重文件非常大，你需要先从 HuggingFace 上将它下载到本地目录 `./gpt-oss-120b` 中。
你可以使用 `huggingface-cli` 进行下载（确保你的硬盘空间足够）：
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download OpenAI/gpt-oss-120b --local-dir ./gpt-oss-120b
```

### 第三步：运行推理引擎

一切准备就绪！我们只需几行 Python 代码就能拉起引擎：

```python
# test_run.py
from llm import LLM

print("正在加载 120B 大模型...")
# 传入模型路径。引擎会自动探测 H100 剩余显存并计算最大并发数 (max_num_seqs)
engine = LLM("./gpt-oss-120b")

prompts = [
    "What is the meaning of life?",
    "Write a quick sort function in Python.",
    "Explain quantum computing to a 5 year old."
]

print("正在提交请求...")
# engine.generate 是异步的，它返回一个 concurrent.futures.Future 对象
futures = engine.generate(prompts, max_tokens=200)

print("等待生成结果 (此时 GPU 正在疯狂进行 Continuous Batching)...")
outputs = futures.result()

for i, out in enumerate(outputs):
    print(f"\n--- Prompt {i+1} ---")
    print(out.text)
```

运行它：`python test_run.py`。你将看到强大的 H100 在几秒钟内以数百 token/s 的高吞吐量喷涌出回答！

---

## 5.2 交互式推理模拟器 (无需 GPU，Mac 可用)

如果你现在在星巴克，用着没有 NVIDIA 显卡的 MacBook，又想巩固一下第 2 章和第 3 章的理论知识，这部分就是为你准备的。

我们在 `tutorial/simulator/` 目录下提供了一套**极简的控制台模拟器**。它不进行任何真实的矩阵运算，而是通过打印可视化的 ASCII 日志，向你“模拟”引擎内部正在发生的事情。

### 实验 1：直观感受 Continuous Batching

在这个模拟实验中，你会看到几个不同长度的请求是如何动态加入和退出 GPU Batch 的。

打开终端，进入模拟器目录并运行：
```bash
cd tutorial/simulator
python simulator_2_batching.py
```

**观察重点**：
1. **木桶效应的消失**：注意观察当短序列（例如请求 B）提前生成完 `[EOS]` 时，它的状态会立刻变为 `Done`。
2. **Slot 的腾空与复用**：请求 B 腾出的位置，是不是立刻被还在排队的请求 D 占上了？这正是我们在 `llm.py` 源码中看到的逻辑重现！

### 实验 2：洞察 KV Cache 的显存分配

在第 2 章中我们讲了 Paged Attention 为什么要像操作系统一样将显存分页。这个模拟器将向你展示“碎片”是如何产生的。

运行：
```bash
python simulator_3_kv_cache.py
```

**观察重点**：
1. **连续分配的浪费**：仔细看第一种模拟（传统连续分配），你会看到大片 `[ ]` 符号，那是预留给 `max_seq_len` 的空洞，也就是**内部碎片**。
2. **分页分配的紧凑**：再看第二种模拟（Paged Attention 风格）。显存被切成了极小的 Block（用 `[Blk 0]` 表示）。你会发现内存几乎被塞得严严实实，毫无浪费！它通过 Block Table 将物理地址和逻辑地址映射了起来。

---

## 🎉 结语与下一步

恭喜你！你已经读完了本教程的所有核心章节。

我们从最基础的**自回归**讲起，拆解了 **Prefill 与 Decode**，直面了 **KV Cache** 带来的显存爆炸。然后，我们学习了工业界引以为傲的 **Continuous Batching** 和 **Paged Attention/Slot 机制**，并深入到了 **SimpleLLM 的真实源码**中去寻找这些理论的落脚点。最后，我们甚至瞥见了 **Kernel Fusion** 和 **CUDA Graphs** 这些极致压榨硬件性能的底层魔法。

AI Infrastructure 是一个深不见底但也魅力无穷的领域。理解了 SimpleLLM，你就拥有了阅读更复杂引擎（如 vLLM, SGLightning）源码的钥匙。

**你的下一步可以做什么？**
*   **黑客时间**：尝试修改 `llm.py`。比如，你能把它的调度策略从先来先服务 (FIFO) 改成最短任务优先 (SJF) 吗？
*   **挑战多卡**：`gpt-oss-120b` 支持张量并行 (Tensor Parallelism)。去研究一下如何在一台挂载了 8 张 H100 的服务器上，把模型切分后在多张卡之间做高效通信（All-Reduce）。
*   **深入 Triton**：如果你对底层极度渴望，去看看官方的 [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)，试着自己写一个简单的矩阵乘法内核！

**旅途愉快！**

*(返回 [教程首页](./README.md))*

