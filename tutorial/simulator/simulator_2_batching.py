#!/usr/bin/env python3
"""
模拟器2：批处理机制模拟
==========================================

这个模拟器帮助理解：
1. 静态批处理的问题：短请求等待长请求
2. 连续批处理的优势：动态加入，即时释放

运行方式：
    python tutorial/simulator/simulator_2_batching.py
"""

import time


class Request:
    """模拟的推理请求"""

    def __init__(self, id, prompt_length, max_tokens):
        self.id = id
        self.prompt_length = prompt_length
        self.max_tokens = max_tokens
        self.generated_tokens = 0
        self.start_time = None
        self.end_time = None
        self.status = "waiting"  # waiting, processing, completed


def simulate_static_batching(requests):
    """模拟静态批处理"""
    print("\n" + "=" * 60)
    print("📊 静态批处理 (Static Batching) 模拟")
    print("=" * 60)

    # 按到达时间排序（模拟同时到达）
    current_time = 0

    print("\n🔄 处理流程:")
    print("-" * 40)

    # 找到最长的请求，确定批次大小
    max_length = max(r.max_tokens for r in requests)

    # 模拟批处理
    print(f"📦 批次中的请求: {[r.id for r in requests]}")
    print(f"   最长需要生成: {max_length} tokens")

    for step in range(1, max_length + 1):
        time.sleep(0.3)  # 模拟每个decode步骤的时间

        completed = []
        for req in requests:
            if req.generated_tokens < req.max_tokens:
                req.generated_tokens += 1
                status = "🔵"
            else:
                completed.append(req.id)
                status = "✅"

        print(
            f"Step {step}: {[r.id for r in requests]} → 生成: {req.generated_tokens}/{req.max_tokens}"
        )

    # 所有请求完成
    print(f"\n📊 静态批处理结果:")
    for req in requests:
        print(f"   请求{req.id}: 完成，生成了{req.generated_tokens} tokens")

    # 计算平均延迟
    avg_latency = sum(req.max_tokens for req in requests) / len(requests)
    print(f"\n⏱️  平均延迟: {avg_latency} steps (每个请求都要等最长的那个)")


def simulate_continuous_batching(requests):
    """模拟连续批处理"""
    print("\n" + "=" * 60)
    print("📊 连续批处理 (Continuous Batching) 模拟")
    print("=" * 60)

    active_requests = []
    completed_requests = []
    waiting_queue = list(requests)
    step = 0

    print("\n🔄 处理流程:")
    print("-" * 40)

    while waiting_queue or active_requests:
        time.sleep(0.3)
        step += 1

        # 1. 新请求加入批次（如果有空闲槽位）
        while waiting_queue and len(active_requests) < 4:  # 假设最多4个槽位
            req = waiting_queue.pop(0)
            req.status = "processing"
            active_requests.append(req)
            print(f"Step {step}: 请求{req.id} 加入批次 (Prefill完成)")

        # 2. 每个活跃请求执行一个decode步骤
        still_active = []
        for req in active_requests:
            req.generated_tokens += 1

            if req.generated_tokens >= req.max_tokens:
                # 完成
                req.status = "completed"
                completed_requests.append(req)
                print(
                    f"Step {step}: 请求{req.id} ✅ 完成！({req.generated_tokens} tokens)"
                )
            else:
                # 继续处理
                still_active.append(req)

        active_requests = still_active

        # 显示当前状态
        if active_requests:
            active_ids = [r.id for r in active_requests]
            progress = [
                f"{r.id}:{r.generated_tokens}/{r.max_tokens}" for r in active_requests
            ]
            print(f"Step {step}: 活跃: {progress}")

    # 结果
    print(f"\n📊 连续批处理结果:")
    total_tokens = sum(r.generated_tokens for r in completed_requests)
    avg_latency = step  # 所有请求都在step步内完成
    print(f"   总步数: {step}")
    print(f"   总生成tokens: {total_tokens}")
    print(f"   有效吞吐量: {total_tokens / step:.2f} tokens/step")


def main():
    print("=" * 60)
    print("模拟器2: 批处理机制对比")
    print("=" * 60)

    print("""
💡 问题场景：
    有3个请求同时到达:
    - 请求A: 需要生成 5 tokens (短请求)
    - 请求B: 需要生成 10 tokens (中请求)
    - 请求C: 需要生成 20 tokens (长请求)
    
    比较两种批处理方式的延迟
    """)

    input("\n按回车继续...")

    # 创建请求
    requests = [
        Request("A", 5, 5),
        Request("B", 10, 10),
        Request("C", 20, 20),
    ]

    # 静态批处理
    requests_copy = [Request(r.id, r.prompt_length, r.max_tokens) for r in requests]
    simulate_static_batching(requests_copy)

    input("\n按回车继续看连续批处理...")

    # 连续批处理
    simulate_continuous_batching(requests)

    print("\n" + "=" * 60)
    print("📝 对比总结")
    print("=" * 60)
    print("""
    ┌─────────────────┬─────────────────┬─────────────────┐
    │                 │  静态批处理      │  连续批处理      │
    ├─────────────────┼─────────────────┼─────────────────┤
    │ 短请求A延迟      │ 20 steps        │  5 steps        │
    │ 中请求B延迟      │ 20 steps        │  10 steps       │
    │ 长请求C延迟      │ 20 steps        │  20 steps       │
    │ 平均延迟         │ 20 steps        │  ~11 steps      │
    │ GPU利用率        │ 可能有空闲       │  始终饱和        │
    └─────────────────┴─────────────────┴─────────────────┘
    
    结论: 连续批处理显著降低平均延迟！
    """)

    print("✅ 模拟器2完成：理解了批处理机制的差异")


if __name__ == "__main__":
    main()
