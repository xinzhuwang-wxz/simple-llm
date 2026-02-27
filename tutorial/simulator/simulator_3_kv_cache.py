#!/usr/bin/env python3
"""
模拟器3：KV Cache 动态演示
==========================================

这个模拟器帮助理解：
1. KV Cache是什么：存储Key和Value
2. 有无KV Cache的差异
3. Prefill阶段和Decode阶段的分工

运行方式：
    python tutorial/simulator/simulator_3_kv_cache.py
"""

import time


def simulate_without_kv_cache():
    """模拟没有KV Cache的情况"""
    print("\n" + "=" * 60)
    print("❌ 没有 KV Cache 的情况")
    print("=" * 60)

    prompt = "What is AI?"
    prompt_tokens = len(prompt.split())
    generate_tokens = 10

    print(f'\n📝 输入: "{prompt}" ({prompt_tokens} tokens)')
    print(f"📝 需要生成: {generate_tokens} tokens")
    print("\n🔄 生成过程:")
    print("-" * 40)

    total_calculations = 0

    for step in range(1, generate_tokens + 1):
        # 每一步都要重新计算所有之前token的K和V
        current_seq_len = prompt_tokens + step
        # 注意力计算复杂度 O(n^2)
        calculations = current_seq_len**2
        total_calculations += calculations

        print(
            f"Step {step}: 序列长度={current_seq_len}, "
            f"注意力计算={calculations}次 (={current_seq_len}²)"
        )

        time.sleep(0.2)

    print(f"\n📊 统计:")
    print(f"   总注意力计算次数: {total_calculations}")
    print(f"   平均每步: {total_calculations / generate_tokens:.0f}次")


def simulate_with_kv_cache():
    """模拟有KV Cache的情况"""
    print("\n" + "=" * 60)
    print("✅ 有 KV Cache 的情况")
    print("=" * 60)

    prompt = "What is AI?"
    prompt_tokens = len(prompt.split())
    generate_tokens = 10

    print(f'\n📝 输入: "{prompt}" ({prompt_tokens} tokens)')
    print(f"📝 需要生成: {generate_tokens} tokens")
    print("\n🔄 生成过程:")
    print("-" * 40)

    # Prefill阶段
    print(f"\n📦 Prefill 阶段 (处理{prompt_tokens}个输入tokens):")
    prefill_calc = prompt_tokens**2
    print(f"   一次性计算所有{prompt_tokens}个token的K和V")
    print(f"   注意力计算: {prefill_calc}次 (= {prompt_tokens}²)")
    print(f"   → 结果存入 KV Cache")

    time.sleep(0.5)

    # Decode阶段
    print(f"\n🔄 Decode 阶段 (自回归生成{generate_tokens}个tokens):")
    total_calculations = prefill_calc

    for step in range(1, generate_tokens + 1):
        # Decode阶段只需要计算当前token的Q
        # 从Cache读取已有的K和V
        current_seq_len = prompt_tokens + step
        # 只计算新token的K和V
        new_calc = current_seq_len  # 线性增长
        total_calculations += new_calc

        print(
            f"Step {step}: 序列长度={current_seq_len}, "
            f"新K/V计算={new_calc}次 (线性增长)"
        )

        print(f"         + 读取Cache中的{current_seq_len - 1}个K/V")

        time.sleep(0.2)

    print(f"\n📊 统计:")
    print(f"   Prefill计算: {prefill_calc}次")
    print(f"   Decode计算: {total_calculations - prefill_calc}次")
    print(f"   总注意力计算次数: {total_calculations}")
    print(f"   平均每步: {total_calculations / (generate_tokens + 1):.0f}次")


def visualize_cache_state():
    """可视化KV Cache状态"""
    print("\n" + "=" * 60)
    print("🔍 KV Cache 状态可视化")
    print("=" * 60)

    print("""
    ┌─────────────────────────────────────────────────────────┐
    │                    KV Cache 内存布局                      │
    ├─────────────────────────────────────────────────────────┤
    │  槽位0  │ K[0] V[0] │ K[1] V[1] │ K[2] V[2] │ ...   │
    │  槽位1  │ K[0] V[0] │ K[1] V[1] │ K[2] V[2] │ ...   │
    │  槽位2  │ (空闲)                                           │
    │  槽位3  │ (空闲)                                           │
    │  ...    │                                                 │
    └─────────────────────────────────────────────────────────┘
    
    说明:
    - 每个槽位存储一个序列的KV Cache
    - SimpleLLM预分配固定数量的槽位
    - 序列完成时，槽位立即释放给新请求
    """)

    print("\n📈 内存使用示例:")
    print("   初始: 槽位0占用40MB, 槽位1占用35MB")
    print("   序列1完成后: 槽位0释放，可立即用于新请求")
    print("   → 无需等待，GPU始终有工作")


def main():
    print("=" * 60)
    print("模拟器3: KV Cache 动态演示")
    print("=" * 60)

    print("""
    💡 学习目标:
    1. 理解KV Cache的作用
    2. 了解有无KV Cache的计算差异
    3. 理解Prefill和Decode的分工
    """)

    input("\n按回车继续看无Cache的情况...")

    # 无Cache
    simulate_without_kv_cache()

    input("\n按回车继续看有Cache的情况...")

    # 有Cache
    simulate_with_kv_cache()

    input("\n按回车查看Cache状态可视化...")

    # 可视化
    visualize_cache_state()

    print("\n" + "=" * 60)
    print("📝 对比总结")
    print("=" * 60)
    print("""
    ┌─────────────────┬─────────────────┬─────────────────┐
    │                 │  无 KV Cache    │  有 KV Cache    │
    ├─────────────────┼─────────────────┼─────────────────┤
    │ Prefill        │ 无特殊处理      │ 一次性计算，存储  │
    │ Decode每步      │ O(n²)计算      │ O(n)计算        │
    │ 总计算量(10步)  │ ~1200次        │ ~155次          │
    │ 内存使用        │ 重复计算K/V    │ 缓存K/V         │
    └─────────────────┴─────────────────┴─────────────────┘
    
    结论: KV Cache显著减少重复计算！
    """)

    print("✅ 模拟器3完成：理解了KV Cache的核心原理")


if __name__ == "__main__":
    main()
