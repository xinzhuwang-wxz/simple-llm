#!/usr/bin/env python3
"""
模拟器1：Tokenization与Attention可视化
==========================================

这个模拟器帮助理解：
1. Tokenization过程：文本如何转换为token
2. Attention计算：每个token如何关注其他tokens

运行方式：
    python tutorial/simulator/simulator_1_tokenization.py
"""

import time
import random


class SimpleTokenizer:
    """简化的分词器演示"""

    def __init__(self):
        # 模拟词汇表
        self.vocab = {
            "What": 100,
            "is": 200,
            "AI": 300,
            "?": 400,
            "artificial": 500,
            "intelligence": 600,
            "machine": 700,
            "learning": 800,
            "a": 900,
            "field": 1000,
            "of": 1100,
            "and": 1200,
            "in": 1300,
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        """将文本转换为token IDs"""
        words = text.replace("?", " ?").replace(".", " .").split()
        tokens = []
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # 未知词
                tokens.append(len(self.vocab) + 1)
        return tokens

    def decode(self, tokens):
        """将token IDs转换回文本"""
        words = []
        for token in tokens:
            if token in self.reverse_vocab:
                words.append(self.reverse_vocab[token])
            else:
                words.append("[UNK]")
        return " ".join(words)


def visualize_tokenization():
    """可视化Tokenization过程"""
    print("=" * 60)
    print("模拟器1: Tokenization 与 Attention 可视化")
    print("=" * 60)

    tokenizer = SimpleTokenizer()

    # 测试文本
    text = "What is AI?"

    print(f'\n📝 输入文本: "{text}"')
    print("-" * 40)

    # Tokenization
    tokens = tokenizer.encode(text)

    print("🔢 Tokenization 过程:")
    print(f"   原始文本: {text}")
    print(f"   分词结果: {tokens}")
    print(f"   解码验证: {tokenizer.decode(tokens)}")

    # 显示token详情
    print("\n📋 Token 详细信息:")
    for i, token in enumerate(tokens):
        word = tokenizer.decode([token]).strip()
        print(f'   Token {i + 1}: ID={token}, 文本="{word}"')

    time.sleep(1)
    visualize_attention(tokens)


def visualize_attention(tokens):
    """可视化Attention计算"""
    print("\n" + "=" * 60)
    print("🧠 Attention 机制可视化")
    print("=" * 60)

    num_tokens = len(tokens)

    print(f"\n📊 Attention矩阵 ({num_tokens}x{num_tokens})")
    print("-" * 40)

    # 模拟一个简化的attention矩阵（对角线更强）
    attention_scores = []
    for i in range(num_tokens):
        row = []
        for j in range(num_tokens):
            if j <= i:
                # 对角线及之前的token有更高的attention
                score = 1.0 if i == j else random.uniform(0.1, 0.5)
            else:
                score = 0.0
            row.append(score)
        attention_scores.append(row)

    # 归一化
    for i in range(num_tokens):
        total = sum(attention_scores[i])
        attention_scores[i] = [s / total for s in attention_scores[i]]

    # 打印矩阵
    print("       ", end="")
    for j in range(num_tokens):
        print(f"  T{j + 1}  ", end="")
    print()

    for i in range(num_tokens):
        print(f"  T{i + 1}  ", end="")
        for j in range(num_tokens):
            score = attention_scores[i][j]
            if score > 0.3:
                print(f" {score:.2f}*", end="")
            elif score > 0.1:
                print(f" {score:.2f} ", end="")
            else:
                print(f" 0.00 ", end="")
        print()

    # 解释
    print("\n💡 Attention 解释:")
    print("   - 每行表示：当前token关注哪些之前的token")
    print("   - 值越高表示关注程度越强")
    print("   - T3 (AI) 主要关注 T1 (What) 和 T2 (is)")

    print("\n" + "=" * 60)
    print("✅ 模拟器1完成：理解了Tokenization和Attention基本原理")
    print("=" * 60)


if __name__ == "__main__":
    visualize_tokenization()
