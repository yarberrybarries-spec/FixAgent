"""
测试 text_embedding 向量化和缓存
直接运行: python test_text_embedding.py
"""

import asyncio
import sys
sys.path.insert(0, '.')

from embeddings.text_embedding import get_text_embedding


async def test():
    print("=" * 50)
    print("Test text_embedding service")
    print("=" * 50)

    embedding = get_text_embedding()

    # 测试文本
    test_text = "电动机轴承过热是怎么回事？"

    # 第一次调用（无缓存）
    print(f"\nInput: {test_text}")
    print("First call (should call API)...")

    vector1 = await embedding.embed(test_text)
    print(f"Vector dimension: {len(vector1)}")
    print(f"First 5 values: {vector1[:5]}")

    # 第二次调用（应该有缓存）
    print("\nSecond call (should hit cache)...")
    vector2 = await embedding.embed(test_text)
    print(f"Vector dimension: {len(vector2)}")
    print(f"First 5 values: {vector2[:5]}")

    # 验证两次结果一致
    if vector1 == vector2:
        print("\n[OK] Cache works! Same result.")
    else:
        print("\n[FAIL] Results differ")

    print("\n" + "=" * 50)
    print("Test batch embedding")
    print("=" * 50)

    texts = [
        "轴承过热的原因",
        "电机保养方法",
        "轴承维护指南"
    ]

    vectors = await embedding.embed_batch(texts)
    for i, (text, vec) in enumerate(zip(texts, vectors)):
        print(f"\n[{i+1}] {text}")
        print(f"    Dimension: {len(vec)}")
        print(f"    First 3: {vec[:3]}")

    print("\n[Test complete!]")


if __name__ == "__main__":
    asyncio.run(test())
