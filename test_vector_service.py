"""
测试 vector_service 向量存储和检索
"""

import asyncio
import sys
sys.path.insert(0, '.')

from services.vector_service import get_vector_service
from embeddings.text_embedding import get_text_embedding


def print_results(query: str, results: list):
    """打印搜索结果"""
    print(f"    Query: {query}")
    print(f"    Results:")
    for i, r in enumerate(results):
        print(f"    [{i+1}] score={r.get('score', 0):.4f} | {r.get('text', '')[:50]}...")


def count_and_cleanup(vector_service, docs):
    """统计和清理"""
    count = vector_service.count()
    print(f"    Total: {count}")
    for doc in docs:
        vector_service.delete(doc["doc_id"])
    print(f"    Deleted {len(docs)} documents")


async def test_async():
    """异步测试"""
    print("=" * 50)
    print("Test VectorService (Async)")
    print("=" * 50)

    vector_service = get_vector_service()
    embedding = get_text_embedding()

    docs = [
        {
            "doc_id": "doc_async_001",
            "text": "电动机轴承过热可能是由于润滑不足导致的",
            "metadata": {"category": "motor", "type": "故障"}
        },
        {
            "doc_id": "doc_async_002",
            "text": "定期给轴承加润滑油可以延长电机寿命",
            "metadata": {"category": "motor", "type": "保养"}
        },
        {
            "doc_id": "doc_async_003",
            "text": "轴承温度过高应该立即停机检查",
            "metadata": {"category": "motor", "type": "维修"}
        },
        {
            "doc_id": "doc_async_004",
            "text": "电流过大可能导致电机发热",
            "metadata": {"category": "motor", "type": "故障"}
        },
        {
            "doc_id": "doc_async_005",
            "text": "电机应该安装在通风良好的环境中",
            "metadata": {"category": "motor", "type": "安装"}
        }
    ]

    print("\n[1] Async embed + add vectors...")
    vectors = await embedding.embed_batch([d["text"] for d in docs])
    for i, doc in enumerate(docs):
        doc["vector"] = vectors[i]  # type: ignore

    success = vector_service.add_vector_batch(docs)
    print(f"    Added {success}/{len(docs)} documents")

    print("\n[2] Search by text (async)...")
    query = "轴承发热是什么原因"
    results = await vector_service.search_by_text(query, top_k=3)
    print_results(query, results)

    print("\n[3] Count vectors...")
    count_and_cleanup(vector_service, docs)

    print("\n[Async test complete!]")


def test_sync():
    """同步测试（需要手动调用embed）"""
    print("=" * 50)
    print("Test VectorService (Sync)")
    print("=" * 50)

    vector_service = get_vector_service()
    embedding = get_text_embedding()

    docs = [
        {
            "doc_id": "doc_sync_001",
            "text": "电动机轴承过热可能是由于润滑不足导致的",
            "metadata": {"category": "motor", "type": "故障"}
        },
        {
            "doc_id": "doc_sync_002",
            "text": "定期给轴承加润滑油可以延长电机寿命",
            "metadata": {"category": "motor", "type": "保养"}
        },
        {
            "doc_id": "doc_sync_003",
            "text": "轴承温度过高应该立即停机检查",
            "metadata": {"category": "motor", "type": "维修"}
        },
        {
            "doc_id": "doc_sync_004",
            "text": "电流过大可能导致电机发热",
            "metadata": {"category": "motor", "type": "故障"}
        },
        {
            "doc_id": "doc_sync_005",
            "text": "电机应该安装在通风良好的环境中",
            "metadata": {"category": "motor", "type": "安装"}
        }
    ]

    print("\n[1] Add vectors (sync)...")
    # 手动构造向量用于同步测试（实际使用时应该用异步embed）
    for doc in docs:
        doc["vector"] = [0.0] * 1024  # 占位向量，仅供测试add功能

    success = vector_service.add_vector_batch(docs)
    print(f"    Added {success}/{len(docs)} documents (using placeholder vectors)")

    print("\n[2] Search by text...")
    query = "轴承发热是什么原因"
    results = vector_service.search_by_text(query, top_k=3)
    print_results(query, results)

    print("\n[3] Count vectors...")
    count_and_cleanup(vector_service, docs)

    print("\n[Sync test complete!]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true", help="Run sync test")
    args = parser.parse_args()

    if args.sync:
        test_sync()
    else:
        asyncio.run(test_async())
