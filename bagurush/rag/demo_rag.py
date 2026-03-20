"""
RAG 四级 Pipeline 交互式测试 Demo。

用法：
    cd bagurush
    /mnt/d/ForWSL/env/bagurush/bin/python rag/demo_rag.py

会进入一个交互循环，你输入 query，它输出四级 Pipeline 每一步的结果。
输入 q 退出。

示例 query（408 相关）：
    什么是死锁
    进程和线程的区别
    TCP 三次握手
    B+树和哈希索引的区别
    LRU 缓存淘汰策略
    操作系统页面置换算法
    Redis 持久化 RDB AOF
    Python GIL
    HashMap 底层实现
"""

import os
import sys
import time
from pathlib import Path

# 加项目根目录到 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main():
    print("=" * 60)
    print("🔍 BaguRush RAG 四级 Pipeline 交互式测试")
    print("=" * 60)

    # ---- 1. 加载 FAISS ----
    print("\n[1/3] 加载 FAISS 向量索引...")
    print("  → 正在加载 BGE 嵌入模型（首次较慢）...")
    t0 = time.time()

    # 直接导入具体模块，绕过 rag/__init__.py 的 eager import
    import importlib
    embeddings_mod = importlib.import_module("rag.embeddings")
    get_embeddings = embeddings_mod.get_embeddings
    print(f"  → 嵌入模型导入完成 ({time.time() - t0:.1f}s)")

    print("  → 正在加载 FAISS 索引文件...")
    t1 = time.time()
    vector_store_mod = importlib.import_module("rag.vector_store")
    VectorStoreManager = vector_store_mod.VectorStoreManager
    vsm = VectorStoreManager()
    if vsm.vectorstore is None:
        print("❌ FAISS 索引未找到，请先运行: python -m rag.vector_store --init")
        return
    print(f"  ✅ FAISS 加载完成 (嵌入模型 {time.time() - t0:.1f}s, 索引 {time.time() - t1:.1f}s)")

    # ---- 2. 加载 BM25 ----
    print("\n[2/3] 加载 BM25 关键词索引...")
    bm25 = None
    try:
        t0 = time.time()
        hybrid_mod = importlib.import_module("rag.hybrid_retriever")
        BM25Searcher = hybrid_mod.BM25Searcher
        bm25 = BM25Searcher()
        print(f"  ✅ BM25 加载完成 ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  ⚠️ BM25 加载失败，将跳过关键词检索: {e}")

    # ---- 3. 加载 Reranker ----
    print("\n[3/3] 加载 BGE Reranker...")
    reranker = None
    try:
        t0 = time.time()
        BGEReranker = hybrid_mod.BGEReranker
        reranker = BGEReranker()
        print(f"  ✅ Reranker 加载完成 ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  ⚠️ Reranker 加载失败，将跳过精排: {e}")

    # ---- 4. RRF 函数 ----
    reciprocal_rank_fusion = hybrid_mod.reciprocal_rank_fusion

    print("\n" + "=" * 60)
    print("✅ 所有组件加载完成！输入 query 开始测试（输入 q 退出）")
    print("=" * 60)

    while True:
        print()
        query = input("🔎 请输入检索 query > ").strip()
        if query.lower() in ("q", "quit", "exit", ""):
            print("👋 再见！")
            break

        final_k = 3

        # ---- Step 1: FAISS 语义检索 ----
        print(f"\n{'─' * 50}")
        print(f"📌 Step 1: FAISS 语义检索 (Top-20)")
        t0 = time.time()
        faiss_results = vsm.search(query, k=20)
        faiss_time = time.time() - t0
        print(f"  找到 {len(faiss_results)} 条结果 ({faiss_time:.3f}s)")
        for i, doc in enumerate(faiss_results[:3], 1):
            source = doc.metadata.get("source_file", doc.metadata.get("source", "?"))
            print(f"  [{i}] ({source}) {doc.page_content[:80]}...")

        # ---- Step 2: BM25 关键词检索 ----
        bm25_results = []
        if bm25:
            print(f"\n📌 Step 2: BM25 关键词检索 (Top-20, jieba 分词)")
            t0 = time.time()
            bm25_results = bm25.search(query, k=20)
            bm25_time = time.time() - t0
            print(f"  找到 {len(bm25_results)} 条结果 ({bm25_time:.3f}s)")
            for i, doc in enumerate(bm25_results[:3], 1):
                source = doc.metadata.get("source_file", doc.metadata.get("source", "?"))
                print(f"  [{i}] ({source}) {doc.page_content[:80]}...")
        else:
            print(f"\n📌 Step 2: BM25 跳过（未加载）")

        # ---- Step 3: RRF 融合 ----
        print(f"\n📌 Step 3: RRF 融合 (k=60)")
        all_lists = [faiss_results]
        if bm25_results:
            all_lists.append(bm25_results)
        fused = reciprocal_rank_fusion(*all_lists)
        print(f"  融合后 {len(fused)} 条（去重）")
        for i, doc in enumerate(fused[:3], 1):
            source = doc.metadata.get("source_file", doc.metadata.get("source", "?"))
            print(f"  [{i}] ({source}) {doc.page_content[:80]}...")

        # ---- Step 4: Reranker 精排 ----
        if reranker and len(fused) > final_k:
            print(f"\n📌 Step 4: BGE Reranker 精排 (Top-{final_k})")
            t0 = time.time()
            final_results = reranker.rerank(query, fused, top_k=final_k)
            rerank_time = time.time() - t0
            print(f"  精排完成 ({rerank_time:.3f}s)")
        else:
            print(f"\n📌 Step 4: Reranker 跳过")
            final_results = fused[:final_k]

        # ---- 最终结果 ----
        print(f"\n{'═' * 50}")
        print(f"🏆 最终结果 (Top-{final_k})")
        print(f"{'═' * 50}")
        for i, doc in enumerate(final_results, 1):
            source = doc.metadata.get("source_file", doc.metadata.get("source", "?"))
            topic = doc.metadata.get("topic", "?")
            print(f"\n--- [{i}] 来源: {source} | topic: {topic} | 长度: {len(doc.page_content)}字 ---")
            print(doc.page_content.strip())


if __name__ == "__main__":
    main()
