"""
verify_index.py — 抽样检索验证索引质量

用法:
    cd bagurush/
    /mnt/d/ForWSL/env/bagurush/bin/python scripts/verify_index.py
"""

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FAISS_INDEX_DIR = PROJECT_DIR / "faiss_index" / "tech_knowledge"
BM25_INDEX_DIR = PROJECT_DIR / "bm25_index"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# 10 个测试 query，覆盖所有 topic
TEST_QUERIES = [
    {"query": "Python GIL 全局解释器锁是什么", "expected_topic": "cs_408"},
    {"query": "HashMap 和 ConcurrentHashMap 的区别", "expected_topic": "java_backend"},
    {"query": "什么是 Transformer 的自注意力机制", "expected_topic": "ml_dl"},
    {"query": "RAG 检索增强生成的基本架构", "expected_topic": "llm_rag_agent"},
    {"query": "TCP 三次握手和四次挥手", "expected_topic": "cs_408"},
    {"query": "Spring Boot 自动配置原理", "expected_topic": "java_backend"},
    {"query": "梯度消失和梯度爆炸的原因和解决方法", "expected_topic": "ml_dl"},
    {"query": "LangChain Agent 的工作流程", "expected_topic": "llm_rag_agent"},
    {"query": "Redis 持久化 RDB 和 AOF", "expected_topic": "java_backend"},
    {"query": "操作系统进程和线程的区别", "expected_topic": "cs_408"},
]


def verify_faiss():
    """验证 FAISS 索引"""
    from langchain_community.vectorstores import FAISS
    from rag.embeddings import get_embeddings

    print("=" * 60)
    print("[Verify] FAISS 向量检索验证")
    print("=" * 60)

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(FAISS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )
    print(f"  向量总数: {vectorstore.index.ntotal}")

    hits = 0
    for i, tq in enumerate(TEST_QUERIES):
        start = time.time()
        results = vectorstore.similarity_search(tq["query"], k=5)
        elapsed = time.time() - start

        topics = [r.metadata.get("topic", "?") for r in results]
        top_topic = max(set(topics), key=topics.count) if topics else "?"
        hit = tq["expected_topic"] in topics
        if hit:
            hits += 1

        print(f"\n  Q{i+1}: {tq['query']}")
        print(f"    期望topic: {tq['expected_topic']} | 召回topics: {topics} | {'✓' if hit else '✗'} | {elapsed:.3f}s")
        print(f"    Top1: [{results[0].metadata.get('source_repo')}] {results[0].page_content[:80]}...")

    print(f"\n  FAISS 命中率: {hits}/{len(TEST_QUERIES)} ({hits/len(TEST_QUERIES)*100:.0f}%)")
    return hits


def verify_bm25():
    """验证 BM25 索引"""
    import jieba
    from rank_bm25 import BM25Okapi

    print("\n" + "=" * 60)
    print("[Verify] BM25 关键词检索验证")
    print("=" * 60)

    with open(BM25_INDEX_DIR / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_INDEX_DIR / "chunks_meta.pkl", "rb") as f:
        chunks_meta = pickle.load(f)

    print(f"  语料库大小: {len(chunks_meta)}")

    hits = 0
    for i, tq in enumerate(TEST_QUERIES):
        start = time.time()
        query_tokens = list(jieba.cut(tq["query"]))
        scores = bm25.get_scores(query_tokens)

        # 取 top 5
        top_indices = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:5]
        top_results = [chunks_meta[j] for j in top_indices]
        elapsed = time.time() - start

        topics = [r["topic"] for r in top_results]
        hit = tq["expected_topic"] in topics
        if hit:
            hits += 1

        print(f"\n  Q{i+1}: {tq['query']}")
        print(f"    期望topic: {tq['expected_topic']} | 召回topics: {topics} | {'✓' if hit else '✗'} | {elapsed:.3f}s")
        print(f"    Top1: [{top_results[0]['source_repo']}] {top_results[0]['text'][:80]}...")

    print(f"\n  BM25 命中率: {hits}/{len(TEST_QUERIES)} ({hits/len(TEST_QUERIES)*100:.0f}%)")
    return hits


def main():
    faiss_hits = verify_faiss()
    bm25_hits = verify_bm25()

    print(f"\n{'='*60}")
    print(f"[Verify] 验证总结")
    print(f"  FAISS: {faiss_hits}/10 topic 命中")
    print(f"  BM25:  {bm25_hits}/10 topic 命中")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
