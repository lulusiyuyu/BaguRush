"""
build_index.py — 从 JSONL chunks 构建 FAISS 向量索引 + BM25 关键词索引

用法:
    cd bagurush/
    /mnt/d/ForWSL/env/bagurush/bin/python scripts/build_index.py

输出:
    faiss_index/tech_knowledge/  — FAISS 向量索引
    bm25_index/bm25.pkl         — BM25 关键词索引（pickle）
    bm25_index/chunks_meta.pkl  — chunk 元数据（用于 BM25 结果还原）
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

# ------------------------------------------------------------------ #
#  路径配置
# ------------------------------------------------------------------ #
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CHUNKS_FILE = PROJECT_DIR / "knowledge_base" / "chunks" / "all_chunks.jsonl"
FAISS_INDEX_DIR = PROJECT_DIR / "faiss_index" / "tech_knowledge"
BM25_INDEX_DIR = PROJECT_DIR / "bm25_index"

# 将项目根加入 path
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def load_chunks(filepath: Path) -> List[Dict]:
    """读取 JSONL 文件"""
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"[BuildIndex] 读取 {len(chunks)} 个 chunks")
    return chunks


def build_faiss_index(chunks: List[Dict]):
    """构建 FAISS 向量索引"""
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from rag.embeddings import get_embeddings

    print("[BuildIndex] 开始构建 FAISS 向量索引...")
    start = time.time()

    # 转换为 LangChain Document
    documents = []
    for c in chunks:
        doc = Document(
            page_content=c["text"],
            metadata={
                "chunk_id": c["chunk_id"],
                "source_repo": c["source_repo"],
                "source_file": c["source_file"],
                "source_path": c["source_path"],
                "topic": c["topic"],
                "subtopic": c["subtopic"],
                "file_type": c["file_type"],
                "language": c["language"],
            }
        )
        documents.append(doc)

    embeddings = get_embeddings()

    # 分批构建（避免一次性加载太多导致内存问题）
    BATCH_SIZE = 500
    vectorstore = None

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} docs)...")

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    # 保存索引
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_DIR))

    elapsed = time.time() - start
    print(f"[BuildIndex] FAISS 索引构建完成！")
    print(f"  向量数: {vectorstore.index.ntotal}")
    print(f"  保存路径: {FAISS_INDEX_DIR}")
    print(f"  耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    return vectorstore


def build_bm25_index(chunks: List[Dict]):
    """构建 BM25 关键词索引（jieba 中文分词）"""
    import jieba
    from rank_bm25 import BM25Okapi

    print("[BuildIndex] 开始构建 BM25 索引（jieba 分词）...")
    start = time.time()

    # jieba 分词
    tokenized_corpus = []
    for i, c in enumerate(chunks):
        tokens = list(jieba.cut(c["text"]))
        tokenized_corpus.append(tokens)
        if (i + 1) % 2000 == 0:
            print(f"  分词进度: {i + 1}/{len(chunks)}")

    bm25 = BM25Okapi(tokenized_corpus)

    # 保存
    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    bm25_path = BM25_INDEX_DIR / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    # 同时保存 chunks metadata 用于还原结果
    meta_path = BM25_INDEX_DIR / "chunks_meta.pkl"
    chunks_meta = []
    for c in chunks:
        chunks_meta.append({
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "source_repo": c["source_repo"],
            "source_file": c["source_file"],
            "source_path": c["source_path"],
            "topic": c["topic"],
            "subtopic": c["subtopic"],
            "file_type": c["file_type"],
            "language": c["language"],
        })
    with open(meta_path, "wb") as f:
        pickle.dump(chunks_meta, f)

    elapsed = time.time() - start
    print(f"[BuildIndex] BM25 索引构建完成！")
    print(f"  语料库大小: {len(tokenized_corpus)}")
    print(f"  保存路径: {BM25_INDEX_DIR}")
    print(f"  耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    return bm25


def main():
    if not CHUNKS_FILE.exists():
        print(f"[ERROR] 找不到 chunks 文件: {CHUNKS_FILE}")
        print("请先运行: python scripts/clean_and_chunk.py")
        sys.exit(1)

    chunks = load_chunks(CHUNKS_FILE)

    if not chunks:
        print("[ERROR] chunks 为空")
        sys.exit(1)

    # 先构建 BM25（快），再构建 FAISS（慢）
    build_bm25_index(chunks)
    print()
    build_faiss_index(chunks)

    print(f"\n{'='*50}")
    print("[BuildIndex] 全部索引构建完成！")
    print(f"  FAISS: {FAISS_INDEX_DIR}")
    print(f"  BM25:  {BM25_INDEX_DIR}")


if __name__ == "__main__":
    main()
