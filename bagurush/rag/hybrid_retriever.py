"""
混合检索器 —— FAISS 语义 + BM25 关键词 + RRF 融合 + BGE Reranker 精排。

降级策略：
  - BM25 或 Reranker 故障时自动回退到纯语义检索
  - Reranker 未安装时跳过精排步骤，直接返回 RRF 结果
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 默认路径
_DEFAULT_BM25_DIR = str(_ROOT / "bm25_index")
_DEFAULT_FAISS_INDEX = str(_ROOT / "faiss_index" / "tech_knowledge")


# ------------------------------------------------------------------ #
#  RRF 融合
# ------------------------------------------------------------------ #

def reciprocal_rank_fusion(
    *result_lists: List[Document],
    k: int = 60,
) -> List[Document]:
    """
    Reciprocal Rank Fusion：将多路召回结果融合为统一排序。

    Args:
        *result_lists: 多个检索结果列表，各自按相关性降序排列。
        k: RRF 常数，默认 60（标准值）。

    Returns:
        去重并按 RRF 分数降序排列的 Document 列表。
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            # 用 page_content 的哈希作为去重 key（比 id 更安全）
            key = str(hash(doc.page_content))
            if key not in doc_map:
                doc_map[key] = doc
                scores[key] = 0.0
            scores[key] += 1.0 / (k + rank)

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


# ------------------------------------------------------------------ #
#  BM25 检索器（加载预构建的 pickle 索引）
# ------------------------------------------------------------------ #

class BM25Searcher:
    """基于预构建的 BM25 pickle 索引做关键词检索。"""

    def __init__(self, bm25_dir: str = _DEFAULT_BM25_DIR):
        bm25_path = Path(bm25_dir) / "bm25.pkl"
        meta_path = Path(bm25_dir) / "chunks_meta.pkl"

        if not bm25_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"BM25 索引文件不完整: {bm25_dir}\n"
                "请先运行 scripts/build_index.py 构建索引。"
            )

        with open(bm25_path, "rb") as f:
            self._bm25 = pickle.load(f)
        with open(meta_path, "rb") as f:
            self._chunks = pickle.load(f)

        print(f"[BM25] 加载完成: {len(self._chunks)} 个 chunks")

    def search(self, query: str, k: int = 20, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """BM25 关键词检索，返回 top-k Document。"""
        import jieba
        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)

        # 构建 (index, score) 对
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored_indices:
            if len(results) >= k:
                break
            if score <= 0:
                continue
            chunk = self._chunks[idx]
            metadata = chunk.get("metadata", {})
            # metadata filter
            if filter and not _match_filter(metadata, filter):
                continue
            results.append(Document(
                page_content=chunk.get("text", chunk.get("page_content", "")),
                metadata=metadata,
            ))
        return results


def _match_filter(metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
    """简单的 metadata 键值过滤。"""
    for key, value in filter.items():
        if isinstance(value, list):
            if metadata.get(key) not in value:
                return False
        elif metadata.get(key) != value:
            return False
    return True


# ------------------------------------------------------------------ #
#  BGE Reranker
# ------------------------------------------------------------------ #

class BGEReranker:
    """BGE Reranker 精排器，使用 sentence-transformers 的 CrossEncoder。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        self._model = CrossEncoder(model_name, device=device)
        print(f"[Reranker] 加载完成: {model_name} (device={device})")

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        """对候选文档做 Cross-Encoder 精排，返回 top-k。"""
        if not docs:
            return []
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


# ------------------------------------------------------------------ #
#  HybridRetriever 主类
# ------------------------------------------------------------------ #

class HybridRetriever:
    """
    多路召回混合检索器。

    Pipeline: FAISS 语义 + BM25 关键词 → RRF 融合 → BGE Reranker 精排
    任一环节异常时自动降级到纯 FAISS 语义检索。
    """

    def __init__(
        self,
        vectorstore_manager=None,
        bm25_dir: str = _DEFAULT_BM25_DIR,
        enable_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-base",
    ):
        self._vsm = vectorstore_manager
        self._bm25: Optional[BM25Searcher] = None
        self._reranker: Optional[BGEReranker] = None

        # 延迟加载 BM25
        try:
            self._bm25 = BM25Searcher(bm25_dir)
        except Exception as e:
            print(f"[HybridRetriever] BM25 加载失败，将降级: {e}")

        # 延迟加载 Reranker
        if enable_reranker:
            try:
                self._reranker = BGEReranker(reranker_model)
            except Exception as e:
                print(f"[HybridRetriever] Reranker 加载失败，跳过精排: {e}")

    def retrieve(
        self,
        query: str,
        final_k: int = 5,
        semantic_k: int = 20,
        bm25_k: int = 20,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        混合检索：语义 + BM25 → RRF → Reranker → top-k。

        Args:
            query: 查询文本。
            final_k: 最终返回的文档数。
            semantic_k: 语义检索召回数。
            bm25_k: BM25 召回数。
            filter: 可选的 metadata 过滤条件 dict。

        Returns:
            List[Document]: 精排后的 top-k 文档。
        """
        try:
            return self._hybrid_pipeline(query, final_k, semantic_k, bm25_k, filter)
        except Exception as e:
            print(f"[HybridRetriever] 混合检索失败，降级到纯语义检索: {e}")
            return self._fallback_search(query, final_k, filter)

    def _hybrid_pipeline(
        self, query: str, final_k: int, semantic_k: int, bm25_k: int,
        filter: Optional[Dict[str, Any]],
    ) -> List[Document]:
        """完整的混合检索流水线。"""
        result_lists = []

        # 1. FAISS 语义检索
        if self._vsm and self._vsm.vectorstore:
            if filter:
                semantic_results = self._vsm.vectorstore.similarity_search(query, k=semantic_k, filter=filter)
            else:
                semantic_results = self._vsm.vectorstore.similarity_search(query, k=semantic_k)
            result_lists.append(semantic_results)

        # 2. BM25 关键词检索
        if self._bm25:
            bm25_results = self._bm25.search(query, k=bm25_k, filter=filter)
            result_lists.append(bm25_results)

        if not result_lists:
            return []

        # 3. RRF 融合
        fused = reciprocal_rank_fusion(*result_lists)

        # 3.5 同源去重：同一文件最多保留 2 条，避免结果被单文件霸占
        fused = self._deduplicate_by_source(fused, max_per_source=2)

        # 4. Reranker 精排（如果可用）
        if self._reranker and len(fused) > final_k:
            return self._reranker.rerank(query, fused, top_k=final_k)

        return fused[:final_k]

    @staticmethod
    def _deduplicate_by_source(docs: List[Document], max_per_source: int = 2) -> List[Document]:
        """同一 source_file 最多保留 max_per_source 条。无 source 的文档不参与去重。"""
        source_count: Dict[str, int] = {}
        result = []
        for doc in docs:
            source = doc.metadata.get("source_file", doc.metadata.get("source", ""))
            if not source:
                result.append(doc)
                continue
            source_count[source] = source_count.get(source, 0) + 1
            if source_count[source] <= max_per_source:
                result.append(doc)
        return result

    def _fallback_search(
        self, query: str, k: int, filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """降级兜底：纯 FAISS 语义检索。"""
        if self._vsm and self._vsm.vectorstore:
            if filter:
                return self._vsm.vectorstore.similarity_search(query, k=k, filter=filter)
            return self._vsm.vectorstore.similarity_search(query, k=k)
        return []
