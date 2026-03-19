"""HybridRetriever 和相关组件的单元测试。"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.hybrid_retriever import (
    BM25Searcher,
    HybridRetriever,
    _match_filter,
    reciprocal_rank_fusion,
)


# ------------------------------------------------------------------ #
#  RRF 融合测试
# ------------------------------------------------------------------ #

class TestRRF:
    def test_single_list(self):
        docs = [Document(page_content=f"doc{i}") for i in range(3)]
        result = reciprocal_rank_fusion(docs)
        assert len(result) == 3
        assert result[0].page_content == "doc0"  # rank 1 → highest score

    def test_two_lists_dedup(self):
        """两路结果有重叠文档时应去重。"""
        d1 = Document(page_content="shared")
        d2 = Document(page_content="only_semantic")
        d3 = Document(page_content="only_bm25")
        list1 = [d1, d2]
        list2 = [d3, d1]  # d1 在两个 list 中都出现
        result = reciprocal_rank_fusion(list1, list2)
        contents = [d.page_content for d in result]
        assert "shared" in contents
        assert "only_semantic" in contents
        assert "only_bm25" in contents
        assert len(result) == 3
        # shared 应该排名第一（两路都有，RRF 分数最高）
        assert result[0].page_content == "shared"

    def test_empty_input(self):
        result = reciprocal_rank_fusion([])
        assert result == []


# ------------------------------------------------------------------ #
#  Metadata filter 测试
# ------------------------------------------------------------------ #

class TestMatchFilter:
    def test_exact_match(self):
        assert _match_filter({"topic": "ml"}, {"topic": "ml"}) is True

    def test_mismatch(self):
        assert _match_filter({"topic": "ml"}, {"topic": "python"}) is False

    def test_list_filter(self):
        assert _match_filter({"topic": "ml"}, {"topic": ["ml", "dl"]}) is True
        assert _match_filter({"topic": "python"}, {"topic": ["ml", "dl"]}) is False

    def test_empty_filter(self):
        assert _match_filter({"topic": "ml"}, {}) is True


# ------------------------------------------------------------------ #
#  HybridRetriever 集成测试（用 mock）
# ------------------------------------------------------------------ #

class TestHybridRetriever:
    def _make_mock_vsm(self, docs):
        vsm = MagicMock()
        vsm.vectorstore = MagicMock()
        vsm.vectorstore.similarity_search.return_value = docs
        return vsm

    def test_fallback_when_no_bm25(self):
        """BM25 不可用时应降级到纯语义检索。"""
        docs = [Document(page_content=f"doc{i}") for i in range(3)]
        vsm = self._make_mock_vsm(docs)

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._vsm = vsm
        retriever._bm25 = None
        retriever._reranker = None

        result = retriever.retrieve("test query", final_k=3)
        assert len(result) == 3

    def test_hybrid_pipeline_without_reranker(self):
        """FAISS + BM25 融合（无 Reranker）。"""
        faiss_docs = [Document(page_content="faiss_doc")]
        bm25_docs = [Document(page_content="bm25_doc")]
        vsm = self._make_mock_vsm(faiss_docs)

        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = bm25_docs

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._vsm = vsm
        retriever._bm25 = mock_bm25
        retriever._reranker = None

        result = retriever.retrieve("test", final_k=5)
        contents = [d.page_content for d in result]
        assert "faiss_doc" in contents
        assert "bm25_doc" in contents

    def test_complete_fallback_on_exception(self):
        """混合检索异常时应降级到纯语义。"""
        docs = [Document(page_content="fallback")]
        vsm = self._make_mock_vsm(docs)

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._vsm = vsm
        retriever._bm25 = MagicMock()
        retriever._bm25.search.side_effect = RuntimeError("BM25 boom")
        retriever._reranker = None

        # _hybrid_pipeline 会抛异常 → retrieve 应降级
        result = retriever.retrieve("test", final_k=3)
        assert len(result) == 1
        assert result[0].page_content == "fallback"
