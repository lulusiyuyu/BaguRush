"""
技术知识 RAG 检索工具。

@tool search_tech_knowledge(query) -> str

功能：
  使用 HybridRetriever（FAISS 语义 + BM25 关键词 + RRF + BGE Reranker）
  从预置技术知识库中检索相关内容。

  当 HybridRetriever 初始化失败时，自动降级到纯 VectorStoreManager 语义检索。

使用场景：
  - Interviewer Agent 出题前查阅相关知识
  - Evaluator Agent 获取参考答案
  - 回答技术问题时提供佐证材料
"""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from langchain.tools import tool

# 将项目根目录加入 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag.vector_store import VectorStoreManager

_DEFAULT_INDEX_PATH = str(_ROOT / "faiss_index" / "tech_knowledge")


@lru_cache(maxsize=1)
def _get_tech_store() -> VectorStoreManager:
    """获取技术知识库向量索引（单例）。"""
    return VectorStoreManager(index_path=_DEFAULT_INDEX_PATH)


@lru_cache(maxsize=1)
def _get_hybrid_retriever():
    """获取 HybridRetriever 实例（单例），失败时返回 None。"""
    try:
        from rag.hybrid_retriever import HybridRetriever
        store = _get_tech_store()
        retriever = HybridRetriever(
            vectorstore_manager=store,
            enable_reranker=True,
        )
        print("[KnowledgeRAG] HybridRetriever 初始化成功")
        return retriever
    except Exception as e:
        print(f"[KnowledgeRAG] HybridRetriever 初始化失败，将使用纯语义检索: {e}")
        return None


# --------------------------------------------------------------------------- #
#  @tool 定义
# --------------------------------------------------------------------------- #

@tool
def search_tech_knowledge(query: str, k: int = 3) -> str:
    """
    在技术知识库中检索与查询最相关的知识片段。

    使用多路召回（语义+关键词）+ Reranker 精排。
    当混合检索不可用时自动降级到纯语义检索。

    涵盖技术领域：Python 基础、数据结构与算法、系统设计、机器学习、推荐系统等。

    Args:
        query: 技术问题或关键词，例如 "Python GIL 是什么"、"B+树和哈希索引的区别"。
        k: 返回的知识片段数量，默认 3。

    Returns:
        格式化的检索结果字符串，包含知识片段内容和来源文档标注。
        若索引未建立，返回错误提示。
    """
    try:
        # 优先使用 HybridRetriever
        hybrid = _get_hybrid_retriever()
        if hybrid is not None:
            results = hybrid.retrieve(query, final_k=k)
        else:
            # 降级到纯语义检索
            store = _get_tech_store()
            results = store.search(query, k=k)

        if not results:
            return f"未找到与 '{query}' 相关的技术知识，请尝试更换关键词。"

        lines = [f"# 技术知识检索：{query}\n"]
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source_file", doc.metadata.get("source", "unknown"))
            lines.append(f"## [{i}] 来源：{source}")
            lines.append(doc.page_content.strip())
            lines.append("")

        return "\n".join(lines)

    except RuntimeError as e:
        return (
            f"❌ 技术知识库未初始化：{str(e)}\n"
            f"请先运行 `python -m rag.vector_store --init` 构建索引。"
        )
    except Exception as e:
        error_msg = f"知识检索失败: {type(e).__name__}: {str(e)}"
        print(f"[KnowledgeRAG] ❌ {error_msg}")
        return f"❌ {error_msg}"
