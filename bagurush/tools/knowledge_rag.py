"""
技术知识 RAG 检索工具。

@tool search_tech_knowledge(query) -> str

功能：
  从预置的技术知识库（FAISS 索引）中语义检索相关内容。
  索引由 `python -m rag.vector_store --init` 构建，包含 Python 基础、数据结构、
  系统设计、机器学习、推荐系统等技术文档。

使用场景：
  - Interviewer Agent 出题前查阅相关知识
  - Evaluator Agent 获取参考答案
  - 回答技术问题时提供佐证材料
"""

import sys
from functools import lru_cache
from pathlib import Path

from langchain.tools import tool

# 将项目根目录加入 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag.vector_store import VectorStoreManager

_DEFAULT_INDEX_PATH = str(_ROOT / "faiss_index" / "tech_knowledge")


@lru_cache(maxsize=1)
def _get_tech_store() -> VectorStoreManager:
    """获取技术知识库向量索引（单例，进程内缓存）。"""
    manager = VectorStoreManager(index_path=_DEFAULT_INDEX_PATH)
    return manager


# --------------------------------------------------------------------------- #
#  @tool 定义
# --------------------------------------------------------------------------- #

@tool
def search_tech_knowledge(query: str, k: int = 3) -> str:
    """
    在技术知识库中语义检索与查询最相关的知识片段。

    涵盖以下技术领域：
    - Python 语言基础（GIL、内存管理、装饰器、异步等）
    - 数据结构与算法（排序、树、图、动态规划等）
    - 系统设计（缓存、消息队列、分布式、限流熔断等）
    - 机器学习（模型、训练、评估指标等）
    - 推荐系统（协同过滤、矩阵分解、序列推荐等）

    Args:
        query: 技术问题或关键词，例如 "Python GIL 是什么"、"B+树和哈希索引的区别"。
        k: 返回的知识片段数量，默认 3。

    Returns:
        格式化的检索结果字符串，包含知识片段内容和来源文档标注。
        若索引未建立，返回错误提示。
    """
    try:
        store = _get_tech_store()
        results = store.search(query, k=k)

        if not results:
            return f"未找到与 '{query}' 相关的技术知识，请尝试更换关键词。"

        lines = [f"# 技术知识检索：{query}\n"]
        for i, doc in enumerate(results, 1):
            source = Path(doc.metadata.get("source", "unknown")).stem
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
