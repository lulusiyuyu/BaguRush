"""
FAISS 向量存储管理模块。

提供 VectorStoreManager 类，封装向量索引的创建、追加、检索和持久化操作。
支持两种使用场景：
  1. 预置知识库索引（全局单例，持久化到磁盘）
  2. Session 级别的简历索引（每次面试会话独立，内存中）

命令行用法：
    python -m rag.vector_store --init
        一次性构建预置知识库的 FAISS 索引并保存到磁盘。

    python -m rag.vector_store --query "Python GIL 是什么"
        从知识库中检索相关内容（调试用）。
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 将项目根目录加入 sys.path，方便以 python -m 方式运行
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag.document_loader import load_directory, load_document, split_documents
from rag.embeddings import get_embeddings

# 默认的知识库目录和索引存储路径
_DEFAULT_KNOWLEDGE_DIR = str(_ROOT / "knowledge_base" / "tech")
_DEFAULT_INDEX_PATH = str(_ROOT / "faiss_index" / "tech_knowledge")


class VectorStoreManager:
    """
    FAISS 向量索引管理器。

    Attributes:
        index_path (str): 索引持久化目录路径。
        vectorstore (FAISS | None): 当前加载的 FAISS 向量库实例。
    """

    def __init__(self, index_path: str = _DEFAULT_INDEX_PATH) -> None:
        """
        初始化管理器。

        若指定路径下已存在索引文件，自动加载；否则 vectorstore 为 None，
        需调用 build_index() 或 add_documents() 后才可检索。

        Args:
            index_path: 索引文件目录，会自动创建。
        """
        self.index_path = index_path
        self.vectorstore: Optional[FAISS] = None
        self._embeddings = get_embeddings()

        if self._index_exists():
            self.load()

    # ------------------------------------------------------------------ #
    #  索引构建
    # ------------------------------------------------------------------ #

    def build_index(self, documents: List[Document]) -> None:
        """
        从文档列表全量构建（或覆盖）FAISS 索引。

        Args:
            documents: 已切分的 Document 列表。
        """
        if not documents:
            raise ValueError("documents 不能为空，请先加载文档。")

        print(f"[VectorStore] 开始构建索引，文档数: {len(documents)}")
        self.vectorstore = FAISS.from_documents(documents, self._embeddings)
        print(
            f"[VectorStore] 索引构建完成，向量数: {self.vectorstore.index.ntotal}"
        )

    def add_documents(self, documents: List[Document]) -> None:
        """
        向已有索引追加文档。若索引不存在则新建。

        Args:
            documents: 待追加的 Document 列表。
        """
        if not documents:
            return

        if self.vectorstore is None:
            self.build_index(documents)
        else:
            self.vectorstore.add_documents(documents)
            print(
                f"[VectorStore] 追加完成，当前向量数: {self.vectorstore.index.ntotal}"
            )

    # ------------------------------------------------------------------ #
    #  检索
    # ------------------------------------------------------------------ #

    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        语义检索 top-k 相关文档片段。

        Args:
            query: 检索查询文本。
            k:     返回结果数量（默认 3）。

        Returns:
            List[Document]: 相关度最高的 k 个 Document 片段。

        Raises:
            RuntimeError: 向量库尚未初始化时抛出。
        """
        if self.vectorstore is None:
            raise RuntimeError(
                "向量库尚未初始化，请先调用 build_index() 或 load()。"
            )
        return self.vectorstore.similarity_search(query, k=k)

    def search_with_score(self, query: str, k: int = 3):
        """
        检索并返回相似度得分。

        Returns:
            List[Tuple[Document, float]]: (document, score) 元组列表，score 越小越相关（L2 距离）。
        """
        if self.vectorstore is None:
            raise RuntimeError("向量库尚未初始化。")
        return self.vectorstore.similarity_search_with_score(query, k=k)

    # ------------------------------------------------------------------ #
    #  持久化
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        """
        将当前向量索引保存到 self.index_path 目录。
        """
        if self.vectorstore is None:
            raise RuntimeError("没有可保存的向量库，请先构建索引。")

        Path(self.index_path).mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(self.index_path)
        print(f"[VectorStore] 索引已保存至: {self.index_path}")

    def load(self) -> None:
        """
        从 self.index_path 加载已持久化的向量索引。
        """
        if not self._index_exists():
            raise FileNotFoundError(
                f"索引目录不存在或不完整: {self.index_path}\n"
                f"请先运行 `python -m rag.vector_store --init` 构建索引。"
            )
        self.vectorstore = FAISS.load_local(
            self.index_path,
            self._embeddings,
            allow_dangerous_deserialization=True,
        )
        print(
            f"[VectorStore] 索引加载完成: {self.index_path}，"
            f"向量数: {self.vectorstore.index.ntotal}"
        )

    # ------------------------------------------------------------------ #
    #  辅助
    # ------------------------------------------------------------------ #

    def _index_exists(self) -> bool:
        """检查索引目录中是否存在必要的 FAISS 文件。"""
        p = Path(self.index_path)
        return (p / "index.faiss").exists() and (p / "index.pkl").exists()

    @property
    def doc_count(self) -> int:
        """返回当前索引中的向量（文档 chunk）数量。"""
        if self.vectorstore is None:
            return 0
        return self.vectorstore.index.ntotal


# --------------------------------------------------------------------------- #
#  工厂函数：创建 session 级别的简历临时索引
# --------------------------------------------------------------------------- #

def create_session_store(session_id: str) -> VectorStoreManager:
    """
    为某个面试会话创建独立的简历向量索引（仅存于内存，不持久化）。

    Args:
        session_id: 唯一的会话 ID（用于日志标识）。

    Returns:
        VectorStoreManager: 空的 VectorStoreManager 实例（无预置索引路径）。
    """
    # 使用临时路径标识，但不会真正写入磁盘（除非手动调用 save()）
    tmp_path = str(Path(_ROOT) / "uploads" / f"session_{session_id}_index")
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.index_path = tmp_path
    manager.vectorstore = None
    manager._embeddings = get_embeddings()
    print(f"[VectorStore] 创建 session 临时索引: {session_id}")
    return manager


# --------------------------------------------------------------------------- #
#  __main__ 入口：--init 构建知识库 / --query 测试检索
# --------------------------------------------------------------------------- #

def _cmd_init(knowledge_dir: str, index_path: str) -> None:
    """构建并保存预置技术知识库的 FAISS 索引。"""
    print(f"[Init] 知识库目录: {knowledge_dir}")
    print(f"[Init] 索引输出路径: {index_path}")

    # 1. 加载所有 Markdown 技术文档
    docs = load_directory(knowledge_dir, extensions=(".md", ".txt"))
    if not docs:
        print("[Init] 警告：知识库目录中没有找到任何文档，请先创建知识库内容（Phase 1.4）。")
        return

    # 2. 切分文档
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)

    # 3. 构建向量索引
    manager = VectorStoreManager(index_path=index_path)
    manager.build_index(chunks)

    # 4. 持久化
    manager.save()

    print(
        f"\n✅ 知识库索引构建完成！\n"
        f"   文档段落: {len(docs)}\n"
        f"   Chunk 数: {len(chunks)}\n"
        f"   向量总数: {manager.doc_count}\n"
        f"   索引路径: {index_path}"
    )


def _cmd_query(query: str, index_path: str, k: int = 3) -> None:
    """从已有知识库中检索并打印结果。"""
    manager = VectorStoreManager(index_path=index_path)
    results = manager.search(query, k=k)

    print(f"\n🔍 查询: {query}\n{'='*60}")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "未知来源")
        print(f"\n[{i}] 来源: {Path(source).name}")
        print(doc.page_content[:400])
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BaguRush RAG 向量库管理工具"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--init",
        action="store_true",
        help="从 knowledge_base/tech/ 构建并保存 FAISS 索引",
    )
    group.add_argument(
        "--query",
        type=str,
        metavar="QUERY",
        help="从已有索引中检索（调试用）",
    )
    parser.add_argument(
        "--knowledge-dir",
        type=str,
        default=_DEFAULT_KNOWLEDGE_DIR,
        help=f"知识库目录（默认: {_DEFAULT_KNOWLEDGE_DIR}）",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=_DEFAULT_INDEX_PATH,
        help=f"索引存储路径（默认: {_DEFAULT_INDEX_PATH}）",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="检索返回结果数量（默认 3）",
    )

    args = parser.parse_args()

    if args.init:
        _cmd_init(args.knowledge_dir, args.index_path)
    elif args.query:
        _cmd_query(args.query, args.index_path, args.k)
