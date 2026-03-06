"""
RAG 知识库管理包。

公共导出：
    get_embeddings          — 获取嵌入模型实例（单例）
    load_document           — 加载单个文档
    load_directory          — 批量加载目录
    split_documents         — 文档切分
    VectorStoreManager      — FAISS 向量索引管理器
    create_session_store    — 创建 session 级简历临时索引
"""

from rag.embeddings import get_embeddings
from rag.document_loader import load_document, load_directory, split_documents
from rag.vector_store import VectorStoreManager, create_session_store

__all__ = [
    "get_embeddings",
    "load_document",
    "load_directory",
    "split_documents",
    "VectorStoreManager",
    "create_session_store",
]
