"""
文档加载与切分模块。

提供三个公共函数：
- load_document(file_path)          — 加载单个文件（PDF / MD / TXT）
- load_directory(dir_path, glob)    — 批量加载目录下所有匹配文件
- split_documents(docs, ...)        — 使用 RecursiveCharacterTextSplitter 切分文档
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------------------------------- #
#  单文件加载
# --------------------------------------------------------------------------- #

def load_document(file_path: str) -> List[Document]:
    """
    根据文件后缀选择合适的 Loader 加载文档。

    支持格式：
    - .pdf  → PyPDFLoader
    - .md   → 直接读取文本（保留 Markdown 结构）
    - .txt  → TextLoader

    Args:
        file_path: 文档绝对路径或相对路径。

    Returns:
        List[Document]: LangChain Document 列表，每个 Document 含 page_content 和 metadata。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
        ValueError: 不支持的文件格式时抛出。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(str(path))
    elif suffix == ".md":
        return _load_markdown(str(path))
    elif suffix == ".txt":
        return _load_text(str(path))
    else:
        raise ValueError(
            f"不支持的文件格式: '{suffix}'，"
            f"当前支持: .pdf / .md / .txt"
        )


def _load_pdf(file_path: str) -> List[Document]:
    """使用 PyPDFLoader 加载 PDF，每页作为一个 Document。"""
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError as e:
        raise ImportError(
            "加载 PDF 需要安装 pypdf。\n请运行: pip install pypdf"
        ) from e

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    # 在 metadata 中记录来源文件名
    for doc in docs:
        doc.metadata.setdefault("source", file_path)
    return docs


def _load_markdown(file_path: str) -> List[Document]:
    """直接以文本方式读取 Markdown 文件，保留原始格式。"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": file_path})]


def _load_text(file_path: str) -> List[Document]:
    """使用 TextLoader 加载纯文本文件。"""
    try:
        from langchain_community.document_loaders import TextLoader
    except ImportError as e:
        raise ImportError(
            "加载 TXT 需要安装 langchain_community。\n"
            "请运行: pip install langchain-community"
        ) from e

    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", file_path)
    return docs


# --------------------------------------------------------------------------- #
#  目录批量加载
# --------------------------------------------------------------------------- #

def load_directory(
    dir_path: str,
    glob: str = "**/*",
    extensions: tuple = (".pdf", ".md", ".txt"),
) -> List[Document]:
    """
    批量加载目录下所有支持格式的文档。

    Args:
        dir_path:   目录路径。
        glob:       glob 匹配模式（默认递归匹配全部文件）。
        extensions: 允许的文件后缀元组（默认 .pdf / .md / .txt）。

    Returns:
        List[Document]: 合并后的 Document 列表。
    """
    dir_p = Path(dir_path)
    if not dir_p.is_dir():
        raise NotADirectoryError(f"目录不存在: {dir_path}")

    all_docs: List[Document] = []
    matched = sorted(dir_p.glob(glob))

    for file_path in matched:
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                docs = load_document(str(file_path))
                all_docs.extend(docs)
                print(f"[DocumentLoader] 已加载: {file_path.name}  ({len(docs)} 页/段)")
            except Exception as e:
                print(f"[DocumentLoader] 跳过 {file_path.name}，原因: {e}")

    print(f"[DocumentLoader] 目录加载完毕，共 {len(all_docs)} 个文档段落")
    return all_docs


# --------------------------------------------------------------------------- #
#  文档切分
# --------------------------------------------------------------------------- #

def split_documents(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    使用 RecursiveCharacterTextSplitter 对文档进行切分。

    Args:
        docs:          待切分的 Document 列表。
        chunk_size:    每个 chunk 的最大字符数（默认 500）。
        chunk_overlap: 相邻 chunk 的重叠字符数（默认 50）。

    Returns:
        List[Document]: 切分后的 Document 列表，metadata 会被继承。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 中文优先按段落/句子切分，避免切断语义
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(
        f"[DocumentLoader] 切分完成: {len(docs)} 段 → {len(chunks)} 个 chunk "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks
