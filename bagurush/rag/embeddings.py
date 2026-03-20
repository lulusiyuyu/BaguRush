"""
嵌入模型封装模块。

使用本地 HuggingFace BGE 模型（BAAI/bge-small-zh-v1.5），完全离线，推荐中文场景。
"""

import os
import warnings
from functools import lru_cache

# 智能离线模式：模型已缓存则跳过联网检查，未缓存则允许下载
if not os.environ.get("HF_HUB_OFFLINE"):
    _cache_dir = os.path.join(
        os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")),
        "hub",
    )
    # 检查嵌入模型和 Reranker 模型是否都已缓存
    _embed_cache = os.path.join(_cache_dir, "models--BAAI--bge-small-zh-v1.5")
    _reranker_cache = os.path.join(_cache_dir, "models--BAAI--bge-reranker-base")
    if os.path.isdir(_embed_cache) and os.path.isdir(_reranker_cache):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

# 过滤 LangChain 内部版本兼容警告（HuggingFaceEmbeddings 内部仍引用旧类名，不影响功能）
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

load_dotenv()


def _auto_device() -> str:
    """自动检测可用设备：优先 CUDA，否则 CPU。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", _auto_device())


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    获取本地 BGE 嵌入模型实例（单例，进程内缓存）。

    Returns:
        Embeddings: 已初始化的 LangChain Embeddings 实例。
    """
    return _get_local_embeddings()


def _get_local_embeddings() -> Embeddings:
    """
    加载本地 HuggingFace BGE 嵌入模型。

    模型名称由 EMBEDDING_MODEL_NAME 控制（默认 BAAI/bge-small-zh-v1.5）。
    运行设备由 EMBEDDING_DEVICE 控制（默认 cpu）。
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings as HuggingFaceEmbeddings
        except ImportError as e:
            raise ImportError(
                "使用本地嵌入模型需要安装 langchain_huggingface 或 langchain_community。\n"
                "请运行: pip install langchain-huggingface"
            ) from e

    model_kwargs = {"device": EMBEDDING_DEVICE}
    # BGE 模型推荐的 encode_kwargs：使用余弦相似度时归一化
    encode_kwargs = {"normalize_embeddings": True}

    print(f"[Embeddings] 加载本地模型: {EMBEDDING_MODEL_NAME} (device={EMBEDDING_DEVICE})")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embeddings
