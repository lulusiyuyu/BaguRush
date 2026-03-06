"""
嵌入模型封装模块。

支持两种方案：
- local: 使用本地 HuggingFace BGE 模型（BAAI/bge-small-zh-v1.5），完全离线，推荐中文场景
- api  : 使用智谱 AI ZhipuAI Embeddings API（需配置 ZHIPUAI_API_KEY）

通过环境变量 EMBEDDING_PROVIDER 切换，默认 local。
"""

import os
import warnings
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

# 过滤 LangChain 内部版本兼容警告（HuggingFaceEmbeddings 内部仍引用旧类名，不影响功能）
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

load_dotenv()

EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    获取嵌入模型实例（单例，进程内缓存）。

    Returns:
        Embeddings: 已初始化的 LangChain Embeddings 实例。

    Raises:
        ValueError: 不支持的 EMBEDDING_PROVIDER 值。
        ImportError: 对应依赖库未安装时抛出。
    """
    provider = EMBEDDING_PROVIDER.strip().lower()

    if provider == "local":
        return _get_local_embeddings()
    elif provider == "api":
        return _get_api_embeddings()
    else:
        raise ValueError(
            f"不支持的嵌入模型提供商: '{provider}'，"
            f"请将 EMBEDDING_PROVIDER 设置为 'local' 或 'api'。"
        )


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


def _get_api_embeddings() -> Embeddings:
    """
    使用智谱 AI ZhipuAI Embeddings API。

    需要环境变量 ZHIPUAI_API_KEY。
    """
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError(
            "使用 API 嵌入方案时需要设置环境变量 ZHIPUAI_API_KEY。"
        )

    try:
        from langchain_community.embeddings import ZhipuAIEmbeddings
    except ImportError as e:
        raise ImportError(
            "使用智谱 AI 嵌入需要安装 zhipuai 包。\n"
            "请运行: pip install zhipuai"
        ) from e

    model = os.getenv("ZHIPUAI_MODEL", "embedding-3")
    print(f"[Embeddings] 使用智谱 AI API 模型: {model}")
    return ZhipuAIEmbeddings(api_key=api_key, model=model)
