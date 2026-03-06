"""
运行时 LLM 配置管理。

前端通过 HTTP 头传入 API Key / Base URL / Model，
后端在处理请求前调用 set_runtime_config() 设置，
各 agent/tool 的 _get_llm() 优先使用此配置，
回退到 os.getenv() 读取 .env 文件。
"""

import os
from typing import List, Optional

from langchain_openai import ChatOpenAI

# 当前请求的 LLM 覆盖配置（单用户场景足够安全）
_runtime: dict = {}

# 当前请求的流式回调（用于 LLM 数据流推送）
_stream_callbacks: list = []


def set_runtime_config(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
):
    """在每个请求开始时调用，设置前端传入的 LLM 配置。"""
    global _runtime
    _runtime = {}
    if api_key:
        _runtime["api_key"] = api_key
    if base_url:
        _runtime["base_url"] = base_url
    if model:
        _runtime["model"] = model


def clear_runtime_config():
    global _runtime
    _runtime = {}


def set_stream_callbacks(callbacks: list):
    """设置当前请求的流式回调处理器。"""
    global _stream_callbacks
    _stream_callbacks = callbacks or []


def clear_stream_callbacks():
    global _stream_callbacks
    _stream_callbacks = []


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """
    创建 ChatOpenAI 实例。
    优先使用 runtime 配置（前端传入），回退到环境变量。
    如果有 stream_callbacks，启用 streaming 模式。
    """
    kwargs = dict(
        api_key=_runtime.get("api_key") or os.getenv("DEEPSEEK_API_KEY"),
        base_url=_runtime.get("base_url") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=_runtime.get("model") or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=temperature,
    )
    if _stream_callbacks:
        kwargs["streaming"] = True
        kwargs["callbacks"] = list(_stream_callbacks)
    return ChatOpenAI(**kwargs)
