"""
Token 消耗跟踪器。

基于 LangChain BaseCallbackHandler，自动从 LLM 响应中累积 token 用量。
支持按 stream_id 隔离（多面试并发），提供 get_summary() 汇总接口。
"""

import threading
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class TokenTracker(BaseCallbackHandler):
    """累积单次面试会话内的 token 消耗。线程安全。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.llm_calls: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """从 LLMResult.llm_output 提取 token_usage 并累加。"""
        usage = (response.llm_output or {}).get("token_usage", {})
        if not usage:
            return
        with self._lock:
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
            self.llm_calls += 1

    def get_summary(self) -> Dict[str, int]:
        """返回当前累积的 token 消耗快照。"""
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "llm_calls": self.llm_calls,
            }

    def reset(self) -> None:
        """重置计数器。"""
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.llm_calls = 0


# ---- 全局 tracker 注册表（按 stream_id 隔离）----
_trackers: Dict[str, TokenTracker] = {}
_registry_lock = threading.Lock()


def get_tracker(stream_id: str) -> TokenTracker:
    """获取或创建指定 stream_id 的 TokenTracker。"""
    with _registry_lock:
        if stream_id not in _trackers:
            _trackers[stream_id] = TokenTracker()
        return _trackers[stream_id]


def remove_tracker(stream_id: str) -> Optional[Dict[str, int]]:
    """移除并返回指定 stream_id 的最终 token 消耗。"""
    with _registry_lock:
        tracker = _trackers.pop(stream_id, None)
    if tracker:
        return tracker.get_summary()
    return None
