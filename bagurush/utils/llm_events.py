"""
LLM 调用事件追踪 —— 用于实时推送 LLM 数据流到前端。

通过 LangChain BaseCallbackHandler 捕获 LLM 调用/token/工具事件，
存入线程安全的队列，由 SSE 端点推送给前端。
"""

import queue
import time
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import BaseCallbackHandler


class EventStore:
    """线程安全的事件队列，一个 stream_id 对应一个实例。"""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self.closed = False

    def push(self, etype: str, data: str):
        self._queue.put({"type": etype, "data": data, "ts": time.time()})

    def drain(self) -> List[dict]:
        events = []
        try:
            while True:
                events.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        return events

    def close(self):
        self.closed = True
        self.push("done", "")


# 全局 event store 注册表
_stores: Dict[str, EventStore] = {}


def get_store(stream_id: str) -> EventStore:
    if stream_id not in _stores:
        _stores[stream_id] = EventStore()
    return _stores[stream_id]


def remove_store(stream_id: str):
    _stores.pop(stream_id, None)


class LLMStreamHandler(BaseCallbackHandler):
    """LangChain 回调，将 LLM/Tool 事件推送到 EventStore。"""

    def __init__(self, store: EventStore):
        self.store = store

    # ---- Chat model 事件 ----
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List, **kwargs: Any
    ):
        model = (
            kwargs.get("invocation_params", {}).get("model_name")
            or kwargs.get("invocation_params", {}).get("model")
            or serialized.get("kwargs", {}).get("model")
            or "LLM"
        )
        self.store.push("llm_start", str(model))

    def on_llm_new_token(self, token: str, **kwargs: Any):
        self.store.push("token", token)

    def on_llm_end(self, response: Any, **kwargs: Any):
        self.store.push("llm_end", "")

    def on_llm_error(self, error: BaseException, **kwargs: Any):
        self.store.push("llm_error", str(error)[:120])

    # ---- Tool 事件 ----
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ):
        name = serialized.get("name", "tool")
        self.store.push("tool_start", name)

    def on_tool_end(self, output: Any, **kwargs: Any):
        self.store.push("tool_end", "")
