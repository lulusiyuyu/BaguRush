"""TokenTracker 单元测试。"""

from unittest.mock import MagicMock

from utils.token_tracker import TokenTracker, get_tracker, remove_tracker


def _make_llm_result(prompt=10, completion=20, total=30):
    """构造模拟的 LLMResult 对象。"""
    result = MagicMock()
    result.llm_output = {
        "token_usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }
    }
    return result


class TestTokenTracker:
    def test_initial_state(self):
        t = TokenTracker()
        s = t.get_summary()
        assert s["prompt_tokens"] == 0
        assert s["completion_tokens"] == 0
        assert s["total_tokens"] == 0
        assert s["llm_calls"] == 0

    def test_accumulate(self):
        t = TokenTracker()
        t.on_llm_end(_make_llm_result(10, 20, 30))
        t.on_llm_end(_make_llm_result(5, 15, 20))
        s = t.get_summary()
        assert s["prompt_tokens"] == 15
        assert s["completion_tokens"] == 35
        assert s["total_tokens"] == 50
        assert s["llm_calls"] == 2

    def test_reset(self):
        t = TokenTracker()
        t.on_llm_end(_make_llm_result(10, 20, 30))
        t.reset()
        s = t.get_summary()
        assert s["total_tokens"] == 0
        assert s["llm_calls"] == 0

    def test_no_usage_info(self):
        """llm_output 无 token_usage 时应安全跳过。"""
        t = TokenTracker()
        result = MagicMock()
        result.llm_output = {}
        t.on_llm_end(result)
        assert t.get_summary()["llm_calls"] == 0

    def test_none_llm_output(self):
        """llm_output 为 None 时应安全跳过。"""
        t = TokenTracker()
        result = MagicMock()
        result.llm_output = None
        t.on_llm_end(result)
        assert t.get_summary()["llm_calls"] == 0


class TestTrackerRegistry:
    def test_get_and_remove(self):
        sid = "test-session-001"
        tracker = get_tracker(sid)
        tracker.on_llm_end(_make_llm_result(100, 200, 300))
        summary = remove_tracker(sid)
        assert summary["total_tokens"] == 300
        # 移除后再次 remove 返回 None
        assert remove_tracker(sid) is None

    def test_get_returns_same_instance(self):
        sid = "test-session-002"
        t1 = get_tracker(sid)
        t2 = get_tracker(sid)
        assert t1 is t2
        remove_tracker(sid)
