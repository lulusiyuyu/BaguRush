"""
Replanner 模块测试。

测试策略：
  - 所有 LLM 调用 mock 掉，不需要真实 API Key
  - 覆盖正常重规划路径、降级兜底路径、router 联动、evaluator 写入

运行：
  cd bagurush
  /mnt/d/ForWSL/env/bagurush/bin/python -m pytest tests/test_replanner.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ============================================================ #
#  Replanner 节点测试
# ============================================================ #

class TestReplannerNode:
    """测试 replanner_node 的 plan 合并、降级和清空逻辑。"""

    BASE_STATE = {
        "interview_plan": [
            {"topic": "Python基础", "difficulty": "medium"},
            {"topic": "系统设计", "difficulty": "hard"},
            {"topic": "算法", "difficulty": "medium"},
        ],
        "current_topic_index": 1,          # 已完成 index 0，当前从 index 1 开始
        "new_findings": ["Kafka: 候选人在项目中用过 Kafka 做消息队列"],
        "completed_topics": ["Python基础"],
        "candidate_profile": {"dimensions": {"algorithm": {"score": 6.0}}},
    }

    def test_replanner_merges_plan(self):
        """正常重规划：前半段不变，后半段被 LLM 新 plan 替换。"""
        from agents.replanner import replanner_node

        new_remaining = [
            {"topic": "Kafka消息队列", "difficulty": "medium"},
            {"topic": "系统设计", "difficulty": "hard"},
        ]
        mock_response = MagicMock()
        mock_response.content = json.dumps(new_remaining)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("agents.replanner.get_llm", return_value=mock_llm):
            result = replanner_node(self.BASE_STATE)

        merged = result["interview_plan"]
        # 前半段（index 0）保持不变
        assert merged[0]["topic"] == "Python基础"
        # 后半段被替换
        assert merged[1]["topic"] == "Kafka消息队列"
        assert merged[2]["topic"] == "系统设计"
        assert len(merged) == 3

    def test_replanner_clears_new_findings(self):
        """正常重规划后 new_findings 必须清空（防止死循环）。"""
        from agents.replanner import replanner_node

        mock_response = MagicMock()
        mock_response.content = json.dumps([{"topic": "Kafka", "difficulty": "medium"}])

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("agents.replanner.get_llm", return_value=mock_llm):
            result = replanner_node(self.BASE_STATE)

        assert result["new_findings"] == []

    def test_replanner_fallback_on_llm_failure(self):
        """LLM 调用失败时：plan 不变，只清空 new_findings（降级兜底）。"""
        from agents.replanner import replanner_node

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")

        with patch("agents.replanner.get_llm", return_value=mock_llm):
            result = replanner_node(self.BASE_STATE)

        # plan 不变
        assert result["interview_plan"] == self.BASE_STATE["interview_plan"]
        # new_findings 清空（防死循环）
        assert result["new_findings"] == []

    def test_replanner_fallback_on_invalid_json(self):
        """LLM 返回非法 JSON 时：plan 不变，new_findings 清空。"""
        from agents.replanner import replanner_node

        mock_response = MagicMock()
        mock_response.content = "这不是有效的JSON数组"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("agents.replanner.get_llm", return_value=mock_llm):
            result = replanner_node(self.BASE_STATE)

        assert result["interview_plan"] == self.BASE_STATE["interview_plan"]
        assert result["new_findings"] == []

    def test_replanner_fallback_on_invalid_plan_format(self):
        """LLM 返回格式非法的 plan（不是 list 或缺少 topic 字段）时降级。"""
        from agents.replanner import replanner_node

        mock_response = MagicMock()
        # 返回对象而非数组
        mock_response.content = json.dumps({"invalid": "format"})

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("agents.replanner.get_llm", return_value=mock_llm):
            result = replanner_node(self.BASE_STATE)

        assert result["interview_plan"] == self.BASE_STATE["interview_plan"]
        assert result["new_findings"] == []

    def test_replanner_at_start_index(self):
        """current_topic_index=0 时：done_plan 为空，整个 plan 被替换。"""
        from agents.replanner import replanner_node

        state = {**self.BASE_STATE, "current_topic_index": 0, "completed_topics": []}
        new_plan = [{"topic": "NewTopic", "difficulty": "easy"}]

        mock_response = MagicMock()
        mock_response.content = json.dumps(new_plan)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("agents.replanner.get_llm", return_value=mock_llm):
            result = replanner_node(state)

        assert result["interview_plan"] == new_plan


# ============================================================ #
#  Router 联动测试（fallback 路径下 new_findings → replan）
# ============================================================ #

class TestRouterReplanAction:
    """验证 router fallback 在 new_findings 非空时触发 replan。"""

    def _run_fallback(self, state):
        """强制走 fallback 路径（mock _llm_router 抛异常）。"""
        from agents.router import router_node
        with patch("agents.router._llm_router", side_effect=RuntimeError("mock")):
            return router_node(state)

    def test_fallback_triggers_replan_when_new_findings(self):
        """new_findings 不为空时，fallback router 应返回 replan。"""
        state = {
            "total_questions_asked": 2, "max_questions": 8,
            "follow_up_count": 0, "max_follow_ups": 2,
            "current_topic_index": 0,
            "interview_plan": [{"topic": "A"}, {"topic": "B"}],
            "current_evaluation": {"overall_score": 7.0},
            "new_findings": ["Kafka: 候选人提到在项目中使用"],
        }
        result = self._run_fallback(state)
        assert result["next_action"] == "replan"

    def test_fallback_no_replan_when_findings_empty(self):
        """new_findings 为空时，不应触发 replan（走正常路由逻辑）。"""
        state = {
            "total_questions_asked": 2, "max_questions": 8,
            "follow_up_count": 0, "max_follow_ups": 2,
            "current_topic_index": 0,
            "interview_plan": [{"topic": "A"}, {"topic": "B"}],
            "current_evaluation": {"overall_score": 7.0},
            "new_findings": [],
        }
        result = self._run_fallback(state)
        assert result["next_action"] != "replan"

    def test_replan_clears_follow_up_count(self):
        """replan action 的状态更新应将 follow_up_count 重置为 0。"""
        state = {
            "total_questions_asked": 3, "max_questions": 8,
            "follow_up_count": 1, "max_follow_ups": 2,
            "current_topic_index": 0,
            "interview_plan": [{"topic": "A"}, {"topic": "B"}],
            "current_evaluation": {"overall_score": 7.0},
            "new_findings": ["Redis: 候选人提到用 Redis 做缓存"],
        }
        result = self._run_fallback(state)
        assert result["next_action"] == "replan"
        assert result["follow_up_count"] == 0


# ============================================================ #
#  Evaluator 写入 new_findings 测试
# ============================================================ #

class TestEvaluatorNewFindings:
    """验证 evaluator_node 能正确从 evaluate_answer 结果中提取并写入 new_findings。"""

    BASE_STATE = {
        "current_question": "请解释 Python GIL",
        "messages": [],
        "interview_plan": [{"topic": "Python基础"}, {"topic": "系统设计"}],
        "current_topic_index": 0,
        "all_evaluations": [],
        "total_questions_asked": 0,
        "candidate_profile": None,
        "new_findings": [],
    }

    def _make_eval_result(self, new_mention=None):
        result = {
            "completeness": 7, "accuracy": 7, "depth": 6, "expression": 7,
            "overall_score": 6.75,
            "feedback": "回答基本正确",
            "follow_up_suggestion": "可以追问 GIL 的影响",
            "profile_update": {"dimension": "fundamentals", "score_delta": 0.5, "evidence": "能解释 GIL"},
        }
        result["new_mention"] = new_mention
        return json.dumps(result)

    def test_new_mention_appended_to_new_findings(self):
        """候选人提到 plan 外的技术时，evaluator 应写入 new_findings。"""
        from agents.evaluator import evaluator_node

        eval_str = self._make_eval_result(
            new_mention={"skill": "Kafka", "context": "候选人在项目中用过 Kafka"}
        )

        with patch("agents.evaluator._get_latest_answer", return_value="我了解 GIL，另外项目里用过 Kafka"), \
             patch("agents.evaluator.search_tech_knowledge", new=MagicMock(invoke=MagicMock(return_value="参考内容"))), \
             patch("agents.evaluator.evaluate_answer", new=MagicMock(invoke=MagicMock(return_value=eval_str))):
            result = evaluator_node(self.BASE_STATE)

        assert any("Kafka" in f for f in result["new_findings"])

    def test_no_new_mention_keeps_findings_empty(self):
        """new_mention 为 null 时，new_findings 不应新增条目。"""
        from agents.evaluator import evaluator_node

        eval_str = self._make_eval_result(new_mention=None)

        with patch("agents.evaluator._get_latest_answer", return_value="GIL 是全局锁"), \
             patch("agents.evaluator.search_tech_knowledge", new=MagicMock(invoke=MagicMock(return_value="参考"))), \
             patch("agents.evaluator.evaluate_answer", new=MagicMock(invoke=MagicMock(return_value=eval_str))):
            result = evaluator_node(self.BASE_STATE)

        assert result["new_findings"] == []

    def test_duplicate_skill_not_appended(self):
        """已记录过的技术不重复追加到 new_findings。"""
        from agents.evaluator import evaluator_node

        state = {**self.BASE_STATE, "new_findings": ["Kafka: 之前已记录"]}
        eval_str = self._make_eval_result(
            new_mention={"skill": "Kafka", "context": "再次提到 Kafka"}
        )

        with patch("agents.evaluator._get_latest_answer", return_value="Kafka"), \
             patch("agents.evaluator.search_tech_knowledge", new=MagicMock(invoke=MagicMock(return_value="参考"))), \
             patch("agents.evaluator.evaluate_answer", new=MagicMock(invoke=MagicMock(return_value=eval_str))):
            result = evaluator_node(state)

        kafka_count = sum(1 for f in result["new_findings"] if "Kafka" in f)
        assert kafka_count == 1  # 不重复

    def test_skill_already_in_plan_not_appended(self):
        """技术点已在 interview_plan 中存在时，不追加到 new_findings。"""
        from agents.evaluator import evaluator_node

        # plan 里已包含 Python基础
        eval_str = self._make_eval_result(
            new_mention={"skill": "Python基础", "context": "候选人提到 Python"}
        )

        with patch("agents.evaluator._get_latest_answer", return_value="Python 很好用"), \
             patch("agents.evaluator.search_tech_knowledge", new=MagicMock(invoke=MagicMock(return_value="参考"))), \
             patch("agents.evaluator.evaluate_answer", new=MagicMock(invoke=MagicMock(return_value=eval_str))):
            result = evaluator_node(self.BASE_STATE)

        assert result["new_findings"] == []


# ============================================================ #
#  graph.py 节点注册测试
# ============================================================ #

class TestGraphReplanNode:
    """验证 replanner 节点已正确注册到 graph 中。"""

    def test_replanner_in_graph_nodes(self):
        from agents.graph import build_interview_graph
        g = build_interview_graph()
        assert "replanner" in g.nodes
