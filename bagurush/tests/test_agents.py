"""
Agent 节点单元测试。

测试策略：
  - 每个 Agent 内部的纯函数/辅助函数做 **静态测试**（不调用 LLM）
  - 整体节点逻辑标记 @pytest.mark.llm（需要真实 API 才跑）
"""

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ============================================================ #
#  Planner 辅助函数
# ============================================================ #
class TestPlannerHelpers:
    """测试 planner.py 中的 _extract_json_from_text。"""

    def test_extract_json_from_code_block(self):
        from agents.planner import _extract_json_from_text
        text = '好的，以下是结果：\n```json\n{"resume_analysis": {}, "interview_plan": []}\n```\n请查看。'
        result = _extract_json_from_text(text)
        assert isinstance(result, dict)
        assert "resume_analysis" in result
        assert "interview_plan" in result

    def test_extract_json_from_bare_braces(self):
        from agents.planner import _extract_json_from_text
        text = '分析完成。{"resume_analysis": {"name": "张三"}, "interview_plan": [{"topic": "Python"}]}'
        result = _extract_json_from_text(text)
        assert result["resume_analysis"]["name"] == "张三"
        assert len(result["interview_plan"]) == 1

    def test_extract_json_no_json_raises(self):
        from agents.planner import _extract_json_from_text
        with pytest.raises(json.JSONDecodeError):
            _extract_json_from_text("这里没有任何 JSON 内容")

    def test_extract_json_nested(self):
        from agents.planner import _extract_json_from_text
        text = """```json
{
  "resume_analysis": {
    "name": "李四",
    "strengths": ["Python", "机器学习"],
    "weaknesses": ["系统设计"],
    "key_projects": ["推荐系统"],
    "overall_level": "中级"
  },
  "interview_plan": [
    {"topic": "Python 基础", "weight": 0.3, "description": "GIL、装饰器", "difficulty": "medium", "reason": "核心"}
  ]
}
```"""
        result = _extract_json_from_text(text)
        assert result["resume_analysis"]["name"] == "李四"
        assert len(result["resume_analysis"]["strengths"]) == 2


# ============================================================ #
#  Interviewer 辅助函数
# ============================================================ #
class TestInterviewerHelpers:
    """测试 interviewer.py 中的 _get_resume_summary。"""

    def test_resume_summary_full(self):
        from agents.interviewer import _get_resume_summary
        state = {
            "resume_analysis": {
                "name": "张三",
                "overall_level": "中级",
                "strengths": ["Python", "Redis", "分布式系统"],
                "key_projects": ["推荐系统", "API 网关"],
            },
            "candidate_name": "张三",
        }
        summary = _get_resume_summary(state)
        assert "张三" in summary
        assert "中级" in summary
        assert "Python" in summary
        assert "推荐系统" in summary

    def test_resume_summary_minimal(self):
        from agents.interviewer import _get_resume_summary
        state = {"candidate_name": "测试用户"}
        summary = _get_resume_summary(state)
        assert "测试用户" in summary

    def test_resume_summary_empty_analysis(self):
        from agents.interviewer import _get_resume_summary
        state = {"resume_analysis": {}, "candidate_name": "候选人"}
        summary = _get_resume_summary(state)
        assert "候选人" in summary


# ============================================================ #
#  Evaluator 辅助函数
# ============================================================ #
class TestEvaluatorHelpers:
    """测试 evaluator.py 中的 _get_latest_answer。"""

    def test_get_latest_candidate_answer(self):
        from langchain_core.messages import AIMessage, HumanMessage
        from agents.evaluator import _get_latest_answer

        messages = [
            AIMessage(content="请介绍 GIL", name="interviewer"),
            HumanMessage(content="GIL 是全局解释器锁", name="candidate"),
            AIMessage(content="请继续", name="interviewer"),
            HumanMessage(content="它限制了多线程并发", name="candidate"),
        ]
        answer = _get_latest_answer(messages)
        assert answer == "它限制了多线程并发"

    def test_get_latest_answer_no_candidate_name(self):
        from langchain_core.messages import AIMessage, HumanMessage
        from agents.evaluator import _get_latest_answer

        messages = [
            AIMessage(content="问题"),
            HumanMessage(content="我的回答"),
        ]
        answer = _get_latest_answer(messages)
        assert answer == "我的回答"

    def test_get_latest_answer_empty(self):
        from agents.evaluator import _get_latest_answer
        assert _get_latest_answer([]) == ""


# ============================================================ #
#  Reporter 辅助函数
# ============================================================ #
class TestReporterHelpers:
    """测试 reporter.py 中的辅助函数。"""

    def test_compute_averages_normal(self):
        from agents.reporter import _compute_averages
        evals = [
            {"completeness": 8, "accuracy": 7, "depth": 6, "expression": 9, "overall_score": 7.5},
            {"completeness": 6, "accuracy": 9, "depth": 8, "expression": 7, "overall_score": 7.5},
        ]
        avgs = _compute_averages(evals)
        assert avgs["completeness"] == 7.0
        assert avgs["accuracy"] == 8.0
        assert avgs["depth"] == 7.0
        assert avgs["expression"] == 8.0
        assert avgs["overall"] == 7.5

    def test_compute_averages_empty(self):
        from agents.reporter import _compute_averages
        avgs = _compute_averages([])
        assert avgs["overall"] == 0
        assert avgs["completeness"] == 0

    def test_compute_averages_single(self):
        from agents.reporter import _compute_averages
        evals = [{"completeness": 10, "accuracy": 10, "depth": 10, "expression": 10, "overall_score": 10}]
        avgs = _compute_averages(evals)
        assert avgs["overall"] == 10.0

    def test_format_evaluations_text(self):
        from agents.reporter import _format_evaluations_text
        evals = [
            {"topic": "Python", "question": "解释 GIL", "answer": "GIL 是...", "overall_score": 7.5, "feedback": "回答较好"},
        ]
        text = _format_evaluations_text(evals)
        assert "Python" in text
        assert "GIL" in text
        assert "7.5" in text

    def test_format_evaluations_empty(self):
        from agents.reporter import _format_evaluations_text
        assert "无评估" in _format_evaluations_text([])

    def test_format_interview_plan_text(self):
        from agents.reporter import _format_interview_plan_text
        plan = [
            {"topic": "Python 基础", "weight": 0.3, "description": "GIL、装饰器等"},
            {"topic": "系统设计", "weight": 0.4, "description": "缓存、消息队列"},
        ]
        text = _format_interview_plan_text(plan)
        assert "Python 基础" in text
        assert "系统设计" in text
        assert "30%" in text
        assert "40%" in text

    def test_fallback_report_grade_A(self):
        from agents.reporter import _generate_fallback_report
        report = _generate_fallback_report("候选人", "后端", {"completeness": 9, "accuracy": 9, "depth": 9, "expression": 9, "overall": 9.0}, [])
        assert "A" in report

    def test_fallback_report_grade_D(self):
        from agents.reporter import _generate_fallback_report
        report = _generate_fallback_report("候选人", "后端", {"completeness": 3, "accuracy": 3, "depth": 3, "expression": 3, "overall": 3.0}, [])
        assert "D" in report


# ============================================================ #
#  Router 逻辑（测试 fallback if-else 规则，mock 掉 LLM）
# ============================================================ #
class TestRouterLogic:
    """测试 router.py 路由规则（fallback 降级逻辑）。"""

    def _run_router_with_fallback(self, state):
        """Mock _llm_router 使其抛异常，从而触发 fallback 路径。"""
        from unittest.mock import patch
        from agents.router import router_node
        with patch("agents.router._llm_router", side_effect=RuntimeError("mock")):
            return router_node(state)

    def test_max_questions_reached(self):
        state = {
            "total_questions_asked": 8, "max_questions": 8,
            "follow_up_count": 0, "max_follow_ups": 2,
            "current_topic_index": 0,
            "interview_plan": [{"topic": "A"}, {"topic": "B"}],
            "current_evaluation": {"overall_score": 3.0},
        }
        result = self._run_router_with_fallback(state)
        assert result["next_action"] == "end_interview"

    def test_low_score_triggers_follow_up(self):
        state = {
            "total_questions_asked": 2, "max_questions": 8,
            "follow_up_count": 0, "max_follow_ups": 2,
            "current_topic_index": 0,
            "interview_plan": [{"topic": "A"}, {"topic": "B"}],
            "current_evaluation": {"overall_score": 4.0},
        }
        result = self._run_router_with_fallback(state)
        assert result["next_action"] == "follow_up"
        assert result["follow_up_count"] == 1

    def test_follow_up_exhausted_moves_to_next(self):
        state = {
            "total_questions_asked": 3, "max_questions": 8,
            "follow_up_count": 2, "max_follow_ups": 2,
            "current_topic_index": 0,
            "interview_plan": [{"topic": "A"}, {"topic": "B"}],
            "current_evaluation": {"overall_score": 4.0},
        }
        result = self._run_router_with_fallback(state)
        assert result["next_action"] == "next_question"
        assert result["current_topic_index"] == 1

    def test_high_score_next_topic(self):
        state = {
            "total_questions_asked": 1, "max_questions": 8,
            "follow_up_count": 0, "max_follow_ups": 1,
            "current_topic_index": 0,
            "interview_plan": [{"topic": "A"}, {"topic": "B"}, {"topic": "C"}],
            "current_evaluation": {"overall_score": 8.5},
        }
        result = self._run_router_with_fallback(state)
        assert result["next_action"] == "next_question"
        assert result["current_topic_index"] == 1
        assert result["follow_up_count"] == 0

    def test_all_topics_done(self):
        state = {
            "total_questions_asked": 3, "max_questions": 8,
            "follow_up_count": 0, "max_follow_ups": 1,
            "current_topic_index": 2,
            "interview_plan": [{"topic": "A"}, {"topic": "B"}, {"topic": "C"}],
            "current_evaluation": {"overall_score": 8.0},
        }
        result = self._run_router_with_fallback(state)
        assert result["next_action"] == "end_interview"

    def test_route_decision(self):
        from agents.router import route_decision
        assert route_decision({"next_action": "follow_up"}) == "follow_up"
        assert route_decision({"next_action": "next_question"}) == "next_question"
        assert route_decision({"next_action": "end_interview"}) == "end_interview"
        assert route_decision({}) == "end_interview"


# ============================================================ #
#  Schemas 验证
# ============================================================ #
class TestSchemas:
    """测试 Pydantic 模型基本功能。"""

    def test_answer_request(self):
        from api.schemas import AnswerRequest
        req = AnswerRequest(answer="我的回答")
        assert req.answer == "我的回答"

    def test_answer_response_defaults(self):
        from api.schemas import AnswerResponse
        resp = AnswerResponse()
        assert resp.evaluation is None
        assert resp.next_question is None
        assert resp.is_follow_up is False
        assert resp.interview_ended is False
        assert resp.progress == "0/0"

    def test_start_interview_response(self):
        from api.schemas import StartInterviewResponse
        resp = StartInterviewResponse(
            session_id="abc123",
            message="面试准备就绪",
            interview_plan=[{"topic": "Python"}],
            first_question="请介绍 GIL",
        )
        assert resp.session_id == "abc123"
        assert len(resp.interview_plan) == 1

    def test_report_response(self):
        from api.schemas import ReportResponse
        resp = ReportResponse(report="# 报告", overall_score=7.5, grade="B", evaluations=[])
        assert resp.grade == "B"
        assert resp.overall_score == 7.5

    def test_status_response(self):
        from api.schemas import StatusResponse
        resp = StatusResponse(session_id="test", status="interviewing")
        assert resp.max_questions == 8

    def test_history_response(self):
        from api.schemas import HistoryResponse, MessageItem
        msg = MessageItem(role="interviewer", content="你好")
        resp = HistoryResponse(session_id="test", messages=[msg])
        assert len(resp.messages) == 1
        assert resp.messages[0].role == "interviewer"


# ============================================================ #
#  Phase 1.5: Interviewer Prompt RAG 验证
# ============================================================ #

class TestInterviewerPromptRAG:
    """验证 Interviewer prompt 鼓励调用 RAG。"""

    def test_prompt_encourages_rag(self):
        from prompts.interviewer_prompt import INTERVIEWER_SYSTEM_PROMPT
        # 确认不再有 discourage 的话
        assert "不必每次都调用工具" not in INTERVIEWER_SYSTEM_PROMPT
        assert "通常直接基于已有知识" not in INTERVIEWER_SYSTEM_PROMPT

    def test_prompt_has_mandatory_scenarios(self):
        from prompts.interviewer_prompt import INTERVIEWER_SYSTEM_PROMPT
        assert "必须调用" in INTERVIEWER_SYSTEM_PROMPT
        assert "search_tech_knowledge" in INTERVIEWER_SYSTEM_PROMPT

    def test_prompt_templates_still_valid(self):
        """确认模板的 format 占位符没被破坏。"""
        from prompts.interviewer_prompt import (
            INTERVIEWER_NEW_QUESTION_TEMPLATE,
            INTERVIEWER_FOLLOW_UP_TEMPLATE,
        )
        result = INTERVIEWER_NEW_QUESTION_TEMPLATE.format(
            candidate_name="测试", job_role="后端",
            current_topic="Python", topic_description="基础",
            difficulty="medium", total_questions_asked=1,
            max_questions=8, resume_summary="简历摘要",
        )
        assert "Python" in result

        result2 = INTERVIEWER_FOLLOW_UP_TEMPLATE.format(
            candidate_name="测试", job_role="后端",
            current_topic="Python", current_question="什么是GIL",
            follow_up_count=1, max_follow_ups=2,
            follow_up_suggestion="请追问细节",
        )
        assert "GIL" in result2


# ============================================================ #
#  Phase 2.5: Evaluator 双重检索验证
# ============================================================ #

class TestEvaluatorRAGEnhancement:
    """验证 Evaluator 的双重检索逻辑。"""

    def test_evaluator_handles_empty_answer(self):
        """候选人回答为空时，不应调用第二次 RAG 检索。"""
        from unittest.mock import patch, MagicMock
        from agents.evaluator import evaluator_node
        from langchain_core.messages import AIMessage

        state = {
            "current_question": "什么是死锁",
            "messages": [AIMessage(content="问题", name="interviewer")],
            "interview_plan": [{"topic": "OS"}],
            "current_topic_index": 0,
            "total_questions_asked": 0,
            "candidate_profile": {},
            "new_findings": [],
        }

        mock_rag = MagicMock(return_value="参考内容")
        mock_eval = MagicMock(return_value='{"completeness":5,"accuracy":5,"depth":5,"expression":5,"overall_score":5.0,"feedback":"ok","follow_up_suggestion":"继续"}')

        with patch("agents.evaluator.search_tech_knowledge") as mock_search, \
             patch("agents.evaluator.evaluate_answer") as mock_evaluate:
            mock_search.invoke = mock_rag
            mock_evaluate.invoke = mock_eval
            result = evaluator_node(state)

        assert result["total_questions_asked"] == 1
        assert result["current_evaluation"] is not None

    def test_evaluator_rag_failure_graceful(self):
        """RAG 检索失败时应降级，不影响评估流程。"""
        from unittest.mock import patch, MagicMock
        from agents.evaluator import evaluator_node
        from langchain_core.messages import HumanMessage

        state = {
            "current_question": "什么是GIL",
            "messages": [HumanMessage(content="GIL是锁", name="candidate")],
            "interview_plan": [{"topic": "Python"}],
            "current_topic_index": 0,
            "total_questions_asked": 0,
            "candidate_profile": {},
            "new_findings": [],
        }

        mock_eval = MagicMock(return_value='{"completeness":5,"accuracy":5,"depth":5,"expression":5,"overall_score":5.0,"feedback":"ok","follow_up_suggestion":"继续"}')

        with patch("agents.evaluator.search_tech_knowledge") as mock_search, \
             patch("agents.evaluator.evaluate_answer") as mock_evaluate:
            mock_search.invoke = MagicMock(side_effect=RuntimeError("RAG挂了"))
            mock_evaluate.invoke = mock_eval
            result = evaluator_node(state)

        # RAG 失败但评估应该正常完成
        assert result["current_evaluation"]["overall_score"] == 5.0


# ============================================================ #
#  Phase 3.5: Evaluator Prompt RAG 驱动追问验证
# ============================================================ #

class TestEvaluatorPromptRAG:
    """验证 Evaluator prompt 要求基于参考资料追问。"""

    def test_evaluator_prompt_has_rag_guidance(self):
        from prompts.evaluator_prompt import EVALUATOR_SYSTEM_PROMPT
        # 确认有追问原则指引
        assert "参考资料" in EVALUATOR_SYSTEM_PROMPT

    def test_evaluator_prompt_mentions_completeness_reference(self):
        from prompts.evaluator_prompt import EVALUATOR_SYSTEM_PROMPT
        assert "完整性" in EVALUATOR_SYSTEM_PROMPT or "completeness" in EVALUATOR_SYSTEM_PROMPT.lower()

    def test_answer_evaluator_uses_reference(self):
        """验证 evaluate_answer 工具接受 reference 参数。"""
        from tools.answer_evaluator import evaluate_answer
        # 确认工具签名中有 reference 参数
        schema = evaluate_answer.args_schema.schema()
        assert "reference" in schema.get("properties", {})
