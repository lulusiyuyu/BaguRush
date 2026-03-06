"""
Evaluator Agent 节点。

职责：
  1. 从 messages 中提取候选人最新回答（HumanMessage，name="candidate"）
  2. （可选）调用 search_tech_knowledge 获取参考答案，提升评估精度
  3. 直接调用 evaluate_answer tool 进行结构化评估
  4. 更新 current_evaluation、all_evaluations、total_questions_asked
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage

# 项目根目录加入 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.state import InterviewState
from tools.answer_evaluator import evaluate_answer
from tools.knowledge_rag import search_tech_knowledge

load_dotenv()


def _get_latest_answer(messages: List[BaseMessage]) -> str:
    """
    从消息历史中提取候选人最新的回答（name="candidate" 的 HumanMessage）。
    若找不到，返回空字符串。
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and getattr(msg, "name", None) == "candidate":
            return msg.content
    # 兜底：取最后一条 HumanMessage
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def evaluator_node(state: InterviewState) -> Dict[str, Any]:
    """
    Evaluator Agent 节点函数。

    评估候选人对当前问题的回答，返回：
      - current_evaluation    : 当前题评估结果 dict
      - all_evaluations       : 追加当前评估后的历史列表
      - total_questions_asked : +1
    """
    current_question = state.get("current_question") or ""
    messages = state.get("messages") or []
    interview_plan = state.get("interview_plan") or []
    current_topic_index = state.get("current_topic_index", 0)
    current_topic = interview_plan[current_topic_index].get("topic", "") if current_topic_index < len(interview_plan) else ""

    print(f"\n[Evaluator] 评估回答 | 话题: {current_topic} | 问题: {current_question[:50]}...")

    # 1. 提取候选人最新回答
    user_answer = _get_latest_answer(messages)
    if not user_answer:
        print("[Evaluator] ⚠️ 未找到候选人回答，使用空字符串评估")
        user_answer = ""

    # 2. 获取参考答案（RAG 检索，不通过 Tool Calling，直接调用）
    reference = ""
    try:
        rag_result = search_tech_knowledge.invoke({"query": current_question, "k": 2})
        reference = rag_result[:1000]  # 截取前 1000 字作为参考
        print(f"[Evaluator] RAG 参考获取成功（{len(reference)} 字符）")
    except Exception as e:
        print(f"[Evaluator] ⚠️ RAG 检索失败，无参考答案: {e}")

    # 3. 直接调用评估工具
    try:
        eval_result_str = evaluate_answer.invoke({
            "question": current_question,
            "answer": user_answer,
            "reference": reference,
        })
        evaluation = json.loads(eval_result_str)
        print(
            f"[Evaluator] 评估完成 | 综合分: {evaluation.get('overall_score', 0):.1f}/10 | "
            f"完整性:{evaluation.get('completeness')} 准确性:{evaluation.get('accuracy')} "
            f"深度:{evaluation.get('depth')} 表达:{evaluation.get('expression')}"
        )
    except Exception as e:
        print(f"[Evaluator] ❌ 评估失败: {e}，使用默认中等分数")
        evaluation = {
            "completeness": 5, "accuracy": 5, "depth": 5, "expression": 5,
            "overall_score": 5.0,
            "feedback": "评估过程出现错误，使用默认分数",
            "follow_up_suggestion": "请重新作答",
        }

    # 4. 追加评估记录（包含问题和回答上下文）
    evaluation_record = {
        **evaluation,
        "question": current_question,
        "answer": user_answer[:200],  # 截断避免状态过大
        "topic": current_topic,
        "question_index": state.get("total_questions_asked", 0) + 1,
    }

    all_evaluations: List[Dict[str, Any]] = list(state.get("all_evaluations") or [])
    all_evaluations.append(evaluation_record)

    return {
        "current_evaluation": evaluation,
        "all_evaluations": all_evaluations,
        "total_questions_asked": state.get("total_questions_asked", 0) + 1,
    }
