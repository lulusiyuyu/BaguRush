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


def _apply_profile_update(profile: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 Evaluator 返回的 profile_update 增量合并到 candidate_profile。

    update 格式: {"dimension": "algorithm", "score_delta": 0.5, "evidence": "..."}
    """
    dimensions = profile.setdefault("dimensions", {})
    dim_name = update.get("dimension", "")
    if not dim_name:
        return profile

    dim = dimensions.setdefault(dim_name, {"score": 5.0, "confidence": 0.3, "evidence": []})
    dim["score"] = max(0.0, min(10.0, dim["score"] + update.get("score_delta", 0)))
    dim["confidence"] = min(1.0, dim["confidence"] + 0.1)
    evidence_text = update.get("evidence", "")
    if evidence_text:
        dim["evidence"] = (dim.get("evidence") or [])[-4:] + [evidence_text]

    # 重新计算 weak_spots / strong_spots
    weak = [k for k, v in dimensions.items() if v.get("score", 5) < 5.0]
    strong = [k for k, v in dimensions.items() if v.get("score", 5) >= 7.5]
    profile["weak_spots"] = weak
    profile["strong_spots"] = strong

    return profile


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

    # 2. 获取参考答案（双重检索：用问题 + 用回答关键词）
    reference = ""
    try:
        # 2a. 用问题检索：找到该话题的标准知识点
        question_rag = search_tech_knowledge.invoke({"query": current_question, "k": 3})

        # 2b. 用候选人回答检索：验证候选人提到的技术细节
        answer_rag = ""
        if user_answer and len(user_answer) > 20:
            answer_rag = search_tech_knowledge.invoke({"query": user_answer[:200], "k": 2})

        # 合并参考（总计约 2000 字）
        reference_parts = []
        if question_rag:
            reference_parts.append(f"【话题知识点参考】\n{question_rag[:2500]}")
        if answer_rag:
            reference_parts.append(f"【候选人提及概念验证】\n{answer_rag[:1500]}")
        reference = "\n---\n".join(reference_parts)

        print(f"[Evaluator] RAG 参考获取成功（{len(reference)} 字符，含回答验证）")
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

    # 5. 增量更新候选人画像（profile_update 由 evaluate_answer 一并返回）
    candidate_profile = dict(state.get("candidate_profile") or {})
    profile_update = evaluation.get("profile_update")
    if profile_update and isinstance(profile_update, dict):
        candidate_profile = _apply_profile_update(candidate_profile, profile_update)

    # 6. 检测 new_mention：若候选人提及了 plan 外的技术，记录到 new_findings
    new_findings: List[str] = list(state.get("new_findings") or [])
    new_mention = evaluation.get("new_mention")
    if new_mention and isinstance(new_mention, dict):
        skill = new_mention.get("skill", "").strip()
        context = new_mention.get("context", "").strip()
        if skill:
            # 只有当 plan 里不存在该 topic 时才追加
            plan_topics = [t.get("topic", "") for t in (state.get("interview_plan") or [])]
            already_in_plan = any(skill.lower() in t.lower() for t in plan_topics)
            already_recorded = any(skill.lower() in f.lower() for f in new_findings)
            if not already_in_plan and not already_recorded:
                finding = f"{skill}: {context}" if context else skill
                new_findings.append(finding)
                print(f"[Evaluator] 新发现技术点: {finding}")

    return {
        "current_evaluation": evaluation,
        "all_evaluations": all_evaluations,
        "total_questions_asked": state.get("total_questions_asked", 0) + 1,
        "candidate_profile": candidate_profile,
        "new_findings": new_findings,
    }
