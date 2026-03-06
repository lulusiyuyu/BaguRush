"""
Reporter Agent 节点。

职责：
  1. 汇总 all_evaluations 中的所有评估数据
  2. 计算各维度平均分和综合分
  3. 调用 LLM 生成 Markdown 格式的完整面试报告
  4. 返回 final_report 和 interview_status = "completed"
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 项目根目录加入 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.state import InterviewState
from prompts.reporter_prompt import REPORTER_SYSTEM_PROMPT, REPORTER_USER_TEMPLATE

load_dotenv()


def _get_llm() -> ChatOpenAI:
    from utils.llm_config import get_llm
    return get_llm(temperature=0.3)


def _compute_averages(evaluations: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算各评估维度的平均分。"""
    if not evaluations:
        return {"completeness": 0, "accuracy": 0, "depth": 0, "expression": 0, "overall": 0}

    dims = ["completeness", "accuracy", "depth", "expression"]
    avgs: Dict[str, float] = {}
    for dim in dims:
        scores = [e.get(dim, 0) for e in evaluations if isinstance(e.get(dim), (int, float))]
        avgs[dim] = round(sum(scores) / len(scores), 2) if scores else 0.0

    overall_scores = [e.get("overall_score", 0) for e in evaluations if isinstance(e.get("overall_score"), (int, float))]
    avgs["overall"] = round(sum(overall_scores) / len(overall_scores), 2) if overall_scores else 0.0

    return avgs


def _format_evaluations_text(evaluations: List[Dict[str, Any]]) -> str:
    """将评估记录格式化为供 LLM 阅读的文本。"""
    if not evaluations:
        return "（无评估记录）"

    parts = []
    for i, ev in enumerate(evaluations, 1):
        topic = ev.get("topic", "未知话题")
        question = ev.get("question", "")[:100]
        answer = ev.get("answer", "")[:120]
        score = ev.get("overall_score", 0)
        feedback = ev.get("feedback", "")

        parts.append(
            f"### 第 {i} 题（{topic}）\n"
            f"**问题**：{question}\n"
            f"**回答摘要**：{answer}{'...' if len(ev.get('answer','')) > 120 else ''}\n"
            f"**综合得分**：{score:.1f}/10\n"
            f"**评价**：{feedback}\n"
        )
    return "\n".join(parts)


def _format_interview_plan_text(interview_plan: List[Dict[str, Any]]) -> str:
    """将面试大纲格式化为文本。"""
    if not interview_plan:
        return "（无大纲）"
    lines = []
    for i, topic in enumerate(interview_plan, 1):
        lines.append(f"{i}. **{topic.get('topic', '')}**（权重 {int(topic.get('weight', 0) * 100)}%）：{topic.get('description', '')}")
    return "\n".join(lines)


def reporter_node(state: InterviewState) -> Dict[str, Any]:
    """
    Reporter Agent 节点函数。

    生成面试报告，返回：
      - final_report    : Markdown 格式报告字符串
      - interview_status: "completed"
      - messages        : 追加报告消息
    """
    all_evaluations = state.get("all_evaluations") or []
    interview_plan = state.get("interview_plan") or []
    candidate_name = state.get("candidate_name", "候选人")
    job_role = state.get("job_role", "")
    total_questions = state.get("total_questions_asked", len(all_evaluations))

    print(f"\n[Reporter] 开始生成面试报告 | 候选人: {candidate_name} | 总题数: {total_questions}")

    # 计算平均分
    avgs = _compute_averages(all_evaluations)
    print(f"[Reporter] 综合得分: {avgs['overall']:.2f}/10")

    # 格式化评估文本
    evaluations_text = _format_evaluations_text(all_evaluations)
    interview_plan_text = _format_interview_plan_text(interview_plan)

    # 调用 LLM 生成报告
    llm = _get_llm()

    user_content = REPORTER_USER_TEMPLATE.format(
        candidate_name=candidate_name,
        job_role=job_role,
        topic_count=len(interview_plan),
        total_questions=total_questions,
        avg_completeness=avgs["completeness"],
        avg_accuracy=avgs["accuracy"],
        avg_depth=avgs["depth"],
        avg_expression=avgs["expression"],
        avg_overall=avgs["overall"],
        evaluations_text=evaluations_text,
        interview_plan_text=interview_plan_text,
    )

    messages = [
        SystemMessage(content=REPORTER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        response = llm.invoke(messages)
        final_report = response.content.strip()
        print(f"[Reporter] 报告生成完成（{len(final_report)} 字符）")
    except Exception as e:
        print(f"[Reporter] ❌ 报告生成失败: {e}，使用简化报告")
        final_report = _generate_fallback_report(candidate_name, job_role, avgs, all_evaluations)

    return {
        "final_report": final_report,
        "interview_status": "completed",
        "messages": [AIMessage(content=f"面试已结束，报告已生成。综合得分：{avgs['overall']:.2f}/10", name="reporter")],
    }


def _generate_fallback_report(
    candidate_name: str,
    job_role: str,
    avgs: Dict[str, float],
    evaluations: List[Dict[str, Any]],
) -> str:
    """LLM 调用失败时的兜底简化报告。"""
    overall = avgs["overall"]
    grade = "A" if overall >= 8.5 else "B" if overall >= 7.0 else "C" if overall >= 5.5 else "D"

    lines = [
        f"# 🎯 面试评估报告",
        f"",
        f"## 📊 面试概览",
        f"| 项目 | 内容 |",
        f"|------|------|",
        f"| 候选人 | {candidate_name} |",
        f"| 目标岗位 | {job_role} |",
        f"| 题目数 | {len(evaluations)} |",
        f"| 综合得分 | {overall:.2f} / 10 |",
        f"| 评级 | {grade} |",
        f"",
        f"## 📈 分数总结",
        f"| 维度 | 得分 |",
        f"|------|------|",
        f"| 完整性 | {avgs['completeness']:.1f} |",
        f"| 准确性 | {avgs['accuracy']:.1f} |",
        f"| 深度 | {avgs['depth']:.1f} |",
        f"| 表达 | {avgs['expression']:.1f} |",
        f"| **综合** | **{overall:.2f}** |",
    ]

    if evaluations:
        lines += ["", "## 📝 逐题回顾"]
        for i, ev in enumerate(evaluations, 1):
            lines.append(f"{i}. **{ev.get('topic', '')}** — 综合分 {ev.get('overall_score', 0):.1f} | {ev.get('feedback', '')[:80]}")

    lines += ["", "---", "*本报告由 BaguRush AI 面试系统自动生成*"]
    return "\n".join(lines)
