"""
Router 节点 —— LLM 自主决策 + if-else 降级兜底。

LLM Router 支持 5 种 action：
  follow_up        → 追问当前话题
  next_question    → 进入下一个话题
  switch_topic     → 跳到 Router 指定的话题 index
  change_difficulty→ 调整难度后重新出题
  end_interview    → 结束面试

降级规则（兜底，LLM 调用失败时）：
  规则 1: total_questions_asked >= max_questions → "end_interview"
  规则 2: overall_score < 6.0 AND follow_up_count < max_follow_ups → "follow_up"
  规则 3: current_topic_index + 1 < len(interview_plan) → "next_question"
  规则 4: 所有话题已问完 → "end_interview"
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# 项目根目录加入 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.state import InterviewState

# 评分低于此阈值时触发追问（兜底逻辑使用）
FOLLOW_UP_THRESHOLD = 6.0

# LLM Router 合法 action 枚举
VALID_ACTIONS = {"follow_up", "next_question", "end_interview", "switch_topic", "change_difficulty", "replan"}


# ------------------------------------------------------------------ #
#  辅助函数
# ------------------------------------------------------------------ #

def _summarize_evaluations(all_evaluations: List[Dict[str, Any]], max_items: int = 5) -> str:
    """将历史评估列表压缩为简短摘要文本，避免 prompt 过长。"""
    if not all_evaluations:
        return "暂无历史评估"
    recent = all_evaluations[-max_items:]
    lines = []
    for e in recent:
        lines.append(
            f"- [{e.get('topic', '?')}] 综合分:{e.get('overall_score', '?')} "
            f"| 完整:{e.get('completeness', '?')} 准确:{e.get('accuracy', '?')} "
            f"深度:{e.get('depth', '?')} 表达:{e.get('expression', '?')}"
        )
    return "\n".join(lines)


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """从 LLM 输出中提取 JSON（支持 markdown 包裹）。"""
    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_match:
        return json.loads(code_match.group(1).strip())
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        return json.loads(brace_match.group(0))
    raise json.JSONDecodeError("无法从输出中找到 JSON", text, 0)


# ------------------------------------------------------------------ #
#  LLM Router
# ------------------------------------------------------------------ #

_ROUTER_PROMPT_TEMPLATE = """你是面试流程的决策者。根据以下信息，决定下一步动作。

【候选人画像】
{candidate_profile}

【当前题目】{current_question}

【当前评分】{current_evaluation}

【面试进度】已问 {total_asked}/{max_questions} 题

【当前难度】{difficulty}

【新发现】（候选人指出的计划外技术，可考虑重规划）
{new_findings}

【剩余 Plan】
{remaining_plan}

【历史评估摘要】
{eval_summary}

请输出 JSON（不要输出额外文字）：
{{
  "action": "follow_up" | "next_question" | "end_interview" | "switch_topic" | "change_difficulty" | "replan",
  "reason": "决策理由",
  "target_topic_index": null,
  "difficulty_adjustment": null
}}

字段说明：
- action: 必填，六选一
- reason: 必填，简述理由
- target_topic_index: 仅 switch_topic 时填写（int，目标 plan index）
- difficulty_adjustment: 仅 change_difficulty 时填写（"up" 或 "down"）

决策原则：
- 候选人某维度已答好 3 题 → 不要再问同类型
- 候选人持续低分 → 降低难度再给一次机会或切换话题
- 已问 >= max-2 → 确保覆盖关键维度
- 候选人回答暴露新弱点 → 可追问或临时插入
- 追问次数已达上限 → 不要选 follow_up
- new_findings 不为空 → 可考虑选 replan 让重规划节点优化后续计划"""


def _llm_router(state: InterviewState) -> Dict[str, Any]:
    """调用 LLM 进行自主决策，返回 action + 状态更新 dict。"""
    from utils.llm_config import get_llm

    llm = get_llm(temperature=0.3)

    interview_plan = state.get("interview_plan") or []
    current_topic_index = state.get("current_topic_index", 0)
    total_asked = state.get("total_questions_asked", 0)
    max_questions = state.get("max_questions", 8)

    prompt_text = _ROUTER_PROMPT_TEMPLATE.format(
        candidate_profile=json.dumps(state.get("candidate_profile") or {}, ensure_ascii=False, indent=2),
        current_question=state.get("current_question", ""),
        current_evaluation=json.dumps(state.get("current_evaluation") or {}, ensure_ascii=False),
        total_asked=total_asked,
        max_questions=max_questions,
        difficulty=state.get("difficulty", "medium"),
        new_findings="\n".join(f"- {f}" for f in (state.get("new_findings") or [])) or "无",
        remaining_plan=json.dumps(interview_plan[current_topic_index:], ensure_ascii=False, indent=2),
        eval_summary=_summarize_evaluations(state.get("all_evaluations") or []),
    )

    response = llm.invoke(prompt_text)
    decision = _parse_json_from_text(response.content)

    action = decision.get("action")
    if action not in VALID_ACTIONS:
        raise ValueError(f"非法 action: {action}")

    reason = decision.get("reason", "")
    print(f"[Router/LLM] action={action} | reason={reason}")

    # 根据 action 构建状态更新
    return _build_state_update(state, action, decision)


# ------------------------------------------------------------------ #
#  Fallback Router（原 if-else 逻辑）
# ------------------------------------------------------------------ #

def _fallback_router(state: InterviewState) -> Dict[str, Any]:
    """原始 if-else 规则路由（降级兜底）。"""
    total_asked = state.get("total_questions_asked", 0)
    max_questions = state.get("max_questions", 8)
    follow_up_count = state.get("follow_up_count", 0)
    max_follow_ups = state.get("max_follow_ups", 1)
    current_topic_index = state.get("current_topic_index", 0)
    interview_plan = state.get("interview_plan") or []
    current_evaluation = state.get("current_evaluation") or {}
    overall_score = current_evaluation.get("overall_score", 10.0)
    new_findings = state.get("new_findings") or []

    # 新御：new_findings 不为空 → 触发重规划（优先级最高）
    if new_findings:
        return _build_state_update(state, "replan", {})
    if total_asked >= max_questions:
        return _build_state_update(state, "end_interview", {})
    if overall_score < FOLLOW_UP_THRESHOLD and follow_up_count < max_follow_ups:
        return _build_state_update(state, "follow_up", {})
    if current_topic_index + 1 < len(interview_plan):
        return _build_state_update(state, "next_question", {})
    return _build_state_update(state, "end_interview", {})


# ------------------------------------------------------------------ #
#  公共状态更新构建
# ------------------------------------------------------------------ #

def _build_state_update(state: InterviewState, action: str, decision: Dict[str, Any]) -> Dict[str, Any]:
    """根据 action 构建对应的状态 patch dict。"""
    current_topic_index = state.get("current_topic_index", 0)
    interview_plan = state.get("interview_plan") or []
    follow_up_count = state.get("follow_up_count", 0)
    completed_topics: List[str] = list(state.get("completed_topics") or [])
    difficulty = state.get("difficulty", "medium")

    if action == "end_interview":
        # 把当前话题标记为已完成
        if current_topic_index < len(interview_plan):
            topic_name = interview_plan[current_topic_index].get("topic", "")
            if topic_name and topic_name not in completed_topics:
                completed_topics.append(topic_name)
        return {
            "next_action": "end_interview",
            "interview_status": "reporting",
            "completed_topics": completed_topics,
        }

    if action == "follow_up":
        return {
            "next_action": "follow_up",
            "follow_up_count": follow_up_count + 1,
        }

    if action == "next_question":
        # 标记当前话题完成，移至下一个
        if current_topic_index < len(interview_plan):
            topic_name = interview_plan[current_topic_index].get("topic", "")
            if topic_name and topic_name not in completed_topics:
                completed_topics.append(topic_name)
        return {
            "next_action": "next_question",
            "current_topic_index": current_topic_index + 1,
            "follow_up_count": 0,
            "completed_topics": completed_topics,
        }

    if action == "switch_topic":
        target_idx = decision.get("target_topic_index")
        if target_idx is None or not (0 <= target_idx < len(interview_plan)):
            target_idx = min(current_topic_index + 1, len(interview_plan) - 1)
        # 标记当前话题完成
        if current_topic_index < len(interview_plan):
            topic_name = interview_plan[current_topic_index].get("topic", "")
            if topic_name and topic_name not in completed_topics:
                completed_topics.append(topic_name)
        return {
            "next_action": "switch_topic",
            "current_topic_index": target_idx,
            "follow_up_count": 0,
            "completed_topics": completed_topics,
        }

    if action == "change_difficulty":
        adj = decision.get("difficulty_adjustment", "")
        levels = ["easy", "medium", "hard"]
        cur_idx = levels.index(difficulty) if difficulty in levels else 1
        if adj == "up":
            new_idx = min(cur_idx + 1, 2)
        elif adj == "down":
            new_idx = max(cur_idx - 1, 0)
        else:
            new_idx = cur_idx
        return {
            "next_action": "change_difficulty",
            "difficulty": levels[new_idx],
            "follow_up_count": 0,
        }

    if action == "replan":
        return {
            "next_action": "replan",
            "follow_up_count": 0,
        }

    # 兜底
    return {"next_action": "next_question", "follow_up_count": 0}


# ------------------------------------------------------------------ #
#  节点函数
# ------------------------------------------------------------------ #

def router_node(state: InterviewState) -> Dict[str, Any]:
    """
    Router 节点函数。

    优先使用 LLM 自主决策，失败时降级到 if-else 规则。
    """
    total_asked = state.get("total_questions_asked", 0)
    max_questions = state.get("max_questions", 8)
    follow_up_count = state.get("follow_up_count", 0)
    max_follow_ups = state.get("max_follow_ups", 1)
    current_topic_index = state.get("current_topic_index", 0)
    interview_plan = state.get("interview_plan") or []
    current_evaluation = state.get("current_evaluation") or {}
    overall_score = current_evaluation.get("overall_score", 10.0)

    print(
        f"\n[Router] 决策 | 已问: {total_asked}/{max_questions} | "
        f"综合分: {overall_score:.1f} | 追问次数: {follow_up_count}/{max_follow_ups} | "
        f"话题: {current_topic_index + 1}/{len(interview_plan)} | "
        f"难度: {state.get('difficulty', 'medium')}"
    )

    try:
        result = _llm_router(state)
        print(f"[Router] LLM 决策: {result.get('next_action')}")
        return result
    except Exception as e:
        print(f"[Router] LLM 决策失败，降级到 if-else: {e}")
        result = _fallback_router(state)
        print(f"[Router] 兜底决策: {result.get('next_action')}")
        return result


def route_decision(state: InterviewState) -> str:
    """
    条件边函数，供 add_conditional_edges 使用。

    直接读取 state["next_action"] 返回路由方向。
    """
    return state.get("next_action", "end_interview")
