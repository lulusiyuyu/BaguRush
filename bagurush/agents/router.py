"""
Router 节点（纯逻辑，不调用 LLM）。

路由规则（优先级从高到低）：
  规则 1: total_questions_asked >= max_questions → "end_interview"
  规则 2: overall_score < 6.0 AND follow_up_count < max_follow_ups → "follow_up"
  规则 3: current_topic_index + 1 < len(interview_plan) → "next_question"
  规则 4: 所有话题已问完 → "end_interview"

同时更新状态：
  follow_up     → follow_up_count += 1
  next_question → current_topic_index += 1, follow_up_count = 0
  end_interview → interview_status = "reporting"
"""

import sys
from pathlib import Path
from typing import Any, Dict

# 项目根目录加入 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.state import InterviewState

# 评分低于此阈值时触发追问
FOLLOW_UP_THRESHOLD = 6.0


def router_node(state: InterviewState) -> Dict[str, Any]:
    """
    Router 节点函数。

    根据当前评估结果和面试进度决定下一步动作，
    更新 next_action 及相关计数字段。
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
        f"话题: {current_topic_index + 1}/{len(interview_plan)}"
    )

    # 规则 1：已达到最大提问数 → 结束
    if total_asked >= max_questions:
        print("[Router] → end_interview（已达最大题数）")
        return {
            "next_action": "end_interview",
            "interview_status": "reporting",
        }

    # 规则 2：当前题得分低 + 还有追问配额 → 追问
    if overall_score < FOLLOW_UP_THRESHOLD and follow_up_count < max_follow_ups:
        print(f"[Router] → follow_up（综合分 {overall_score:.1f} < {FOLLOW_UP_THRESHOLD}，追问）")
        return {
            "next_action": "follow_up",
            "follow_up_count": follow_up_count + 1,
        }

    # 规则 3：还有未问的话题 → 下一个话题
    if current_topic_index + 1 < len(interview_plan):
        print(f"[Router] → next_question（移至话题 {current_topic_index + 2}/{len(interview_plan)}）")
        return {
            "next_action": "next_question",
            "current_topic_index": current_topic_index + 1,
            "follow_up_count": 0,
        }

    # 规则 4：所有话题已覆盖 → 结束
    print("[Router] → end_interview（所有话题已完成）")
    return {
        "next_action": "end_interview",
        "interview_status": "reporting",
    }


def route_decision(state: InterviewState) -> str:
    """
    条件边函数，供 add_conditional_edges 使用。

    直接读取 state["next_action"] 返回路由方向。
    注意：此函数在 router_node 执行完毕后调用，next_action 已是最新值。
    """
    return state.get("next_action", "end_interview")
