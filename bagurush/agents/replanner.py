"""
Replanner 节点 — 动态重规划面试计划。

触发条件：Router 检测到 new_findings 不为空时路由到此节点。

职责：
  1. 读取 new_findings（Evaluator 发现的候选人主动提及的 plan 外技术）
  2. 读取已完成话题 completed_topics 和剩余 plan
  3. 调 LLM 输出调整后的"剩余计划"（current_topic_index 之后的部分）
  4. 合并：原 plan 前半段（已完成）+ 新 plan（LLM 输出）
  5. 清空 new_findings（避免下次再次触发）

降级兜底：
  LLM 调用失败或 JSON 解析失败 → plan 不变，仅清空 new_findings，面试继续。
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
from utils.llm_config import get_llm


# ------------------------------------------------------------------ #
#  Prompt 模板
# ------------------------------------------------------------------ #

_REPLAN_PROMPT_TEMPLATE = """你是面试流程规划师。面试进行到一半，候选人的回答中暴露了一些新情况，需要调整后续的面试计划。

【原始面试计划（完整版）】
{full_plan}

【已完成的话题】（index 0 ~ {last_done_index}，这部分不需要调整）
{completed_info}

【新发现】（候选人主动提及的计划外技术，值得深入考察）
{new_findings}

【候选人画像】
{candidate_profile}

请输出调整后的【剩余面试计划】（即 index {next_index} 开始的部分）。

要求：
- 尽量保留原有话题，只做必要的调整
- 若新发现值得考察，可在剩余计划中插入相关话题
- 总话题数不要超过原剩余话题数 + 2
- 每个话题包含 topic（话题名）和 difficulty（easy/medium/hard）

严格输出 JSON 数组，不要输出任何额外文字：
[
  {{"topic": "话题名称", "difficulty": "medium"}},
  ...
]"""


# ------------------------------------------------------------------ #
#  辅助函数
# ------------------------------------------------------------------ #

def _parse_plan_from_text(text: str) -> List[Dict[str, Any]]:
    """从 LLM 输出中提取 JSON 数组格式的 plan。"""
    # 尝试匹配 markdown 代码块
    code_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if code_match:
        return json.loads(code_match.group(1).strip())
    # 尝试直接匹配数组
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        return json.loads(arr_match.group(0))
    raise json.JSONDecodeError("无法从输出中找到 JSON 数组", text, 0)


def _validate_plan(plan: Any) -> bool:
    """校验 plan 格式：必须是 list，每项有 topic 字段。"""
    if not isinstance(plan, list):
        return False
    return all(isinstance(item, dict) and "topic" in item for item in plan)


# ------------------------------------------------------------------ #
#  节点函数
# ------------------------------------------------------------------ #

def replanner_node(state: InterviewState) -> Dict[str, Any]:
    """
    Replanner 节点函数。

    基于 new_findings 和候选人画像，调 LLM 调整剩余面试计划。
    失败时保持原 plan 不变，仅清空 new_findings。
    """
    interview_plan: List[Dict[str, Any]] = list(state.get("interview_plan") or [])
    current_topic_index: int = state.get("current_topic_index", 0)
    new_findings: List[str] = list(state.get("new_findings") or [])
    completed_topics: List[str] = list(state.get("completed_topics") or [])
    candidate_profile = state.get("candidate_profile") or {}

    print(f"\n[Replanner] 触发重规划 | new_findings: {new_findings}")
    print(f"[Replanner] 当前进度: index={current_topic_index}/{len(interview_plan)}")

    # 已完成部分（index 0 ~ current_topic_index-1 保持不变）
    done_plan = interview_plan[:current_topic_index]
    remaining_plan = interview_plan[current_topic_index:]

    prompt_text = _REPLAN_PROMPT_TEMPLATE.format(
        full_plan=json.dumps(interview_plan, ensure_ascii=False, indent=2),
        last_done_index=current_topic_index - 1,
        completed_info=json.dumps(completed_topics, ensure_ascii=False),
        new_findings="\n".join(f"- {f}" for f in new_findings),
        candidate_profile=json.dumps(candidate_profile, ensure_ascii=False, indent=2),
        next_index=current_topic_index,
    )

    try:
        llm = get_llm(temperature=0.5)
        response = llm.invoke(prompt_text)
        new_remaining = _parse_plan_from_text(response.content)

        if not _validate_plan(new_remaining):
            raise ValueError(f"LLM 输出的 plan 格式非法: {new_remaining}")

        merged_plan = done_plan + new_remaining
        print(f"[Replanner] 重规划成功 | 原剩余 {len(remaining_plan)} 个话题 → 新剩余 {len(new_remaining)} 个话题")
        print(f"[Replanner] 新 plan: {[t.get('topic') for t in new_remaining]}")
        return {
            "interview_plan": merged_plan,
            "new_findings": [],
        }

    except Exception as e:
        print(f"[Replanner] ❌ 重规划失败，保持原 plan: {e}")
        # 降级：plan 不变，只清空 new_findings 防止死循环
        return {
            "interview_plan": interview_plan,
            "new_findings": [],
        }
