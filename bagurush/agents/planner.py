"""
Planner Agent 节点。

职责：
  1. 调用 parse_resume 工具解析候选人简历
  2. 调用 search_job_requirements 工具获取岗位要求
  3. 综合分析，制定个性化的面试大纲（interview_plan）
  4. 返回 resume_analysis 和 interview_plan 字段更新

设计说明：
  Planner 采用 ReAct 风格的工具调用循环。LLM 会先调用工具获取信息，
  然后输出最终的 JSON 规划结果。节点会解析 LLM 最后一条非工具调用消息
  中的 JSON 内容作为 interview_plan。
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

# 项目根目录加入 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.state import InterviewState
from prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE
from tools.job_search import search_job_requirements
from tools.resume_parser import parse_resume

load_dotenv()

_TOOLS = [parse_resume, search_job_requirements]
_TOOL_MAP = {t.name: t for t in _TOOLS}


def _get_llm() -> ChatOpenAI:
    from utils.llm_config import get_llm
    return get_llm(temperature=0.2)


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """从 LLM 输出文本中提取 JSON 对象（支持 markdown 代码块）。"""
    # 尝试从 ```json 块提取
    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_match:
        return json.loads(code_match.group(1).strip())

    # 尝试直接找花括号
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        return json.loads(brace_match.group(0))

    raise json.JSONDecodeError("无法从 LLM 输出中找到 JSON", text, 0)


def planner_node(state: InterviewState) -> Dict[str, Any]:
    """
    Planner Agent 节点函数。

    从 InterviewState 读取 resume_file_path / resume_text / job_role / session_id，
    通过 ReAct 工具调用循环制定面试大纲，更新并返回：
      - resume_analysis : dict
      - interview_plan  : list[dict]
      - interview_status: "interviewing"
      - messages        : 追加 AI 消息（含面试大纲）
    """
    print(f"\n[Planner] 开始制定面试大纲 | 岗位: {state['job_role']} | 候选人: {state.get('candidate_name', '未知')}")

    llm = _get_llm()
    llm_with_tools = llm.bind_tools(_TOOLS)

    # 构建初始消息
    resume_file_path = state.get("resume_file_path") or ""
    session_id = state.get("session_id", "default")
    job_role = state.get("job_role", "")

    user_msg = PLANNER_USER_TEMPLATE.format(
        resume_file_path=resume_file_path,
        job_role=job_role,
        session_id=session_id,
    )

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    # -------------------------------------------------------- #
    #  ReAct 工具调用循环（最多 6 轮，防止死循环）
    # -------------------------------------------------------- #
    for iteration in range(6):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # 没有工具调用 → LLM 已给出最终结果
        if not response.tool_calls:
            break

        # 执行工具调用
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"[Planner] 调用工具: {tool_name}({tool_args})")

            if tool_name in _TOOL_MAP:
                try:
                    tool_result = _TOOL_MAP[tool_name].invoke(tool_args)
                except Exception as e:
                    tool_result = f"工具调用失败: {e}"
            else:
                tool_result = f"未知工具: {tool_name}"

            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            )
    else:
        print("[Planner] ⚠️ 超过最大迭代次数，强制退出工具调用循环")

    # -------------------------------------------------------- #
    #  解析最终输出中的 JSON
    # -------------------------------------------------------- #
    final_content = response.content if hasattr(response, "content") else ""

    resume_analysis: Dict[str, Any] = {}
    interview_plan: List[Dict[str, Any]] = []

    try:
        parsed = _extract_json_from_text(final_content)
        resume_analysis = parsed.get("resume_analysis", {})
        interview_plan = parsed.get("interview_plan", [])
        print(f"[Planner] 面试大纲制定完成，话题数: {len(interview_plan)}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Planner] ⚠️ JSON 解析失败: {e}，使用默认大纲")
        # 构造一个兜底大纲，确保面试可以继续
        interview_plan = [
            {"topic": "技术基础", "weight": 0.4, "description": f"{job_role} 核心技术栈考察", "difficulty": "medium", "reason": "兜底"},
            {"topic": "项目经验", "weight": 0.4, "description": "候选人项目深度挖掘", "difficulty": "medium", "reason": "兜底"},
            {"topic": "系统设计", "weight": 0.2, "description": "架构设计思维考察", "difficulty": "medium", "reason": "兜底"},
        ]
        resume_analysis = {"name": state.get("candidate_name", ""), "overall_level": "未知"}

    return {
        "resume_analysis": resume_analysis,
        "interview_plan": interview_plan,
        "interview_status": "interviewing",
        "current_topic_index": 0,
        "follow_up_count": 0,
        "total_questions_asked": 0,
        "all_evaluations": [],
        "next_action": "next_question",
        "messages": [AIMessage(content=f"面试大纲制定完成，共 {len(interview_plan)} 个话题：" +
                               "、".join(t["topic"] for t in interview_plan))],
    }
