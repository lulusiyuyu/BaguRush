"""
Interviewer Agent 节点。

职责：
  1. 根据 next_action 决定提新问题还是追问
  2. 可选调用 search_tech_knowledge 查阅背景知识
  3. 生成问题文本，更新 current_question 和 messages
  4. 调用 interrupt() 暂停图执行，等待候选人回答

LangGraph interrupt 机制说明：
  调用 interrupt(value) 后，图会暂停并将 value 作为 interrupt 数据返回给外部。
  外部（API 层）调用 graph.invoke(Command(resume=answer), config) 来恢复执行。
  恢复时，interrupt() 的返回值就是 answer 字符串。
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

# 项目根目录加入 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.state import InterviewState
from prompts.interviewer_prompt import (
    INTERVIEWER_FOLLOW_UP_TEMPLATE,
    INTERVIEWER_NEW_QUESTION_TEMPLATE,
    INTERVIEWER_SYSTEM_PROMPT,
)
from tools.knowledge_rag import search_tech_knowledge

load_dotenv()

_TOOLS = [search_tech_knowledge]
_TOOL_MAP = {t.name: t for t in _TOOLS}


def _get_llm() -> ChatOpenAI:
    from utils.llm_config import get_llm
    return get_llm(temperature=0.7)


def _get_resume_summary(state: InterviewState) -> str:
    """从 resume_analysis 中提取简短摘要。"""
    analysis = state.get("resume_analysis") or {}
    name = analysis.get("name", state.get("candidate_name", "候选人"))
    level = analysis.get("overall_level", "")
    strengths = analysis.get("strengths", [])
    projects = analysis.get("key_projects", [])

    parts = [f"候选人：{name}"]
    if level:
        parts.append(f"技术水平：{level}")
    if strengths:
        parts.append(f"优势：{', '.join(strengths[:3])}")
    if projects:
        parts.append(f"核心项目：{', '.join(projects[:2])}")
    return "；".join(parts)


def _generate_question(state: InterviewState) -> str:
    """调用 LLM 生成面试问题，必要时使用 search_tech_knowledge 工具。"""
    llm = _get_llm()
    llm_with_tools = llm.bind_tools(_TOOLS)

    next_action = state.get("next_action", "next_question")
    interview_plan = state.get("interview_plan") or []
    current_topic_index = state.get("current_topic_index", 0)
    current_topic = interview_plan[current_topic_index] if current_topic_index < len(interview_plan) else {}

    # 构建用户消息
    if next_action == "follow_up":
        eval_data = state.get("current_evaluation") or {}
        follow_up_suggestion = eval_data.get("follow_up_suggestion", "请对上一个问题进行追问")
        user_content = INTERVIEWER_FOLLOW_UP_TEMPLATE.format(
            candidate_name=state.get("candidate_name", "候选人"),
            job_role=state.get("job_role", ""),
            current_topic=current_topic.get("topic", ""),
            current_question=state.get("current_question", ""),
            follow_up_count=state.get("follow_up_count", 0),
            max_follow_ups=state.get("max_follow_ups", 1),
            follow_up_suggestion=follow_up_suggestion,
        )
    else:
        user_content = INTERVIEWER_NEW_QUESTION_TEMPLATE.format(
            candidate_name=state.get("candidate_name", "候选人"),
            job_role=state.get("job_role", ""),
            current_topic=current_topic.get("topic", ""),
            topic_description=current_topic.get("description", ""),
            difficulty=current_topic.get("difficulty", "medium"),
            total_questions_asked=state.get("total_questions_asked", 0),
            max_questions=state.get("max_questions", 8),
            resume_summary=_get_resume_summary(state),
        )

    messages = [
        SystemMessage(content=INTERVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    # ReAct 工具调用循环（最多 3 轮）
    for _ in range(3):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            if tool_name in _TOOL_MAP:
                try:
                    result = _TOOL_MAP[tool_name].invoke(tool_call["args"])
                except Exception as e:
                    result = f"工具调用失败: {e}"
            else:
                result = f"未知工具: {tool_name}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

    question_text = response.content.strip() if hasattr(response, "content") else "请介绍一下您的技术背景。"
    return question_text


def interviewer_node(state: InterviewState) -> Dict[str, Any]:
    """
    Interviewer Agent 节点函数。

    生成面试问题后，调用 interrupt() 暂停等待候选人回答。
    返回时更新：
      - current_question : 当前问题文本
      - messages         : 追加面试官消息
    """
    next_action = state.get("next_action", "next_question")
    interview_plan = state.get("interview_plan") or []
    current_topic_index = state.get("current_topic_index", 0)
    current_topic = interview_plan[current_topic_index].get("topic", "") if current_topic_index < len(interview_plan) else ""

    action_label = "追问" if next_action == "follow_up" else "新问题"
    print(f"\n[Interviewer] 生成{action_label} | 话题: {current_topic} | 进度: {state.get('total_questions_asked', 0)}/{state.get('max_questions', 8)}")

    # 生成问题
    question_text = _generate_question(state)
    print(f"[Interviewer] 问题: {question_text[:80]}{'...' if len(question_text) > 80 else ''}")

    # 将问题加入消息历史（面试官消息）
    ai_message = AIMessage(content=question_text, name="interviewer")

    # -------------------------------------------------------- #
    #  Human-in-the-Loop：暂停等待候选人回答
    #
    #  interrupt() 会抛出 GraphInterrupt 异常，LangGraph 捕获后：
    #  1. 保存当前状态到 checkpointer（MemorySaver）
    #  2. 将 interrupt 数据返回给外部调用者
    #
    #  外部调用 graph.invoke(Command(resume=answer), config) 后：
    #  1. 从 checkpointer 恢复状态
    #  2. interrupt() 返回 answer 字符串，候选人回答进入对话历史
    # -------------------------------------------------------- #
    user_answer = interrupt({
        "question": question_text,
        "topic": current_topic,
        "question_index": state.get("total_questions_asked", 0) + 1,
    })

    # interrupt 恢复后，user_answer 是候选人的回答字符串
    # 将候选人回答也加入消息历史
    human_message = HumanMessage(content=str(user_answer), name="candidate")

    return {
        "current_question": question_text,
        "messages": [ai_message, human_message],
    }
