"""
BaguRush LangGraph 状态定义。

InterviewState 是整个面试流程中流转于各 Agent 节点之间的共享状态字典。
所有节点返回的 dict 都是对该状态的部分更新（patch）。
"""

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class InterviewState(TypedDict):
    """
    面试流程的全局共享状态。

    字段分组：
    - 【输入】    候选人信息，由外部传入
    - 【Planner】 简历分析结果 + 面试大纲
    - 【Interviewer】 当前提问状态
    - 【Evaluator】  当前题和历史评估
    - 【Router】    下一步路由决策
    - 【Reporter】  最终面试报告
    - 【全局控制】  进度计数、终止标志
    """

    # -------------------------------------------------------- #
    #  输入信息
    # -------------------------------------------------------- #
    messages: Annotated[List[BaseMessage], add_messages]
    """对话历史消息列表，使用 add_messages 累加器自动追加新消息。"""

    resume_text: str
    """候选人简历全文（UTF-8 字符串，由 API 层在启动时注入）。"""

    job_role: str
    """目标岗位名称，例如 '推荐系统工程师'。"""

    candidate_name: str
    """候选人姓名。"""

    resume_file_path: Optional[str]
    """简历文件绝对路径（可选），供 parse_resume tool 使用。"""

    session_id: str
    """会话唯一 ID，用于隔离 session 级别的 FAISS 向量索引。"""

    # -------------------------------------------------------- #
    #  Planner 输出
    # -------------------------------------------------------- #
    resume_analysis: Optional[Dict[str, Any]]
    """
    结构化简历分析结果（JSON 解析后的 dict），例如：
    {"name": "张三", "skills": [...], "projects": [...], ...}
    """

    interview_plan: Optional[List[Dict[str, Any]]]
    """
    面试大纲，由 Planner 制定，格式：
    [
      {"topic": "Python 基础", "weight": 0.3, "description": "考察 GIL、装饰器等"},
      {"topic": "系统设计",    "weight": 0.4, "description": "高并发场景设计"},
      ...
    ]
    """

    # -------------------------------------------------------- #
    #  Interviewer 状态
    # -------------------------------------------------------- #
    current_topic_index: int
    """当前正在考察的 interview_plan 索引（从 0 开始）。"""

    current_question: Optional[str]
    """面试官当前提出的问题文本。"""

    follow_up_count: int
    """当前话题已追问次数。超过 max_follow_ups 后强制换题。"""

    max_follow_ups: int
    """每道题最多追问次数（默认 1）。"""

    # -------------------------------------------------------- #
    #  Evaluator 输出
    # -------------------------------------------------------- #
    current_evaluation: Optional[Dict[str, Any]]
    """
    当前题的评估结果（最新一次），格式：
    {
      "completeness": 8, "accuracy": 7, "depth": 6, "expression": 8,
      "overall_score": 7.25,
      "feedback": "...",
      "follow_up_suggestion": "..."
    }
    """

    all_evaluations: List[Dict[str, Any]]
    """历次评估结果列表，按问答顺序追加。"""

    # -------------------------------------------------------- #
    #  Router 状态
    # -------------------------------------------------------- #
    next_action: Optional[str]
    """
    Router 决定的下一步动作：
    - "next_question"      : 进入下一个话题
    - "follow_up"          : 对当前话题追问
    - "end_interview"      : 结束面试，进入 Reporter
    - "switch_topic"       : 跳到指定话题（Router 指定 index）
    - "change_difficulty"  : 调整难度后重新出题
    """

    router_reason: Optional[str]
    """Router 决策理由，用于前端展示 Agent 思考过程。"""

    # -------------------------------------------------------- #
    #  候选人画像 & 动态决策
    # -------------------------------------------------------- #
    candidate_profile: Optional[Dict[str, Any]]
    """
    候选人多维画像，Router 决策依据。格式示例：
    {
      "dimensions": {
        "algorithm":     {"score": 6.5, "confidence": 0.3, "evidence": [...]},
        "system_design": {"score": 4.0, "confidence": 0.3, "evidence": [...]},
        ...
      },
      "weak_spots": ["fundamentals"],
      "strong_spots": ["communication"],
    }
    由 Planner 初始化，Evaluator 每次评估后增量更新。
    """

    difficulty: str
    """当前面试难度（easy/medium/hard），默认 medium。Router 可动态调整。"""

    new_findings: Optional[List[str]]
    """面试中发现的新信息列表（触发 RePlan 的依据）。"""

    completed_topics: List[str]
    """已完成的话题名称列表（供 RePlan 和 Router 参考）。"""

    # -------------------------------------------------------- #
    #  Reporter 输出
    # -------------------------------------------------------- #
    final_report: Optional[str]
    """Markdown 格式的完整面试报告，由 Reporter Agent 生成。"""

    # -------------------------------------------------------- #
    #  全局控制
    # -------------------------------------------------------- #
    total_questions_asked: int
    """已经提问的总题数（含追问）。"""

    max_questions: int
    """面试最多提问题数（含追问），达到上限后强制结束。"""

    interview_status: str
    """
    面试生命周期状态：
    - "planning"     : Planner 正在制定大纲
    - "interviewing" : 面试进行中
    - "reporting"    : Reporter 正在生成报告
    - "completed"    : 面试结束
    """
