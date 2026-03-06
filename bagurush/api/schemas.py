"""
Pydantic 请求/响应模型定义。

对应 API 端点：
  POST /api/interview/start      → StartInterviewResponse
  POST /api/interview/{id}/answer → AnswerResponse
  GET  /api/interview/{id}/status → StatusResponse
  GET  /api/interview/{id}/report → ReportResponse
  GET  /api/interview/{id}/history → HistoryResponse
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
#  启动面试
# ------------------------------------------------------------------ #

class StartInterviewResponse(BaseModel):
    """POST /api/interview/start 的响应。"""
    session_id: str = Field(description="面试会话 ID")
    message: str = Field(description="Planner 初始分析摘要")
    interview_plan: List[Dict[str, Any]] = Field(default_factory=list, description="面试大纲")
    first_question: Optional[str] = Field(default=None, description="第一个面试问题")


# ------------------------------------------------------------------ #
#  提交回答
# ------------------------------------------------------------------ #

class AnswerRequest(BaseModel):
    """POST /api/interview/{session_id}/answer 的请求体。"""
    answer: str = Field(description="候选人的回答")


class AnswerResponse(BaseModel):
    """POST /api/interview/{session_id}/answer 的响应。"""
    evaluation: Optional[Dict[str, Any]] = Field(default=None, description="当前回答评估")
    next_question: Optional[str] = Field(default=None, description="下一个面试问题")
    is_follow_up: bool = Field(default=False, description="是否为追问")
    interview_ended: bool = Field(default=False, description="面试是否已结束")
    progress: str = Field(default="0/0", description="进度 (已答/总数)")
    topic: Optional[str] = Field(default=None, description="当前话题")


# ------------------------------------------------------------------ #
#  查询状态
# ------------------------------------------------------------------ #

class StatusResponse(BaseModel):
    """GET /api/interview/{session_id}/status 的响应。"""
    session_id: str
    status: str = Field(description="面试状态: planning/interviewing/reporting/completed")
    total_questions_asked: int = 0
    max_questions: int = 8
    progress: str = "0/0"
    current_topic: Optional[str] = None


# ------------------------------------------------------------------ #
#  获取报告
# ------------------------------------------------------------------ #

class ReportResponse(BaseModel):
    """GET /api/interview/{session_id}/report 的响应。"""
    report: str = Field(description="Markdown 格式面试报告")
    overall_score: float = Field(default=0.0, description="综合得分")
    grade: str = Field(default="N/A", description="等级 A/B/C/D")
    evaluations: List[Dict[str, Any]] = Field(default_factory=list, description="所有评估记录")


# ------------------------------------------------------------------ #
#  对话历史
# ------------------------------------------------------------------ #

class MessageItem(BaseModel):
    """单条消息。"""
    role: str = Field(description="消息角色: interviewer / candidate / system")
    content: str = Field(description="消息内容")
    name: Optional[str] = None


class HistoryResponse(BaseModel):
    """GET /api/interview/{session_id}/history 的响应。"""
    session_id: str
    messages: List[MessageItem] = Field(default_factory=list)


# ------------------------------------------------------------------ #
#  通用错误
# ------------------------------------------------------------------ #

class ErrorResponse(BaseModel):
    """通用错误响应。"""
    detail: str
