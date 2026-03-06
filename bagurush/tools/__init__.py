"""
tools 包 — BaguRush 工具函数模块

对外暴露 5 个 LangChain @tool：
  - parse_resume           : 简历解析（PDF/MD → 结构化 JSON）
  - search_job_requirements: 岗位要求检索
  - search_tech_knowledge  : 技术知识 RAG 检索
  - evaluate_answer        : 回答多维度评估
  - evaluate_code          : 代码质量分析

Session 管理辅助：
  - get_session_store      : 获取指定会话的简历向量索引
  - SESSION_STORES         : 全局 session 向量索引字典
"""

from tools.answer_evaluator import evaluate_answer
from tools.code_analyzer import evaluate_code
from tools.job_search import search_job_requirements
from tools.knowledge_rag import search_tech_knowledge
from tools.resume_parser import SESSION_STORES, get_session_store, parse_resume

__all__ = [
    "parse_resume",
    "search_job_requirements",
    "search_tech_knowledge",
    "evaluate_answer",
    "evaluate_code",
    "get_session_store",
    "SESSION_STORES",
]
