"""
简历解析工具。

@tool parse_resume(file_path, session_id) -> str

功能：
  1. 加载 PDF 或 Markdown 格式的简历文件
  2. 调用 DeepSeek LLM 提取结构化信息（姓名、教育、技能、项目、经历、竞赛）
  3. 将简历全文向量化并存入 session 级别的 FAISS 索引（供后续 Agent 检索）
  4. 返回 JSON 字符串格式的结构化简历信息

Session 索引管理：
  - 全局字典 SESSION_STORES 并按 session_id 缓存 VectorStoreManager 实例
  - 外部可通过 get_session_store(session_id) 获取对应的向量库
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 将项目根目录加入 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag.document_loader import load_document, split_documents
from rag.vector_store import VectorStoreManager, create_session_store

load_dotenv()

# --------------------------------------------------------------------------- #
#  Session 存储：全局字典，按 session_id 缓存简历向量索引
# --------------------------------------------------------------------------- #
SESSION_STORES: Dict[str, VectorStoreManager] = {}


def get_session_store(session_id: str) -> Optional[VectorStoreManager]:
    """获取指定会话的简历向量索引，不存在则返回 None。"""
    return SESSION_STORES.get(session_id)


def _get_llm() -> ChatOpenAI:
    """创建 LLM 实例（优先使用前端配置）。"""
    from utils.llm_config import get_llm
    return get_llm(temperature=0.1)


_EXTRACT_SYSTEM_PROMPT = """你是一位专业的简历分析专家。请从用户提供的简历文本中提取结构化信息，严格按照以下 JSON 格式输出，不要输出任何额外内容：

{
  "name": "候选人姓名（字符串，未找到则为空字符串）",
  "contact": {
    "email": "邮箱（未找到则为空字符串）",
    "phone": "电话（未找到则为空字符串）",
    "github": "GitHub 主页（未找到则为空字符串）"
  },
  "education": [
    {
      "school": "学校名称",
      "degree": "学历（本科/硕士/博士等）",
      "major": "专业",
      "period": "在读/毕业时间段（如 2020.09 - 2024.06）"
    }
  ],
  "skills": ["技能1", "技能2", "..."],
  "work_experience": [
    {
      "company": "公司名称",
      "position": "职位",
      "period": "时间段",
      "description": "工作内容描述"
    }
  ],
  "projects": [
    {
      "name": "项目名称",
      "tech_stack": ["技术1", "技术2"],
      "description": "项目描述",
      "highlights": "亮点或成果"
    }
  ],
  "competitions": [
    {
      "name": "竞赛名称",
      "award": "奖项",
      "period": "时间"
    }
  ],
  "summary": "对候选人的整体评价（50字以内）"
}

注意：所有字段都必须保留，即使值为空列表或空字符串。输出必须是合法的 JSON。"""


@tool
def parse_resume(file_path: str, session_id: str = "default") -> str:
    """
    解析候选人简历文件，提取结构化信息并建立简历向量索引。

    Args:
        file_path: 简历文件的绝对路径（支持 .pdf 和 .md 格式）。
        session_id: 面试会话 ID，用于隔离不同候选人的向量索引，默认 "default"。

    Returns:
        JSON 字符串，包含候选人的姓名、教育背景、技能、项目、工作经历、竞赛等信息。
        若解析失败，返回包含 error 字段的 JSON。
    """
    try:
        # ---------------------------------------------------------- #
        #  1. 加载文档
        # ---------------------------------------------------------- #
        docs = load_document(file_path)
        if not docs:
            return json.dumps({"error": f"无法加载文件: {file_path}"}, ensure_ascii=False)

        # 合并所有页面文本
        full_text = "\n\n".join(doc.page_content for doc in docs)
        print(f"[ResumeParser] 加载简历: {Path(file_path).name}，字符数: {len(full_text)}")

        # ---------------------------------------------------------- #
        #  2. 调用 LLM 提取结构化信息
        # ---------------------------------------------------------- #
        llm = _get_llm()
        messages = [
            SystemMessage(content=_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=f"请分析以下简历内容：\n\n{full_text[:8000]}"),  # 截取前 8000 字符避免超 token
        ]
        response = llm.invoke(messages)
        raw_content = response.content.strip()

        # ---------------------------------------------------------- #
        #  3. 解析 JSON 输出
        # ---------------------------------------------------------- #
        # 去掉可能的 markdown 代码块包装
        if raw_content.startswith("```"):
            lines = raw_content.split("\n")
            # 去掉第一行 ```json 和最后一行 ```
            raw_content = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        resume_data = json.loads(raw_content)
        print(f"[ResumeParser] 提取完成，候选人: {resume_data.get('name', '未知')}")

        # ---------------------------------------------------------- #
        #  4. 向量化并存入 session 级索引
        # ---------------------------------------------------------- #
        _build_session_index(session_id, docs, file_path)

        return json.dumps(resume_data, ensure_ascii=False, indent=2)

    except json.JSONDecodeError as e:
        error_msg = f"LLM 返回的内容无法解析为 JSON: {str(e)}"
        print(f"[ResumeParser] ⚠️ {error_msg}")
        # 即使 JSON 解析失败，也尝试返回原始文本摘要
        return json.dumps({
            "error": error_msg,
            "raw_text": full_text[:500] if "full_text" in dir() else "",
            "summary": "JSON 解析失败，请查看 raw_text 字段获取原始内容"
        }, ensure_ascii=False)

    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    except Exception as e:
        error_msg = f"简历解析失败: {type(e).__name__}: {str(e)}"
        print(f"[ResumeParser] ❌ {error_msg}")
        return json.dumps({"error": error_msg}, ensure_ascii=False)


def _build_session_index(session_id: str, docs: list, file_path: str) -> None:
    """
    将简历文档向量化并存入 session 级 FAISS 索引。

    Args:
        session_id: 会话 ID。
        docs: 已加载的 Document 列表。
        file_path: 原始文件路径（用于元数据记录）。
    """
    try:
        # 为文档添加 source 元数据
        for doc in docs:
            doc.metadata["source"] = file_path
            doc.metadata["type"] = "resume"
            doc.metadata["session_id"] = session_id

        # 切分文档
        chunks = split_documents(docs, chunk_size=300, chunk_overlap=30)

        # 创建或更新 session 向量索引
        if session_id not in SESSION_STORES:
            SESSION_STORES[session_id] = create_session_store(session_id)

        SESSION_STORES[session_id].add_documents(chunks)
        print(
            f"[ResumeParser] Session '{session_id}' 索引更新，"
            f"chunk 数: {len(chunks)}，"
            f"总向量数: {SESSION_STORES[session_id].doc_count}"
        )
    except Exception as e:
        print(f"[ResumeParser] ⚠️ 向量化索引构建失败（不影响返回结果）: {e}")
