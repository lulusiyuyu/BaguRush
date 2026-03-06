"""
岗位要求检索工具。

@tool search_job_requirements(role) -> str

功能：
  1. 从 knowledge_base/jobs/*.json 加载所有岗位数据
  2. 优先精确匹配 role 字段
  3. 精确匹配失败时，使用关键词模糊匹配
  4. 返回格式化的岗位要求文本
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.tools import tool

# 将项目根目录加入 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv()

_JOBS_DIR = _ROOT / "knowledge_base" / "jobs"

# --------------------------------------------------------------------------- #
#  内部函数
# --------------------------------------------------------------------------- #

def _load_all_jobs() -> List[Dict[str, Any]]:
    """加载 jobs 目录下所有 JSON 岗位文件。"""
    jobs = []
    if not _JOBS_DIR.exists():
        return jobs

    for json_file in sorted(_JOBS_DIR.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_file"] = json_file.stem  # 记录来源文件名（无扩展名）
                jobs.append(data)
        except Exception as e:
            print(f"[JobSearch] ⚠️ 加载 {json_file.name} 失败: {e}")

    return jobs


def _format_job(job: Dict[str, Any]) -> str:
    """将岗位 JSON 格式化为易读的文本字符串。"""
    lines = []
    lines.append(f"# 岗位：{job.get('role', '未知')}")

    if desc := job.get("description"):
        lines.append(f"\n**岗位描述**：{desc}")

    if required := job.get("required_skills"):
        lines.append("\n**必备技能**：")
        for skill in required:
            lines.append(f"  - {skill}")

    if preferred := job.get("preferred_skills"):
        lines.append("\n**加分项技能**：")
        for skill in preferred:
            lines.append(f"  - {skill}")

    if topics := job.get("interview_topics"):
        lines.append("\n**面试考察方向**：")
        for t in topics:
            weight_pct = int(t.get("weight", 0) * 100)
            desc_str = f"（{t['description']}）" if t.get("description") else ""
            lines.append(f"  - {t['topic']} {weight_pct}% {desc_str}")

    if dist := job.get("difficulty_distribution"):
        easy = dist.get("easy", 0)
        medium = dist.get("medium", 0)
        hard = dist.get("hard", 0)
        lines.append(f"\n**题目难度分布**：简单 {easy} 题 / 中等 {medium} 题 / 困难 {hard} 题")

    if projs := job.get("typical_projects"):
        lines.append("\n**典型项目经历**：")
        for p in projs[:3]:
            lines.append(f"  - {p}")

    return "\n".join(lines)


def _match_job(role: str, jobs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    匹配最合适的岗位。
    优先级：精确匹配 > 部分匹配 > 关键词匹配。
    """
    if not jobs:
        return None

    role_lower = role.strip().lower()

    # 1. 精确匹配（忽略大小写）
    for job in jobs:
        if job.get("role", "").lower() == role_lower:
            return job

    # 2. 部分匹配（role 包含在 job.role 中，或 job.role 包含在 role 中）
    for job in jobs:
        job_role = job.get("role", "").lower()
        if role_lower in job_role or job_role in role_lower:
            return job

    # 3. 关键词匹配（检查 role 中的每个词是否出现在岗位名称或描述中）
    keywords = set(role_lower.replace("工程师", "").replace("开发", "").split())
    best_match = None
    best_score = 0

    for job in jobs:
        job_text = (job.get("role", "") + " " + job.get("description", "")).lower()
        score = sum(1 for kw in keywords if kw in job_text)
        if score > best_score:
            best_score = score
            best_match = job

    # 4. 如果还是没有匹配，直接根据文件名匹配
    if best_score == 0:
        for job in jobs:
            file_name = job.get("_file", "").lower()
            for kw in ["backend", "ml", "agent", "recsys", "后端", "机器学习", "推荐"]:
                if kw in role_lower and kw in file_name:
                    return job

    return best_match


# --------------------------------------------------------------------------- #
#  @tool 定义
# --------------------------------------------------------------------------- #

@tool
def search_job_requirements(role: str) -> str:
    """
    检索指定职位的面试要求，包括必备技能、加分技能、考察方向和题目难度分布。

    Args:
        role: 目标岗位名称，例如 "后端开发工程师"、"机器学习工程师"、"推荐系统工程师" 等。

    Returns:
        格式化的岗位要求文本字符串。若未找到匹配岗位，返回所有可用岗位列表。
    """
    try:
        jobs = _load_all_jobs()

        if not jobs:
            return f"❌ 未找到任何岗位数据，请检查 {_JOBS_DIR} 目录是否存在 JSON 文件。"

        matched = _match_job(role, jobs)

        if matched:
            role_name = matched.get("role", "未知")
            file_name = matched.get("_file", "")
            print(f"[JobSearch] 匹配岗位: '{role_name}'（来源文件: {file_name}.json）")
            return _format_job(matched)
        else:
            available_roles = [j.get("role", "未知") for j in jobs]
            return (
                f"未找到与 '{role}' 匹配的岗位。\n"
                f"当前可用岗位：{', '.join(available_roles)}\n"
                f"请尝试使用更精确的岗位名称。"
            )

    except Exception as e:
        error_msg = f"岗位检索失败: {type(e).__name__}: {str(e)}"
        print(f"[JobSearch] ❌ {error_msg}")
        return f"❌ {error_msg}"
