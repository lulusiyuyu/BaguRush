"""
代码质量分析工具。

@tool evaluate_code(code, language) -> str

功能：
  调用 LLM 对面试候选人提交的代码进行全面分析：
  - 正确性：逻辑是否正确，边界情况处理
  - 时间复杂度：大 O 分析
  - 空间复杂度：大 O 分析
  - 代码风格：命名、注释、可读性
  - 改进建议：具体的优化方向

输出 JSON 格式：
  {
    "is_correct": true/false,
    "correctness_notes": "正确性说明",
    "time_complexity": "O(n log n)",
    "space_complexity": "O(n)",
    "complexity_analysis": "复杂度详细说明",
    "style_score": 8,
    "style_notes": "代码风格评价",
    "improvements": ["改进建议1", "改进建议2"],
    "overall_score": 7.5,
    "summary": "总体评价"
  }
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 将项目根目录加入 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv()

_CODE_ANALYSIS_SYSTEM_PROMPT = """你是一位资深软件工程师和代码审查专家。请对候选人提交的代码进行全面、客观的分析。

## 分析维度

1. **正确性** — 代码逻辑是否正确？是否处理了边界情况（空输入、溢出、空指针等）？
2. **时间复杂度** — 用大 O 记号分析，需考虑最坏情况。
3. **空间复杂度** — 用大 O 记号分析，包含递归栈空间。
4. **代码风格** — 命名是否规范、注释是否充分、代码是否整洁（0-10 分）。
5. **改进建议** — 具体、可操作的优化建议（最多 4 条）。

## 输出格式

严格按以下 JSON 格式输出，不得有任何额外文字：

```json
{
  "is_correct": <true 或 false>,
  "correctness_notes": "<正确性说明，指出逻辑错误或边界问题，50字以内>",
  "time_complexity": "<如 O(n)、O(n log n)、O(n²) 等>",
  "space_complexity": "<如 O(1)、O(n) 等>",
  "complexity_analysis": "<时间和空间复杂度的推导分析，100字以内>",
  "style_score": <整数 0-10>,
  "style_notes": "<代码风格评价，50字以内>",
  "improvements": [
    "<改进建议1>",
    "<改进建议2>",
    "<改进建议3>"
  ],
  "overall_score": <浮点数，综合所有维度的得分，保留两位小数，满分 10 分>,
  "summary": "<总体评价，100字以内>"
}
```"""


def _parse_json_result(raw: str) -> dict:
    """从 LLM 输出中提取 JSON，支持多种格式容错。"""
    content = raw.strip()

    # 从 markdown 代码块中提取
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if code_block_match:
        content = code_block_match.group(1).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 提取花括号 JSON 对象
    brace_match = re.search(r"\{[\s\S]*\}", content)
    if brace_match:
        return json.loads(brace_match.group(0))

    raise json.JSONDecodeError("无法从 LLM 输出中提取 JSON", content, 0)


def _get_llm() -> ChatOpenAI:
    from utils.llm_config import get_llm
    return get_llm(temperature=0.1)


# --------------------------------------------------------------------------- #
#  @tool 定义
# --------------------------------------------------------------------------- #

@tool
def evaluate_code(code: str, language: str = "python") -> str:
    """
    分析候选人提交的代码，评估正确性、时间/空间复杂度、代码风格并提供改进建议。

    Args:
        code    : 待分析的代码字符串。
        language: 编程语言（默认 "python"），支持 python、java、cpp、javascript 等。

    Returns:
        JSON 字符串，包含：正确性标志、复杂度分析、风格评分、改进建议和综合评价。
        若分析失败，返回包含 error 字段的 JSON。
    """
    if not code.strip():
        return json.dumps({"error": "代码内容为空，请提供待分析的代码。"}, ensure_ascii=False)

    try:
        llm = _get_llm()

        user_content = (
            f"请分析以下 {language.upper()} 代码：\n\n"
            f"```{language}\n{code}\n```"
        )

        messages = [
            SystemMessage(content=_CODE_ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        response = llm.invoke(messages)
        raw = response.content

        try:
            result = _parse_json_result(raw)
        except json.JSONDecodeError:
            # 重试
            print("[CodeAnalyzer] ⚠️ 第一次 JSON 解析失败，尝试重试...")
            retry_messages = messages + [
                response,
                HumanMessage(content="请只输出 JSON，不要有任何额外文字或 markdown 代码块。"),
            ]
            retry_response = llm.invoke(retry_messages)
            result = _parse_json_result(retry_response.content)

        is_correct = result.get("is_correct", False)
        overall = result.get("overall_score", 0)
        print(
            f"[CodeAnalyzer] 分析完成 | 正确: {'✅' if is_correct else '❌'} | "
            f"时间: {result.get('time_complexity', '未知')} | "
            f"空间: {result.get('space_complexity', '未知')} | "
            f"综合: {overall}/10"
        )

        return json.dumps(result, ensure_ascii=False, indent=2)

    except json.JSONDecodeError as e:
        fallback = {
            "error": f"LLM 输出解析失败: {str(e)}",
            "is_correct": None,
            "correctness_notes": "解析失败，无法评估",
            "time_complexity": "未知",
            "space_complexity": "未知",
            "complexity_analysis": "解析失败",
            "style_score": 5,
            "style_notes": "解析失败",
            "improvements": ["请重新提交代码"],
            "overall_score": 5.0,
            "summary": "代码分析结果解析失败",
        }
        return json.dumps(fallback, ensure_ascii=False)

    except Exception as e:
        error_msg = f"代码分析失败: {type(e).__name__}: {str(e)}"
        print(f"[CodeAnalyzer] ❌ {error_msg}")
        return json.dumps({"error": error_msg}, ensure_ascii=False)
