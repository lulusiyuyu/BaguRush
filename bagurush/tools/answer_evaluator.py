"""
回答多维度评估工具。

@tool evaluate_answer(question, answer, reference) -> str

评分维度（每项 0~10 分）：
  - completeness：完整性（是否涵盖了问题的所有关键点）
  - accuracy    ：准确性（技术细节是否正确）
  - depth       ：深度  （是否有原理分析、延伸讨论）
  - expression  ：表达  （语言是否清晰、逻辑是否通顺）
  - overall_score：综合得分（加权平均）

输出 JSON 格式：
  {
    "completeness": 8,
    "accuracy": 7,
    "depth": 6,
    "expression": 8,
    "overall_score": 7.25,
    "feedback": "总体评价...",
    "follow_up_suggestion": "建议追问..."
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

_EVALUATION_SYSTEM_PROMPT = """你是一位严格且公正的技术面试评估专家。请根据候选人对面试问题的回答，从以下四个维度打分（每项 0~10 分整数），并给出反馈和追问建议。

## 评分维度说明

| 维度 | 满分 | 评分标准 |
|------|------|----------|
| completeness（完整性） | 10 | 是否覆盖了问题的全部关键要点（对照【参考资料】判断）：9-10=极其完整；7-8=基本完整；5-6=有遗漏；3-4=遗漏较多；0-2=严重缺失 |
| accuracy（准确性） | 10 | 技术细节是否正确（对照【参考资料】验证）：9-10=完全正确；7-8=基本正确，小错；5-6=部分正确；3-4=错误较多；0-2=基本错误 |
| depth（深度） | 10 | 是否有原理分析、底层机制、对比讨论：9-10=有深度原理分析；7-8=有一定深度；5-6=表面回答；3-4=过于浅显；0-2=几乎没有 |
| expression（表达） | 10 | 语言是否清晰、结构是否合理：9-10=条理清晰，逻辑严密；7-8=较清晰；5-6=基本清楚；3-4=有些混乱；0-2=难以理解 |

## 输出要求

严格按以下 JSON 格式输出，不得有任何额外文字：

```json
{
  "completeness": <整数 0-10>,
  "accuracy": <整数 0-10>,
  "depth": <整数 0-10>,
  "expression": <整数 0-10>,
  "overall_score": <浮点数，四个维度的简单平均，保留两位小数>,
  "feedback": "<对候选人回答的总体评价，100字以内，指出优点和不足>",
  "follow_up_suggestion": "<指出【参考资料】中候选人未提到的1-2个最关键知识点，格式为'候选人没有提到[具体概念]，可以追问：[具体问题]'；如参考资料为空则基于你自己的知识判断，50字以内>",
  "profile_update": {
    "dimension": "<algorithm|system_design|project_depth|communication|fundamentals>",
    "score_delta": <浮点数，-2.0 ~ +2.0，表示本次回答对该维度评分的调整量>,
    "evidence": "<简短依据，30字以内>"
  },
  "new_mention": {
    "skill": "<候选人主动提及的技术/话题名称，如 Kafka、Flink、Raft>",
    "context": "<候选人提及该技术的具体语境，20字以内>"
  }
}
```

profile_update 说明：
- dimension: 本题最相关的候选人能力维度（五选一）
- score_delta: 正数表示表现好于预期，负数表示差于预期，0 表示符合预期
- evidence: 用一句话说明为什么这样调整

new_mention 说明：
- 仅当候选人在回答中**主动提及**了面试问题范围之外的技术或话题，且该技术值得深入考察时才填写
- 如果候选人没有提及任何计划外技术，请将 new_mention 设为 null（不要随意填写）
- skill: 技术名称，尽量简短（如 "Kafka"，不要写 "Apache Kafka 消息队列"）"""


def _parse_evaluation_json(raw: str) -> dict:
    """
    从 LLM 输出中提取评估 JSON，支持多种格式容错。

    Args:
        raw: LLM 原始返回文本。

    Returns:
        解析后的评估字典。

    Raises:
        json.JSONDecodeError: 无法解析时抛出。
    """
    # 去除首尾空白
    content = raw.strip()

    # 尝试从 markdown 代码块中提取
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if code_block_match:
        content = code_block_match.group(1).strip()

    # 直接尝试解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 尝试提取第一个花括号包围的 JSON 对象
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
def evaluate_answer(question: str, answer: str, reference: str = "") -> str:
    """
    对面试问题的候选人回答进行多维度评估，返回结构化评分结果。

    评分维度：完整性、准确性、深度、表达力（各 0-10 分）。

    Args:
        question : 面试题目内容。
        answer   : 候选人的回答文本。
        reference: 可选的参考答案（由 RAG 检索提供），提供后评估更精准。

    Returns:
        JSON 字符串，包含各维度分数、综合分数、总体评价和追问建议。
        若评估失败，返回包含 error 字段的 JSON。
    """
    try:
        llm = _get_llm()

        # 构建用户消息
        user_content_parts = [
            f"## 面试题目\n{question}",
            f"\n## 候选人回答\n{answer}",
        ]
        if reference.strip():
            user_content_parts.append(f"\n## 【参考资料】（用于评分和生成追问建议）\n{reference[:2000]}")

        user_content = "\n".join(user_content_parts)

        messages = [
            SystemMessage(content=_EVALUATION_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        # 首次尝试
        response = llm.invoke(messages)
        raw = response.content

        try:
            eval_data = _parse_evaluation_json(raw)
        except json.JSONDecodeError:
            # 重试：明确要求输出纯 JSON
            print("[Evaluator] ⚠️ 第一次解析失败，正在重试...")
            retry_messages = messages + [
                response,
                HumanMessage(content="请只输出 JSON，不要有任何额外文字或 markdown 代码块。"),
            ]
            retry_response = llm.invoke(retry_messages)
            eval_data = _parse_evaluation_json(retry_response.content)

        # 确保 overall_score 是数值计算的正确值
        scores = [
            eval_data.get("completeness", 0),
            eval_data.get("accuracy", 0),
            eval_data.get("depth", 0),
            eval_data.get("expression", 0),
        ]
        eval_data["overall_score"] = round(sum(scores) / len(scores), 2)

        print(
            f"[Evaluator] 评估完成，综合分: {eval_data['overall_score']:.1f}/10 | "
            f"完整性:{eval_data.get('completeness')} 准确性:{eval_data.get('accuracy')} "
            f"深度:{eval_data.get('depth')} 表达:{eval_data.get('expression')}"
        )

        return json.dumps(eval_data, ensure_ascii=False, indent=2)

    except json.JSONDecodeError as e:
        error_result = {
            "error": f"LLM 输出无法解析为 JSON: {str(e)}",
            "completeness": 5,
            "accuracy": 5,
            "depth": 5,
            "expression": 5,
            "overall_score": 5.0,
            "feedback": "评估解析失败，已使用默认中等分数",
            "follow_up_suggestion": "请重新作答并提供更详细的解释",
        }
        return json.dumps(error_result, ensure_ascii=False)

    except Exception as e:
        error_msg = f"评估失败: {type(e).__name__}: {str(e)}"
        print(f"[Evaluator] ❌ {error_msg}")
        return json.dumps({"error": error_msg}, ensure_ascii=False)
