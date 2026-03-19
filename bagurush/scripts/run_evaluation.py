#!/usr/bin/env python3
"""
RAGAS 评测脚本。

使用预标注的 Golden QA 对评测 RAG pipeline 质量。
评测指标：faithfulness / context_precision / context_recall / answer_relevancy。

用法：
  python scripts/run_evaluation.py --output results.json
  python scripts/run_evaluation.py --dataset scripts/golden_qa.json --output results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv()


def _load_golden_dataset(dataset_path: str) -> list:
    """加载 Golden QA 数据集。"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _retrieve_contexts(question: str, k: int = 5) -> list:
    """使用 HybridRetriever 检索参考上下文。"""
    try:
        from tools.knowledge_rag import _get_hybrid_retriever, _get_tech_store
        hybrid = _get_hybrid_retriever()
        if hybrid:
            docs = hybrid.retrieve(question, final_k=k)
        else:
            store = _get_tech_store()
            docs = store.search(question, k=k)
        return [doc.page_content for doc in docs]
    except Exception as e:
        print(f"  [WARN] 检索失败: {e}")
        return []


def _generate_answer(question: str, contexts: list) -> str:
    """使用 LLM 基于检索上下文生成答案。"""
    from utils.llm_config import get_llm

    llm = get_llm(temperature=0.1)
    context_text = "\n\n".join(contexts[:3])
    prompt = f"""基于以下参考资料回答问题。

参考资料：
{context_text}

问题：{question}

请直接回答，简洁准确。"""

    response = llm.invoke(prompt)
    return response.content


def run_evaluation(dataset_path: str, output_path: str):
    """执行 RAGAS 评测流程。"""
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from datasets import Dataset

    golden_data = _load_golden_dataset(dataset_path)
    print(f"[Eval] 加载 {len(golden_data)} 条 Golden QA 对")

    # 构建评测数据
    eval_records = []
    for i, item in enumerate(golden_data):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"  [{i+1}/{len(golden_data)}] {question[:50]}...")

        contexts = _retrieve_contexts(question)
        answer = _generate_answer(question, contexts)

        eval_records.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    # 转为 RAGAS 格式
    eval_dataset = Dataset.from_dict({
        "question": [r["question"] for r in eval_records],
        "answer": [r["answer"] for r in eval_records],
        "contexts": [r["contexts"] for r in eval_records],
        "ground_truth": [r["ground_truth"] for r in eval_records],
    })

    print("\n[Eval] 开始 RAGAS 评测...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
    )

    # 汇总
    scores = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
    print("\n[Eval] 评测结果:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print(f"\n[Eval] 结果已保存至: {output_path}")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS 评测脚本")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(_ROOT / "scripts" / "golden_qa.json"),
        help="Golden QA 数据集路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="评测结果输出路径",
    )
    args = parser.parse_args()
    run_evaluation(args.dataset, args.output)
