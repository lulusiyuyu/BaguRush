#!/usr/bin/env python3
"""
质量门禁检查脚本。

读取 RAGAS 评测结果 JSON，检查各指标是否达到阈值。
任一指标不达标则 exit(1)，用于阻止 CI/CD 合并。

用法：
  python scripts/quality_gate.py --results results.json
"""

import argparse
import json
import sys

QUALITY_THRESHOLDS = {
    "faithfulness": 0.85,
    "context_precision": 0.80,
    "context_recall": 0.75,
    "answer_relevancy": 0.80,
}


def check_quality_gate(results: dict) -> bool:
    """质量门禁检查，任一指标低于阈值则拒绝。"""
    passed = True
    print("=" * 50)
    print("  RAG 质量门禁检查")
    print("=" * 50)

    for metric, threshold in QUALITY_THRESHOLDS.items():
        score = results.get(metric, 0)
        status = "PASS" if score >= threshold else "BLOCKED"
        marker = "+" if score >= threshold else "-"
        print(f"  [{marker}] {metric}: {score:.3f} (阈值: {threshold}) {status}")
        if score < threshold:
            passed = False

    print("=" * 50)
    if not passed:
        print("  质量门禁未通过！请检查最近改动。")
        return False

    print("  质量门禁通过！")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 质量门禁")
    parser.add_argument("--results", type=str, required=True, help="RAGAS 评测结果 JSON")
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not check_quality_gate(results):
        sys.exit(1)
