#!/usr/bin/env python3
"""
Agent Benchmark 评测脚本 —— 五维指标评估面试官 Agent 打分质量。

指标：
  1. 区分度（Discrimination）: excellent 与 poor 的平均分差 > 4.0
  2. 一致性（Consistency）:    同一答案多次评估的标准差 < 1.0
  3. 校准度（Calibration）:    Agent 分 vs 人工分的 Spearman 相关 > 0.8
  4. 排序正确率（Ordering）:   excellent > mediocre > poor 的比例 > 95%
  5. 鲁棒性（Robustness）:     暂留（需要同义改写数据，后续扩展）

用法：
  python scripts/agent_benchmark.py
  python scripts/agent_benchmark.py --dataset scripts/golden_dataset.json --repeats 3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _evaluate_single(question: str, answer: str) -> float:
    """调用 evaluate_answer tool 获取综合分。"""
    from tools.answer_evaluator import evaluate_answer

    result_str = evaluate_answer.invoke({
        "question": question,
        "answer": answer,
        "reference": "",
    })
    try:
        result = json.loads(result_str)
        return float(result.get("overall_score", 5.0))
    except (json.JSONDecodeError, TypeError):
        return 5.0


def run_benchmark(dataset_path: str, repeats: int = 1, output_path: str = None):
    """执行 Agent Benchmark 评测。"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    print(f"[Benchmark] 加载 {len(golden_data)} 条 Golden 数据，重复 {repeats} 次")

    all_results = []
    for idx, item in enumerate(golden_data):
        q = item["question"]
        answers = item["answers"]
        human = item["human_scores"]
        print(f"\n[{idx+1}/{len(golden_data)}] {q[:50]}...")

        item_result = {"question": q, "scores": {}, "human_scores": human}
        for level in ["excellent", "mediocre", "poor"]:
            ans = answers.get(level, "")
            scores_for_level = []
            for r in range(repeats):
                score = _evaluate_single(q, ans)
                scores_for_level.append(score)
                if repeats > 1:
                    print(f"  {level} run{r+1}: {score:.1f}")
            item_result["scores"][level] = scores_for_level
            avg = np.mean(scores_for_level)
            print(f"  {level}: avg={avg:.2f} (human={human.get(level, '?')})")

        all_results.append(item_result)

    # 计算五维指标
    metrics = _compute_metrics(all_results)

    # 打印报告
    print("\n" + "=" * 60)
    print("  Agent Benchmark 评测报告")
    print("=" * 60)

    thresholds = {
        "discrimination": 4.0,
        "ordering_accuracy": 0.95,
        "calibration_spearman": 0.8,
        "consistency_std": 1.0,
    }

    for k, v in metrics.items():
        threshold = thresholds.get(k)
        if threshold is not None:
            if k == "consistency_std":
                status = "PASS" if v < threshold else "FAIL"
            else:
                status = "PASS" if v >= threshold else "FAIL"
            print(f"  {k}: {v:.4f} (阈值: {threshold}) [{status}]")
        else:
            print(f"  {k}: {v:.4f}")

    print("=" * 60)

    # 保存结果
    if output_path:
        report = {"metrics": metrics, "details": all_results}
        # Convert numpy types for JSON serialization
        def _convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=_convert)
        print(f"\n[Benchmark] 结果已保存至: {output_path}")

    return metrics


def _compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """从所有评测结果中计算五维指标。"""
    excellent_avgs = []
    mediocre_avgs = []
    poor_avgs = []
    agent_scores = []
    human_scores = []
    ordering_correct = 0
    ordering_total = 0
    all_stds = []

    for item in results:
        for level in ["excellent", "mediocre", "poor"]:
            scores = item["scores"].get(level, [])
            if scores:
                avg = np.mean(scores)
                if level == "excellent":
                    excellent_avgs.append(avg)
                elif level == "mediocre":
                    mediocre_avgs.append(avg)
                else:
                    poor_avgs.append(avg)

                agent_scores.append(avg)
                human_scores.append(item["human_scores"].get(level, 5.0))

                if len(scores) > 1:
                    all_stds.append(np.std(scores))

        # 排序正确性：excellent > mediocre > poor
        e_scores = item["scores"].get("excellent", [])
        m_scores = item["scores"].get("mediocre", [])
        p_scores = item["scores"].get("poor", [])
        if e_scores and m_scores and p_scores:
            e_avg = np.mean(e_scores)
            m_avg = np.mean(m_scores)
            p_avg = np.mean(p_scores)
            ordering_total += 1
            if e_avg > m_avg > p_avg:
                ordering_correct += 1

    # 1. 区分度
    discrimination = np.mean(excellent_avgs) - np.mean(poor_avgs) if excellent_avgs and poor_avgs else 0.0

    # 2. 排序正确率
    ordering_accuracy = ordering_correct / ordering_total if ordering_total > 0 else 0.0

    # 3. 校准度（Spearman 相关系数）
    from scipy.stats import spearmanr
    if len(agent_scores) >= 3:
        corr, _ = spearmanr(agent_scores, human_scores)
        calibration = corr if not np.isnan(corr) else 0.0
    else:
        calibration = 0.0

    # 4. 一致性（平均标准差）
    consistency_std = np.mean(all_stds) if all_stds else 0.0

    return {
        "discrimination": float(discrimination),
        "ordering_accuracy": float(ordering_accuracy),
        "calibration_spearman": float(calibration),
        "consistency_std": float(consistency_std),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent Benchmark 评测")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(_ROOT / "scripts" / "golden_dataset.json"),
        help="Golden Dataset 路径",
    )
    parser.add_argument("--repeats", type=int, default=1, help="每个答案重复评估次数（>1 才计算一致性）")
    parser.add_argument("--output", type=str, default=None, help="结果输出路径")
    args = parser.parse_args()

    run_benchmark(args.dataset, args.repeats, args.output)
