from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from regenrec.metrics import mean_confidence_interval


METRICS = ("recall_at_5", "ndcg_at_5", "recall_at_10", "ndcg_at_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two multi-seed reports with paired 95% Student-t intervals"
    )
    parser.add_argument("--left-summary", required=True, type=Path)
    parser.add_argument("--right-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--left-name")
    parser.add_argument("--right-name")
    return parser.parse_args()


def reports_by_seed(summary: dict[str, object]) -> dict[int, dict[str, object]]:
    reports = summary.get("reports")
    if not isinstance(reports, list):
        raise ValueError("summary.reports must be a list")
    indexed: dict[int, dict[str, object]] = {}
    for report in reports:
        if not isinstance(report, dict):
            raise ValueError("each summary report must be an object")
        seed = int(report["seed"])
        if seed in indexed:
            raise ValueError(f"duplicate seed in summary: {seed}")
        indexed[seed] = report
    return indexed


def metric_value(report: dict[str, object], split: str, metric: str) -> float:
    split_payload = report.get(split)
    if not isinstance(split_payload, dict):
        raise ValueError(f"missing report split: {split}")
    value = float(split_payload[metric])
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric: {split}.{metric}")
    return value


def main() -> None:
    args = parse_args()
    left = json.loads(args.left_summary.read_text(encoding="utf-8"))
    right = json.loads(args.right_summary.read_text(encoding="utf-8"))
    left_reports = reports_by_seed(left)
    right_reports = reports_by_seed(right)
    left_seeds = set(left_reports)
    right_seeds = set(right_reports)
    if left_seeds != right_seeds:
        raise ValueError(
            f"paired comparison requires identical seeds: left={sorted(left_seeds)}, "
            f"right={sorted(right_seeds)}"
        )
    if len(left_seeds) < 2:
        raise ValueError("paired Student-t intervals require at least two seeds")

    seeds = sorted(left_seeds)
    left_name = args.left_name or str(left.get("model", "left"))
    right_name = args.right_name or str(right.get("model", "right"))
    output: dict[str, object] = {
        "comparison": f"{left_name} minus {right_name}",
        "interval": "two-sided 95% Student-t confidence interval across paired seeds",
        "seeds": seeds,
        "interpretation_rule": (
            "claim a stable positive advantage only when the paired absolute-difference "
            "95% interval is strictly above zero"
        ),
        "source_summaries": {
            "left": str(args.left_summary.resolve()),
            "right": str(args.right_summary.resolve()),
        },
    }
    for split in ("validation", "test"):
        split_output: dict[str, object] = {}
        for metric in METRICS:
            left_values = [
                metric_value(left_reports[seed], split, metric) for seed in seeds
            ]
            right_values = [
                metric_value(right_reports[seed], split, metric) for seed in seeds
            ]
            absolute_differences = [
                left_value - right_value
                for left_value, right_value in zip(left_values, right_values, strict=True)
            ]
            relative_differences = [
                100.0 * (left_value / right_value - 1.0)
                for left_value, right_value in zip(left_values, right_values, strict=True)
            ]
            absolute_interval = mean_confidence_interval(absolute_differences)
            relative_interval = mean_confidence_interval(relative_differences)
            split_output[metric] = {
                "left_values": left_values,
                "right_values": right_values,
                "paired_absolute_difference": absolute_interval,
                "paired_relative_difference_percent": relative_interval,
                "stable_positive_advantage": absolute_interval["ci95_low"] > 0.0,
                "stable_negative_difference": absolute_interval["ci95_high"] < 0.0,
            }
        output[split] = split_output

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
