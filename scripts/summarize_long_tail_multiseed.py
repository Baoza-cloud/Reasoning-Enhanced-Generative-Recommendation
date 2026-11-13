from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from regenrec.metrics import mean_confidence_interval


METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "recall_at_5": ("recall_at_5",),
    "ndcg_at_5": ("ndcg_at_5",),
    "recall_at_10": ("recall_at_10",),
    "ndcg_at_10": ("ndcg_at_10",),
    "head_recall_at_5": ("target_groups", "head", "recall_at_5"),
    "head_ndcg_at_5": ("target_groups", "head", "ndcg_at_5"),
    "head_recall_at_10": ("target_groups", "head", "recall_at_10"),
    "head_ndcg_at_10": ("target_groups", "head", "ndcg_at_10"),
    "tail_recall_at_5": ("target_groups", "tail", "recall_at_5"),
    "tail_ndcg_at_5": ("target_groups", "tail", "ndcg_at_5"),
    "tail_recall_at_10": ("target_groups", "tail", "recall_at_10"),
    "tail_ndcg_at_10": ("target_groups", "tail", "ndcg_at_10"),
    "catalog_coverage": ("coverage", "catalog_coverage"),
    "tail_catalog_coverage": ("coverage", "tail_catalog_coverage"),
    "tail_recommendation_share": ("coverage", "tail_recommendation_share"),
    "milliseconds_per_example": ("efficiency", "milliseconds_per_example"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate long-tail recommendation metrics with Student-t intervals"
    )
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--baseline-report", action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", default="long_tail_generative")
    return parser.parse_args()


def nested_float(payload: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = payload
    for component in path:
        value = value[component]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric at {'.'.join(path)}")
    return result


def load_reports(paths: list[Path], *, require_long_tail: bool) -> dict[int, dict[str, Any]]:
    loaded: dict[int, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        if seed in loaded:
            raise ValueError(f"duplicate seed {seed}")
        audit = payload.get("data_audit", {})
        synthetic = payload.get("synthetic_data_used", audit.get("synthetic_data_used"))
        if synthetic is not False:
            raise ValueError(f"report is not verified real data: {path}")
        for split in ("validation", "test"):
            paths_to_check = METRIC_PATHS if require_long_tail else {
                name: metric_path
                for name, metric_path in METRIC_PATHS.items()
                if len(metric_path) == 1
            }
            for metric_path in paths_to_check.values():
                nested_float(payload[split], metric_path)
        loaded[seed] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed": seed,
            "validation": payload["validation"],
            "test": payload["test"],
        }
    if len(loaded) < 2:
        raise ValueError("at least two independent seeds are required")
    return loaded


def main() -> None:
    args = parse_args()
    reports = load_reports(args.report, require_long_tail=True)
    summary: dict[str, Any] = {
        "model": args.model_name,
        "interval": "two-sided 95% Student-t confidence interval across independent seeds",
        "seeds": sorted(reports),
        "reports": [reports[seed] for seed in sorted(reports)],
    }
    for split in ("validation", "test"):
        summary[split] = {
            metric: mean_confidence_interval(
                [nested_float(reports[seed][split], path) for seed in sorted(reports)]
            )
            for metric, path in METRIC_PATHS.items()
        }

    if args.baseline_report:
        baselines = load_reports(args.baseline_report, require_long_tail=False)
        if set(baselines) != set(reports):
            raise ValueError("baseline and treatment reports must use identical seeds")
        paired: dict[str, Any] = {
            "baseline_reports": [baselines[seed] for seed in sorted(baselines)],
            "interpretation": (
                "A stable gain requires the paired 95% CI to exclude zero; otherwise the "
                "result is inconclusive across these seeds."
            ),
        }
        flat_metrics = {
            name: path for name, path in METRIC_PATHS.items() if len(path) == 1
        }
        for split in ("validation", "test"):
            paired[split] = {
                metric: mean_confidence_interval(
                    [
                        nested_float(reports[seed][split], path)
                        - nested_float(baselines[seed][split], path)
                        for seed in sorted(reports)
                    ]
                )
                for metric, path in flat_metrics.items()
            }
        summary["paired_delta_vs_original_v2"] = paired

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
