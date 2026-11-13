from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from regenrec.metrics import mean_confidence_interval


METRICS = ("recall_at_5", "ndcg_at_5", "recall_at_10", "ndcg_at_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate independent Amazon recommendation runs with 95% t intervals"
    )
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", default="generative_top10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports: list[dict[str, object]] = []
    seeds: set[int] = set()
    for path in args.report:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        if seed in seeds:
            raise ValueError(f"duplicate seed {seed}; confidence intervals require independent runs")
        seeds.add(seed)
        audit = payload.get("data_audit", {})
        synthetic = payload.get("synthetic_data_used", audit.get("synthetic_data_used"))
        if synthetic is not False:
            raise ValueError(f"report is not verified real data: {path}")
        for split in ("validation", "test"):
            for metric in METRICS:
                value = float(payload[split][metric])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {split}.{metric} in {path}")
        reports.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "seed": seed,
                "validation": payload["validation"],
                "test": payload["test"],
            }
        )

    summary: dict[str, object] = {
        "model": args.model_name,
        "interval": "two-sided 95% Student-t confidence interval across independent seeds",
        "seeds": sorted(seeds),
        "reports": reports,
    }
    for split in ("validation", "test"):
        summary[split] = {
            metric: mean_confidence_interval(
                [float(report[split][metric]) for report in reports]  # type: ignore[index]
            )
            for metric in METRICS
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
