from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from scripts.train_v2_5_bounded_residual import (
    catalog_groups,
    evaluate,
    load_cache,
    make_features,
    paired_tests,
    score_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate every V2.5 Validation variant with strict paired gates"
    )
    parser.add_argument("--config", default="configs/v2_final_default.yaml")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--torch-threads", type=int, default=2)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0 or args.torch_threads <= 0:
        raise ValueError("bootstrap and thread settings must be positive")
    torch.set_num_threads(args.torch_threads)
    fit_report = json.loads(args.fit_report.read_text(encoding="utf-8"))
    if fit_report.get("test_evaluated") is not False:
        raise ValueError("strict V2.5 gating must not use Test")
    validation = load_cache(args.validation_cache, "validation")
    config = load_config(Path(args.config))
    catalog = SemanticIdCatalog.load(args.artifact_dir / "catalog.json")
    group_lookup = catalog_groups(config, catalog)
    candidate_features, row_features, base = make_features(validation)
    baseline_metrics, baseline_hits = evaluate(
        base, validation, group_lookup, catalog_size=len(catalog.item_ids)
    )
    target_groups = group_lookup[validation["target_catalog_indices"].long()]
    variants: dict[str, object] = {}
    passing: list[str] = []
    baseline_tail_coverage = baseline_metrics["catalog"][
        "observed_tail_coverage_at_10"
    ]
    for index, (name, stored) in enumerate(fit_report["variants"].items()):
        model = stored["model"]
        scores, gate, residual = score_candidates(
            candidate_features,
            row_features,
            base,
            torch.tensor(model["residual_weights"], dtype=torch.float32),
            torch.tensor(model["gate_weights"], dtype=torch.float32),
            torch.tensor(model["gate_bias"], dtype=torch.float32),
            bound=float(stored["hyperparameters"]["effective_bound"]),
        )
        metrics, hits = evaluate(
            scores, validation, group_lookup, catalog_size=len(catalog.item_ids)
        )
        paired = paired_tests(
            baseline_hits,
            hits,
            target_groups,
            samples=args.bootstrap_samples,
            seed=args.seed + index * 2,
        )
        overall = paired["paired_user_bootstrap_top10"]
        tail = paired["observed_tail_paired_user_bootstrap_top10"]
        checks = {
            "recall_at_10_retention_ge_0_99": (
                metrics["ranking"]["recall_at_10"]
                >= 0.99 * baseline_metrics["ranking"]["recall_at_10"]
            ),
            "ndcg_at_10_drop_le_0_0004": (
                metrics["ranking"]["ndcg_at_10"]
                >= baseline_metrics["ranking"]["ndcg_at_10"] - 0.0004
            ),
            "overall_recall_delta_ci_low_ge_minus_0_0004": (
                overall["ci95_low"] >= -0.0004
            ),
            "observed_tail_recall_delta_ci_low_gt_0": tail["ci95_low"] > 0.0,
            "observed_tail_coverage_point_delta_gt_0": (
                metrics["catalog"]["observed_tail_coverage_at_10"]
                > baseline_tail_coverage
            ),
            "ranking_latency_le_5ms": (
                metrics["ranking_milliseconds_per_example"] <= 5.0
            ),
        }
        passed = all(checks.values())
        if passed:
            passing.append(name)
        variants[name] = {
            "passed_provisional_single_seed_gates": passed,
            "checks": checks,
            "hyperparameters": stored["hyperparameters"],
            "metrics": metrics,
            "paired_tests": paired,
            "gate_mean": float(gate.mean()),
            "absolute_residual_mean": float(residual.abs().mean()),
        }
    selected = (
        max(
            passing,
            key=lambda name: (
                variants[name]["metrics"]["target_groups"]["observed_tail"][
                    "recall_at_10"
                ],
                variants[name]["metrics"]["ranking"]["recall_at_10"],
                variants[name]["metrics"]["ranking"]["ndcg_at_10"],
            ),
        )
        if passing
        else None
    )
    report = {
        "status": "completed",
        "split": "validation",
        "test_evaluated": False,
        "seed": args.seed,
        "baseline": baseline_metrics,
        "provisional_single_seed_gate_note": (
            "Tail coverage has only a point-delta check here; its required three-seed "
            "CI remains pending and no Test result is authorized."
        ),
        "passing_variants": passing,
        "selected_provisional_variant": selected,
        "selected_result": variants.get(selected) if selected else None,
        "variants": variants,
        "provenance": {
            "validation_cache_sha256": sha256_file(args.validation_cache),
            "fit_report_sha256": sha256_file(args.fit_report),
            "catalog_sha256": sha256_file(args.artifact_dir / "catalog.json"),
            "config_sha256": sha256_file(Path(args.config)),
        },
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": "completed",
                "passing_variants": len(passing),
                "selected_provisional_variant": selected,
                "output": str(args.output.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
