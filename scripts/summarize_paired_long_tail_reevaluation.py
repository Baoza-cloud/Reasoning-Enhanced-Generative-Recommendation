from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from regenrec.metrics import mean_confidence_interval


METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "recall_at_5": ("ranking", "recall_at_5"),
    "ndcg_at_5": ("ranking", "ndcg_at_5"),
    "recall_at_10": ("ranking", "recall_at_10"),
    "ndcg_at_10": ("ranking", "ndcg_at_10"),
    "head_recall_at_5": ("target_groups", "head", "recall_at_5"),
    "head_recall_at_10": ("target_groups", "head", "recall_at_10"),
    "torso_recall_at_5": ("target_groups", "torso", "recall_at_5"),
    "torso_recall_at_10": ("target_groups", "torso", "recall_at_10"),
    "observed_tail_recall_at_5": (
        "target_groups",
        "observed_tail",
        "recall_at_5",
    ),
    "observed_tail_recall_at_10": (
        "target_groups",
        "observed_tail",
        "recall_at_10",
    ),
    "cold_start_recall_at_5": ("target_groups", "cold_start", "recall_at_5"),
    "cold_start_recall_at_10": ("target_groups", "cold_start", "recall_at_10"),
    "catalog_coverage": ("coverage", "catalog_coverage"),
    "head_catalog_coverage": (
        "strata_coverage",
        "strata",
        "head",
        "catalog_coverage",
    ),
    "torso_catalog_coverage": (
        "strata_coverage",
        "strata",
        "torso",
        "catalog_coverage",
    ),
    "observed_tail_catalog_coverage": (
        "strata_coverage",
        "strata",
        "observed_tail",
        "catalog_coverage",
    ),
    "cold_start_catalog_coverage": (
        "strata_coverage",
        "strata",
        "cold_start",
        "catalog_coverage",
    ),
    "head_recommendation_share": (
        "strata_coverage",
        "strata",
        "head",
        "recommendation_share",
    ),
    "torso_recommendation_share": (
        "strata_coverage",
        "strata",
        "torso",
        "recommendation_share",
    ),
    "observed_tail_recommendation_share": (
        "strata_coverage",
        "strata",
        "observed_tail",
        "recommendation_share",
    ),
    "cold_start_recommendation_share": (
        "strata_coverage",
        "strata",
        "cold_start",
        "recommendation_share",
    ),
    "unique_recommended_items": ("coverage", "unique_recommended_items"),
    "milliseconds_per_example": ("efficiency", "milliseconds_per_example"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired three-seed long-tail comparison under one evaluation harness"
    )
    parser.add_argument(
        "--baseline-analysis",
        action="append",
        required=True,
        type=Path,
        help="Original V2 analysis JSON; provide validation and test for every seed",
    )
    parser.add_argument(
        "--treatment-analysis",
        action="append",
        required=True,
        type=Path,
        help="Token-corrected analysis JSON; provide validation and test for every seed",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def nested_float(payload: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = payload
    for component in path:
        value = value[component]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric at {'.'.join(path)}")
    return result


def load_family(paths: list[Path]) -> dict[int, dict[str, dict[str, Any]]]:
    family: dict[int, dict[str, dict[str, Any]]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("synthetic_data_used") is not False:
            raise ValueError(f"analysis is not verified real data: {path}")
        seed = int(payload["seed"])
        split = str(payload["split"])
        if split not in {"validation", "test"}:
            raise ValueError(f"unsupported split in {path}: {split}")
        if split in family.setdefault(seed, {}):
            raise ValueError(f"duplicate seed/split analysis: seed={seed} split={split}")
        payload["_provenance"] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "model_sha256": payload["model_sha256"],
            "catalog_sha256": payload["catalog_sha256"],
        }
        family[seed][split] = payload
    for seed, splits in family.items():
        if set(splits) != {"validation", "test"}:
            raise ValueError(f"seed {seed} must contain validation and test analyses")
    if len(family) < 2:
        raise ValueError("at least two independent seeds are required")
    return family


def select_result(analysis: dict[str, Any], diversity_weight: float) -> dict[str, Any]:
    matches = [
        result
        for result in analysis["results"]
        if int(result["beam_size"]) == 20
        and int(result["candidate_count"]) == 20
        and int(result["return_count"]) == 10
        and math.isclose(
            float(result["diversity_weight"]), diversity_weight, abs_tol=1e-12
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one Beam-20/MMR={diversity_weight} result, found {len(matches)}"
        )
    return matches[0]


def summarize_model(
    family: dict[int, dict[str, dict[str, Any]]],
    *,
    diversity_weight: float,
) -> dict[str, Any]:
    seeds = sorted(family)
    output: dict[str, Any] = {
        "decoder": {
            "beam_size": 20,
            "candidate_count": 20,
            "return_count": 10,
            "semantic_mmr_weight": diversity_weight,
        },
        "provenance": [
            family[seed][split]["_provenance"]
            for seed in seeds
            for split in ("validation", "test")
        ],
    }
    for split in ("validation", "test"):
        output[split] = {
            metric: mean_confidence_interval(
                [
                    nested_float(
                        select_result(family[seed][split], diversity_weight), path
                    )
                    for seed in seeds
                ]
            )
            for metric, path in METRIC_PATHS.items()
        }
    return output


def paired_comparison(
    treatment: dict[int, dict[str, dict[str, Any]]],
    control: dict[int, dict[str, dict[str, Any]]],
    *,
    treatment_weight: float,
    control_weight: float,
    treatment_name: str,
    control_name: str,
) -> dict[str, Any]:
    if set(treatment) != set(control):
        raise ValueError("paired comparison requires identical seeds")
    seeds = sorted(treatment)
    output: dict[str, Any] = {
        "treatment": treatment_name,
        "control": control_name,
        "interval": "two-sided 95% Student-t CI over paired per-seed differences",
        "interpretation": (
            "A stable difference requires the paired 95% CI to exclude zero. "
            "Lower latency is better; higher values are better for all other metrics."
        ),
    }
    for split in ("validation", "test"):
        split_output: dict[str, Any] = {}
        for metric, path in METRIC_PATHS.items():
            treatment_values = [
                nested_float(
                    select_result(treatment[seed][split], treatment_weight), path
                )
                for seed in seeds
            ]
            control_values = [
                nested_float(select_result(control[seed][split], control_weight), path)
                for seed in seeds
            ]
            deltas = [
                treatment_value - control_value
                for treatment_value, control_value in zip(
                    treatment_values, control_values, strict=True
                )
            ]
            treatment_mean = sum(treatment_values) / len(treatment_values)
            control_mean = sum(control_values) / len(control_values)
            interval = mean_confidence_interval(deltas)
            split_output[metric] = {
                "treatment_mean": treatment_mean,
                "control_mean": control_mean,
                "relative_delta_percent": (
                    100.0 * (treatment_mean - control_mean) / control_mean
                    if control_mean != 0.0
                    else None
                ),
                "paired_delta": interval,
                "stable_difference": not (
                    float(interval["ci95_low"]) <= 0.0 <= float(interval["ci95_high"])
                ),
                "direction": (
                    "lower_is_better"
                    if metric == "milliseconds_per_example"
                    else "higher_is_better"
                ),
            }
        output[split] = split_output
    return output


def main() -> None:
    args = parse_args()
    baseline = load_family(args.baseline_analysis)
    treatment = load_family(args.treatment_analysis)
    if set(baseline) != set(treatment):
        raise ValueError("baseline and treatment must contain identical seeds")
    models = {
        "original_v2_mmr000": summarize_model(baseline, diversity_weight=0.0),
        "original_v2_mmr010": summarize_model(baseline, diversity_weight=0.1),
        "token_loss_010_mmr000": summarize_model(treatment, diversity_weight=0.0),
        "token_loss_010_mmr010": summarize_model(treatment, diversity_weight=0.1),
    }
    comparisons = {
        "training_effect_no_mmr": paired_comparison(
            treatment,
            baseline,
            treatment_weight=0.0,
            control_weight=0.0,
            treatment_name="token_loss_010_mmr000",
            control_name="original_v2_mmr000",
        ),
        "training_effect_same_decoder": paired_comparison(
            treatment,
            baseline,
            treatment_weight=0.1,
            control_weight=0.1,
            treatment_name="token_loss_010_mmr010",
            control_name="original_v2_mmr010",
        ),
        "full_pipeline_vs_original": paired_comparison(
            treatment,
            baseline,
            treatment_weight=0.1,
            control_weight=0.0,
            treatment_name="token_loss_010_mmr010",
            control_name="original_v2_mmr000",
        ),
        "mmr_effect_on_original_v2": paired_comparison(
            baseline,
            baseline,
            treatment_weight=0.1,
            control_weight=0.0,
            treatment_name="original_v2_mmr010",
            control_name="original_v2_mmr000",
        ),
        "mmr_effect_on_token_corrected": paired_comparison(
            treatment,
            treatment,
            treatment_weight=0.1,
            control_weight=0.0,
            treatment_name="token_loss_010_mmr010",
            control_name="token_loss_010_mmr000",
        ),
    }
    report = {
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "synthetic_data_used": False,
        "seeds": sorted(baseline),
        "evaluation_protocol": (
            "One shared full-split evaluation harness for both model families; Beam-20, "
            "20 candidates, top-10 return, MMR weights 0 and 0.10."
        ),
        "models": models,
        "paired_comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
