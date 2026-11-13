from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SEEDS = (2026, 2027, 2028)
FIXED_VARIANT = "b0.5_tail8_preserve4_scale1"
T_CRITICAL_DF2_975 = 4.302652729911275
METRICS = {
    "recall_at_5": ("ranking", "recall_at_5"),
    "ndcg_at_5": ("ranking", "ndcg_at_5"),
    "recall_at_10": ("ranking", "recall_at_10"),
    "ndcg_at_10": ("ranking", "ndcg_at_10"),
    "observed_tail_recall_at_10": (
        "target_groups",
        "observed_tail",
        "recall_at_10",
    ),
    "catalog_coverage_at_10": ("catalog", "coverage_at_10"),
    "observed_tail_catalog_coverage_at_10": (
        "catalog",
        "observed_tail_coverage_at_10",
    ),
    "observed_tail_recommendation_share_at_10": (
        "catalog",
        "observed_tail_recommendation_share_at_10",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize frozen V2.5 confirmation over seeds 2026/2027/2028"
    )
    parser.add_argument(
        "--strict-report",
        action="append",
        required=True,
        help="SEED=path/to/strict_gate_report.json",
    )
    parser.add_argument(
        "--validation-report",
        action="append",
        required=True,
        help="SEED=path/to/validation_cache/report.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_seed_paths(values: list[str]) -> dict[int, Path]:
    output: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"expected SEED=PATH, got {value}")
        seed = int(seed_text)
        if seed in output:
            raise ValueError(f"duplicate seed path: {seed}")
        output[seed] = Path(path_text)
    if set(output) != set(SEEDS):
        raise ValueError(f"exactly seeds {SEEDS} are required")
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested(payload: dict[str, object], path: tuple[str, ...]) -> float:
    value: object = payload
    for key in path:
        value = value[key]
    return float(value)


def student_t(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) != 3:
        raise ValueError("the preregistered Student-t summary requires three seeds")
    mean = float(array.mean())
    standard_deviation = float(array.std(ddof=1))
    margin = T_CRITICAL_DF2_975 * standard_deviation / math.sqrt(len(array))
    return {
        "n": len(array),
        "values": [float(value) for value in array],
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "critical_value": T_CRITICAL_DF2_975,
        "degrees_of_freedom": 2,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_markdown(path: Path, report: dict[str, object]) -> None:
    rows = []
    for seed in SEEDS:
        current = report["seeds"][str(seed)]
        rows.append(
            "| {seed} | {base:.6f} | {candidate:.6f} | {tail_base:.6f} | "
            "{tail_candidate:.6f} | {coverage_base:.4%} | {coverage_candidate:.4%} |".format(
                seed=seed,
                base=current["baseline"]["recall_at_10"],
                candidate=current["candidate"]["recall_at_10"],
                tail_base=current["baseline"]["observed_tail_recall_at_10"],
                tail_candidate=current["candidate"]["observed_tail_recall_at_10"],
                coverage_base=current["baseline"][
                    "observed_tail_catalog_coverage_at_10"
                ],
                coverage_candidate=current["candidate"][
                    "observed_tail_catalog_coverage_at_10"
                ],
            )
        )
    tail = report["student_t"]["observed_tail_recall_at_10_delta"]
    overall = report["student_t"]["recall_at_10_delta"]
    coverage = report["student_t"][
        "observed_tail_catalog_coverage_at_10_delta"
    ]
    content = "\n".join(
        [
            "# V2.5 三种子冻结确认",
            "",
            f"状态：{'通过' if report['acceptance']['passed'] else '未通过'}。Test 未参与。",
            "",
            "| Seed | Baseline R@10 | V2.5 R@10 | Baseline Tail R@10 | V2.5 Tail R@10 | Baseline Tail Coverage | V2.5 Tail Coverage |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"- 总体 R@10 delta：{overall['mean']:+.6f}，95% Student-t CI "
            f"[{overall['ci95_low']:+.6f}, {overall['ci95_high']:+.6f}]。",
            f"- Observed-Tail R@10 delta：{tail['mean']:+.6f}，95% Student-t CI "
            f"[{tail['ci95_low']:+.6f}, {tail['ci95_high']:+.6f}]。",
            f"- Observed-Tail coverage delta：{coverage['mean']:+.6f}，95% Student-t CI "
            f"[{coverage['ci95_low']:+.6f}, {coverage['ci95_high']:+.6f}]。",
            "",
            "固定配置：bound=0.5，tail weight=8，preserve weight=4，deployment scale=1.0。",
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    strict_paths = parse_seed_paths(args.strict_report)
    validation_paths = parse_seed_paths(args.validation_report)
    seed_results: dict[str, object] = {}
    metric_deltas = {name: [] for name in METRICS}
    incremental_latencies: list[float] = []
    provenance: dict[str, object] = {}
    for seed in SEEDS:
        strict_path = strict_paths[seed]
        validation_path = validation_paths[seed]
        strict = json.loads(strict_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if strict.get("test_evaluated") is not False or validation.get(
            "test_evaluated"
        ) is not False:
            raise ValueError(f"seed {seed} improperly evaluated Test")
        if int(strict.get("seed")) != seed or int(validation.get("seed")) != seed:
            raise ValueError(f"seed provenance mismatch for {seed}")
        if FIXED_VARIANT not in strict["variants"]:
            raise ValueError(f"seed {seed} is missing frozen variant {FIXED_VARIANT}")
        variant = strict["variants"][FIXED_VARIANT]
        hyperparameters = variant["hyperparameters"]
        expected = {
            "training_bound": 0.5,
            "deployment_scale": 1.0,
            "effective_bound": 0.5,
            "tail_positive_weight": 8.0,
            "preserve_weight": 4.0,
            "distill_weight": 0.2,
            "gate_supervision_weight": 1.0,
            "gate_positive_weight": 20.0,
        }
        for key, expected_value in expected.items():
            if not math.isclose(float(hyperparameters[key]), expected_value):
                raise ValueError(f"seed {seed} changed frozen hyperparameter {key}")
        baseline_values = {
            name: nested(strict["baseline"], path) for name, path in METRICS.items()
        }
        candidate_values = {
            name: nested(variant["metrics"], path) for name, path in METRICS.items()
        }
        deltas = {
            name: candidate_values[name] - baseline_values[name] for name in METRICS
        }
        for name, delta in deltas.items():
            metric_deltas[name].append(delta)
        examples = int(validation["examples"])
        retrieval_ms = 1000.0 * (
            float(validation["timing"]["profile_seconds"])
            + float(validation["timing"]["retrieval_seconds"])
        ) / examples
        incremental_latency = retrieval_ms + float(
            variant["metrics"]["ranking_milliseconds_per_example"]
        )
        incremental_latencies.append(incremental_latency)
        seed_results[str(seed)] = {
            "baseline": baseline_values,
            "candidate": candidate_values,
            "delta": deltas,
            "incremental_latency_ms_per_example": incremental_latency,
            "single_seed_paired_tests": variant["paired_tests"],
            "passed_provisional_single_seed_gates": variant[
                "passed_provisional_single_seed_gates"
            ],
        }
        provenance[str(seed)] = {
            "strict_report": str(strict_path.resolve()),
            "strict_report_sha256": sha256_file(strict_path),
            "validation_report": str(validation_path.resolve()),
            "validation_report_sha256": sha256_file(validation_path),
        }
    intervals = {
        f"{name}_delta": student_t(values) for name, values in metric_deltas.items()
    }
    intervals["incremental_latency_ms_per_example"] = student_t(
        incremental_latencies
    )
    mean_baseline_r10 = float(
        np.mean([seed_results[str(seed)]["baseline"]["recall_at_10"] for seed in SEEDS])
    )
    mean_candidate_r10 = float(
        np.mean([seed_results[str(seed)]["candidate"]["recall_at_10"] for seed in SEEDS])
    )
    checks = {
        "mean_recall_at_10_retention_ge_0_99": (
            mean_candidate_r10 / mean_baseline_r10 >= 0.99
        ),
        "overall_recall_at_10_delta_ci_low_ge_minus_0_0004": (
            intervals["recall_at_10_delta"]["ci95_low"] >= -0.0004
        ),
        "ndcg_at_10_not_stably_lower": (
            intervals["ndcg_at_10_delta"]["ci95_high"] >= 0.0
        ),
        "observed_tail_recall_at_10_delta_ci_low_gt_0": (
            intervals["observed_tail_recall_at_10_delta"]["ci95_low"] > 0.0
        ),
        "observed_tail_coverage_delta_ci_low_gt_0": (
            intervals["observed_tail_catalog_coverage_at_10_delta"]["ci95_low"]
            > 0.0
        ),
        "incremental_latency_mean_le_5ms": (
            intervals["incremental_latency_ms_per_example"]["mean"] <= 5.0
        ),
    }
    report = {
        "status": "completed",
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "category": "Musical_Instruments",
        "split": "validation",
        "test_evaluated": False,
        "synthetic_data_used": False,
        "seeds": seed_results,
        "frozen_variant": FIXED_VARIANT,
        "student_t": intervals,
        "mean_recall_at_10_retention": mean_candidate_r10 / mean_baseline_r10,
        "acceptance": {"checks": checks, "passed": all(checks.values())},
        "provenance": provenance,
    }
    output_dir = args.output_dir
    write_json(output_dir / "report.json", report)
    write_markdown(output_dir / "REPORT.md", report)
    write_json(
        output_dir / "state.json",
        {
            "status": "completed",
            "test_evaluated": False,
            "acceptance_passed": report["acceptance"]["passed"],
        },
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "acceptance_passed": report["acceptance"]["passed"],
                "report": str((output_dir / "report.json").resolve()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
