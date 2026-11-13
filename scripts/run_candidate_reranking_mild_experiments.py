from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "candidate_reranking_mild_v1"
SEEDS = (2026, 2027, 2028)
MINIMUM_RECALL_RATIO = 0.97


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only mild reranker selection followed by locked three-seed "
            "test evaluation"
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--linger", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def artifact_dir(seed: int) -> Path:
    return (
        ROOT
        / "artifacts"
        / "amazon2023_musical_instruments_long_tail_training_v1"
        / "formal"
        / "token_loss_010"
        / f"seed_{seed}"
    )


def parameter_key(parameters: dict[str, object]) -> tuple[float, float, str, float]:
    return (
        round(float(parameters["retrieval_weight"]), 8),
        round(float(parameters["frequency_alpha"]), 8),
        str(parameters["quota_profile"]),
        round(float(parameters["quota_strength"]), 8),
    )


def parameters_from_key(key: tuple[float, float, str, float]) -> dict[str, object]:
    return {
        "retrieval_weight": key[0],
        "frequency_alpha": key[1],
        "quota_profile": key[2],
        "quota_strength": key[3],
    }


def parameters_equal(left: dict[str, object], right: dict[str, object]) -> bool:
    return parameter_key(left) == parameter_key(right)


def report_is_complete(
    path: Path,
    *,
    seed: int,
    split: str,
    mode: str,
    parameters: dict[str, object] | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    valid = (
        report.get("dataset") == "McAuley-Lab/Amazon-Reviews-2023"
        and report.get("synthetic_data_used") is False
        and int(report.get("seed", -1)) == seed
        and report.get("split") == split
        and report.get("mode") == mode
        and report.get("sampling", {}).get("max_examples") is None
    )
    if mode == "tune":
        valid = valid and split == "validation" and report.get("sweep_profile") == "mild"
        valid = valid and report.get("selection", {}).get("test_used_for_selection") is False
    if parameters is not None:
        valid = valid and parameters_equal(
            report.get("selection", {}).get("selected_parameters", {}), parameters
        )
    return bool(valid)


def run_command(command: list[str], log_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = "src:."
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{utc_now()}] command={json.dumps(command)}\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with return code {completed.returncode}; log={log_path}"
        )


def validation_tune_command(output_dir: Path, seed: int) -> list[str]:
    return [
        sys.executable,
        "scripts/analyze_candidate_reranking.py",
        "--config",
        "configs/v2_amazon2023_top10.yaml",
        "--artifact-dir",
        str(artifact_dir(seed)),
        "--seed",
        str(seed),
        "--split",
        "validation",
        "--mode",
        "tune",
        "--sweep-profile",
        "mild",
        "--beam-size",
        "20",
        "--candidate-count",
        "20",
        "--retrieval-count",
        "50",
        "--top-k",
        "10",
        "--batch-size",
        "128",
        "--retrieval-batch-size",
        "256",
        "--minimum-recall-ratio",
        str(MINIMUM_RECALL_RATIO),
        "--log-every-batches",
        "50",
        "--output-dir",
        str(output_dir),
    ]


def test_command(
    output_dir: Path, seed: int, parameters: dict[str, object]
) -> list[str]:
    return [
        sys.executable,
        "scripts/analyze_candidate_reranking.py",
        "--config",
        "configs/v2_amazon2023_top10.yaml",
        "--artifact-dir",
        str(artifact_dir(seed)),
        "--seed",
        str(seed),
        "--split",
        "test",
        "--mode",
        "evaluate",
        "--beam-size",
        "20",
        "--candidate-count",
        "20",
        "--retrieval-count",
        "50",
        "--top-k",
        "10",
        "--batch-size",
        "128",
        "--retrieval-batch-size",
        "256",
        "--retrieval-weight",
        str(parameters["retrieval_weight"]),
        "--frequency-alpha",
        str(parameters["frequency_alpha"]),
        "--quota-profile",
        str(parameters["quota_profile"]),
        "--quota-strength",
        str(parameters["quota_strength"]),
        "--log-every-batches",
        "50",
        "--output-dir",
        str(output_dir),
    ]


def all_variants(report: dict[str, object]) -> dict[tuple[float, float, str, float], dict[str, object]]:
    variants = (
        list(report["stage_a_fusion_frequency_sweep"])
        + list(report["stage_b_soft_quota_sweep"])
    )
    output: dict[tuple[float, float, str, float], dict[str, object]] = {}
    for variant in variants:
        key = parameter_key(variant["parameters"])
        if key in output:
            raise ValueError(f"duplicate validation parameter tuple: {key}")
        output[key] = variant
    return output


def locked_validation_selection(
    reports: dict[int, dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    variants = {seed: all_variants(report) for seed, report in reports.items()}
    shared = set.intersection(*(set(seed_variants) for seed_variants in variants.values()))
    if not shared:
        raise RuntimeError("validation sweeps have no shared parameter configurations")
    candidates: list[dict[str, object]] = []
    for key in sorted(shared):
        per_seed: dict[str, object] = {}
        tail_recalls: list[float] = []
        tail_ndcgs: list[float] = []
        overall_recalls: list[float] = []
        overall_ndcgs: list[float] = []
        tail_coverages: list[float] = []
        retentions: list[float] = []
        for seed in SEEDS:
            report = reports[seed]
            result = variants[seed][key]
            baseline_r10 = float(report["baseline"]["metrics"]["ranking"]["recall_at_10"])
            ranking = result["metrics"]["ranking"]
            tail = result["metrics"]["target_groups"]["observed_tail"]
            coverage = result["metrics"]["strata_coverage"]["strata"]["observed_tail"]
            retention = float(ranking["recall_at_10"]) / baseline_r10
            retentions.append(retention)
            tail_recalls.append(float(tail["recall_at_10"]))
            tail_ndcgs.append(float(tail["ndcg_at_10"]))
            overall_recalls.append(float(ranking["recall_at_10"]))
            overall_ndcgs.append(float(ranking["ndcg_at_10"]))
            tail_coverages.append(float(coverage["catalog_coverage"]))
            per_seed[str(seed)] = {
                "baseline_recall_at_10": baseline_r10,
                "recall_at_10": float(ranking["recall_at_10"]),
                "recall_retention": retention,
                "ndcg_at_10": float(ranking["ndcg_at_10"]),
                "observed_tail_recall_at_10": float(tail["recall_at_10"]),
                "observed_tail_ndcg_at_10": float(tail["ndcg_at_10"]),
                "observed_tail_catalog_coverage": float(coverage["catalog_coverage"]),
            }
        feasible = min(retentions) >= MINIMUM_RECALL_RATIO
        candidates.append(
            {
                "parameters": parameters_from_key(key),
                "feasible": feasible,
                "minimum_per_seed_recall_retention": min(retentions),
                "mean_recall_retention": statistics.mean(retentions),
                "mean_recall_at_10": statistics.mean(overall_recalls),
                "mean_ndcg_at_10": statistics.mean(overall_ndcgs),
                "mean_observed_tail_recall_at_10": statistics.mean(tail_recalls),
                "mean_observed_tail_ndcg_at_10": statistics.mean(tail_ndcgs),
                "mean_observed_tail_catalog_coverage": statistics.mean(tail_coverages),
                "per_seed": per_seed,
            }
        )
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    if not feasible:
        raise RuntimeError(
            "no mild configuration retains at least 97% Recall@10 on every validation seed"
        )
    selected = max(
        feasible,
        key=lambda candidate: (
            candidate["mean_observed_tail_recall_at_10"],
            candidate["mean_observed_tail_ndcg_at_10"],
            candidate["minimum_per_seed_recall_retention"],
            candidate["mean_recall_at_10"],
            candidate["mean_ndcg_at_10"],
            candidate["mean_observed_tail_catalog_coverage"],
        ),
    )
    audit = {
        "locked_at": utc_now(),
        "selection_data": "validation only",
        "validation_seeds": list(SEEDS),
        "test_used_for_selection": False,
        "test_reports_read_before_lock": False,
        "search_space": {
            "retrieval_weight": [0.15, 0.20, 0.25],
            "frequency_alpha": [0.05, 0.075, 0.10],
            "quota": {
                "off": True,
                "profiles": ["mild", "balanced", "tail_20"],
                "positive_strengths": [0.005, 0.01],
            },
        },
        "gate": "Recall@10 retention >= 0.97 on each validation seed",
        "objective": [
            "maximize mean Observed-Tail Recall@10",
            "then mean Observed-Tail NDCG@10",
            "then minimum per-seed Recall@10 retention",
            "then overall Recall@10, NDCG@10, and tail coverage",
        ],
        "shared_configurations": len(shared),
        "feasible_configurations": len(feasible),
        "selected": selected,
        "all_candidates": candidates,
    }
    return dict(selected["parameters"]), audit


def student_t_ci(values: list[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("Student-t CI requires at least two values")
    critical = 4.303 if len(values) == 3 else 1.96
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values)
    margin = critical * standard_deviation / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "ci95_margin": margin,
    }


def metric_values(metrics: dict[str, object]) -> dict[str, float]:
    return {
        "recall_at_5": float(metrics["ranking"]["recall_at_5"]),
        "ndcg_at_5": float(metrics["ranking"]["ndcg_at_5"]),
        "recall_at_10": float(metrics["ranking"]["recall_at_10"]),
        "ndcg_at_10": float(metrics["ranking"]["ndcg_at_10"]),
        "observed_tail_recall_at_5": float(
            metrics["target_groups"]["observed_tail"]["recall_at_5"]
        ),
        "observed_tail_ndcg_at_5": float(
            metrics["target_groups"]["observed_tail"]["ndcg_at_5"]
        ),
        "observed_tail_recall_at_10": float(
            metrics["target_groups"]["observed_tail"]["recall_at_10"]
        ),
        "observed_tail_ndcg_at_10": float(
            metrics["target_groups"]["observed_tail"]["ndcg_at_10"]
        ),
        "catalog_coverage": float(metrics["coverage"]["catalog_coverage"]),
        "observed_tail_catalog_coverage": float(
            metrics["strata_coverage"]["strata"]["observed_tail"]["catalog_coverage"]
        ),
        "observed_tail_recommendation_share": float(
            metrics["strata_coverage"]["strata"]["observed_tail"]["recommendation_share"]
        ),
    }


def aggregate_rows(
    split: str, rows: dict[int, dict[str, dict[str, float]]], per_seed: dict[str, object]
) -> dict[str, object]:
    metric_names = tuple(rows[SEEDS[0]]["baseline"])
    aggregate = {
        section: {
            metric: student_t_ci(
                [rows[seed][section][metric] for seed in SEEDS]
                if section != "delta"
                else [
                    rows[seed]["selected"][metric] - rows[seed]["baseline"][metric]
                    for seed in SEEDS
                ]
            )
            for metric in metric_names
        }
        for section in ("baseline", "selected", "delta")
    }
    return {
        "split": split,
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "student_t_95": aggregate,
    }


def summarize_validation(
    output_dir: Path,
    reports: dict[int, dict[str, object]],
    parameters: dict[str, object],
) -> dict[str, object]:
    key = parameter_key(parameters)
    rows: dict[int, dict[str, dict[str, float]]] = {}
    per_seed: dict[str, object] = {}
    for seed in SEEDS:
        report = reports[seed]
        selected = all_variants(report)[key]
        baseline_values = metric_values(report["baseline"]["metrics"])
        selected_values = metric_values(selected["metrics"])
        examples = int(report["examples"])
        baseline_values["latency_ms_per_example"] = (
            1000.0 * float(report["timing"]["generator_seconds"]) / examples
            + float(report["baseline"]["rerank_efficiency"]["milliseconds_per_example"])
        )
        selected_values["latency_ms_per_example"] = float(
            report["timing"]["fixed_pipeline_milliseconds_per_example"]
        ) + float(selected["rerank_efficiency"]["milliseconds_per_example"])
        rows[seed] = {"baseline": baseline_values, "selected": selected_values}
        per_seed[str(seed)] = {
            "baseline": baseline_values,
            "selected": selected_values,
            "delta": {
                metric: selected_values[metric] - baseline_values[metric]
                for metric in baseline_values
            },
            "report_path": str(
                (output_dir / "validation" / f"seed_{seed}" / "report.json").resolve()
            ),
        }
    return aggregate_rows("validation", rows, per_seed)


def summarize_test(output_dir: Path) -> dict[str, object]:
    rows: dict[int, dict[str, dict[str, float]]] = {}
    per_seed: dict[str, object] = {}
    for seed in SEEDS:
        report_path = output_dir / "test" / f"seed_{seed}" / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        baseline_values = metric_values(report["baseline"]["metrics"])
        selected_values = metric_values(report["selected"]["metrics"])
        examples = int(report["examples"])
        baseline_values["latency_ms_per_example"] = (
            1000.0 * float(report["timing"]["generator_seconds"]) / examples
            + float(report["baseline"]["rerank_efficiency"]["milliseconds_per_example"])
        )
        selected_values["latency_ms_per_example"] = float(
            report["timing"]["selected_total_milliseconds_per_example"]
        )
        rows[seed] = {"baseline": baseline_values, "selected": selected_values}
        per_seed[str(seed)] = {
            "baseline": baseline_values,
            "selected": selected_values,
            "delta": {
                metric: selected_values[metric] - baseline_values[metric]
                for metric in baseline_values
            },
            "candidate_oracle": report["candidate_oracle"],
            "evaluation_ablations": report["evaluation_ablations"],
            "report_path": str(report_path.resolve()),
        }
    return aggregate_rows("test", rows, per_seed)


def exact_mcnemar(baseline: np.ndarray, selected: np.ndarray) -> dict[str, float | int]:
    gains = int(np.sum((baseline == 0) & (selected == 1)))
    losses = int(np.sum((baseline == 1) & (selected == 0)))
    discordant = gains + losses
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(gains, losses)
        probability = sum(math.comb(discordant, index) for index in range(lower + 1)) / (
            2**discordant
        )
        p_value = min(1.0, 2.0 * probability)
    return {
        "selected_only_hits": gains,
        "baseline_only_hits": losses,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
    }


def bootstrap_ci(differences: np.ndarray, *, samples: int, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    for start in range(0, samples, 100):
        current = min(100, samples - start)
        indices = rng.integers(0, len(differences), size=(current, len(differences)))
        estimates.append(differences[indices].mean(axis=1))
    values = np.concatenate(estimates)
    return {
        "samples": samples,
        "mean_delta": float(differences.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def paired_test_summary(output_dir: Path, bootstrap_samples: int) -> dict[str, object]:
    output: dict[str, object] = {"per_seed": {}}
    for seed in SEEDS:
        indicator_path = output_dir / "test" / f"seed_{seed}" / "hit_indicators.jsonl"
        records = [
            json.loads(line)
            for line in indicator_path.read_text(encoding="utf-8").splitlines()
        ]
        seed_result: dict[str, object] = {}
        for group_name, keep in (
            ("overall", np.ones(len(records), dtype=bool)),
            (
                "observed_tail",
                np.asarray(
                    [record["target_group"] == "observed_tail" for record in records],
                    dtype=bool,
                ),
            ),
        ):
            group_result: dict[str, object] = {}
            for cutoff in (5, 10):
                baseline = np.asarray(
                    [
                        record["baseline_rank"] is not None
                        and int(record["baseline_rank"]) <= cutoff
                        for record in records
                    ],
                    dtype=np.int8,
                )[keep]
                selected = np.asarray(
                    [
                        record["selected_rank"] is not None
                        and int(record["selected_rank"]) <= cutoff
                        for record in records
                    ],
                    dtype=np.int8,
                )[keep]
                differences = selected.astype(np.float64) - baseline.astype(np.float64)
                group_result[f"hit_at_{cutoff}"] = {
                    "examples": len(differences),
                    "mcnemar": exact_mcnemar(baseline, selected),
                    "paired_user_bootstrap": bootstrap_ci(
                        differences,
                        samples=bootstrap_samples,
                        seed=seed + cutoff * 100 + (10000 if group_name != "overall" else 0),
                    ),
                }
            seed_result[group_name] = group_result
        output["per_seed"][str(seed)] = seed_result
    return output


def acceptance(test_summary: dict[str, object]) -> dict[str, object]:
    aggregate = test_summary["student_t_95"]
    baseline_r10 = float(aggregate["baseline"]["recall_at_10"]["mean"])
    selected_r10 = float(aggregate["selected"]["recall_at_10"]["mean"])
    retention = selected_r10 / baseline_r10
    tail_delta = aggregate["delta"]["observed_tail_recall_at_10"]
    per_seed_retention = {
        seed: float(values["selected"]["recall_at_10"])
        / float(values["baseline"]["recall_at_10"])
        for seed, values in test_summary["per_seed"].items()
    }
    recall_gate = retention >= MINIMUM_RECALL_RATIO
    tail_gate = float(tail_delta["ci95_low"]) > 0.0
    return {
        "overall_recall_at_10_retention": retention,
        "per_seed_recall_at_10_retention": per_seed_retention,
        "overall_recall_gate": {
            "threshold": MINIMUM_RECALL_RATIO,
            "passed": recall_gate,
        },
        "observed_tail_recall_delta_student_t_95": tail_delta,
        "tail_delta_ci_lower_above_zero": tail_gate,
        "passed": recall_gate and tail_gate,
    }


def create_pareto_summary(
    test_summary: dict[str, object], acceptance_result: dict[str, object]
) -> dict[str, object]:
    prior_path = ROOT / "artifacts" / "v2_long_tail_paired_reevaluation" / "paired_summary_student_t_95.json"
    aggressive_path = ROOT / "artifacts" / "candidate_reranking_v1" / "final_summary.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    aggressive = json.loads(aggressive_path.read_text(encoding="utf-8"))
    metric_names = (
        "recall_at_5",
        "ndcg_at_5",
        "recall_at_10",
        "ndcg_at_10",
        "observed_tail_recall_at_5",
        "observed_tail_recall_at_10",
        "catalog_coverage",
        "observed_tail_catalog_coverage",
        "observed_tail_recommendation_share",
        "milliseconds_per_example",
    )
    original = prior["models"]["original_v2_mmr000"]["test"]
    token_loss = prior["models"]["token_loss_010_mmr000"]["test"]
    mild = test_summary["student_t_95"]["selected"]
    aggressive_metrics = aggressive["test"]["student_t_95"]["selected"]
    mild_metrics = {
        ("milliseconds_per_example" if name == "latency_ms_per_example" else name): value
        for name, value in mild.items()
    }
    rows = {
        "original_v2": {name: original[name] for name in metric_names},
        "token_loss_010": {name: token_loss[name] for name in metric_names},
        "blair_frequency_mild": {name: mild_metrics[name] for name in metric_names},
        "aggressive_soft_quota": {
            name: aggressive_metrics[
                "latency_ms_per_example" if name == "milliseconds_per_example" else name
            ]
            for name in metric_names
        },
    }
    default_scheme = "blair_frequency_mild" if acceptance_result["passed"] else "token_loss_010"
    return {
        "comparison_scope": "same Amazon Reviews 2023 Musical Instruments test split, Beam-20, three seeds",
        "schemes": rows,
        "deployment_decision": {
            "default_online": default_scheme,
            "default_reason": (
                "mild reranker passed both the 97% overall Recall@10 retention gate and "
                "the positive Observed-Tail Recall@10 Student-t CI gate"
                if acceptance_result["passed"]
                else "mild reranker did not pass both pre-registered acceptance gates"
            ),
            "long_tail_enhancement_mode": "aggressive_soft_quota",
            "long_tail_mode_warning": (
                "opt-in only: strongest tail exposure/hit lift, but its prior formal run "
                "degraded overall Recall/NDCG"
            ),
        },
        "sources": {
            "original_and_token_loss": str(prior_path.resolve()),
            "aggressive_soft_quota": str(aggressive_path.resolve()),
        },
    }


def run(args: argparse.Namespace) -> None:
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    output_dir = args.output_dir.resolve()
    state_path = output_dir / "state.json"
    state: dict[str, object] = {
        "status": "running_validation_selection",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "completed_tasks": [],
        "current_task": None,
        "locked_parameters": None,
        "error": None,
    }
    write_json(state_path, state)

    validation_reports: dict[int, dict[str, object]] = {}
    for seed in SEEDS:
        task_name = f"validation_tune/seed_{seed}"
        report_path = output_dir / "validation" / f"seed_{seed}" / "report.json"
        state["current_task"] = task_name
        state["updated_at"] = utc_now()
        write_json(state_path, state)
        if not report_is_complete(report_path, seed=seed, split="validation", mode="tune"):
            run_command(
                validation_tune_command(report_path.parent, seed),
                output_dir / "logs" / f"validation_tune_seed_{seed}.log",
            )
        validation_reports[seed] = json.loads(report_path.read_text(encoding="utf-8"))
        state["completed_tasks"].append(task_name)
        state["updated_at"] = utc_now()
        write_json(state_path, state)

    parameters, selection_audit = locked_validation_selection(validation_reports)
    lock_path = output_dir / "locked_validation_selection.json"
    write_json(lock_path, selection_audit)
    state["locked_parameters"] = parameters
    state["selection_audit"] = str(lock_path.resolve())
    state["status"] = "running_locked_test_evaluation"
    state["current_task"] = None
    state["locked_at"] = selection_audit["locked_at"]
    state["updated_at"] = utc_now()
    write_json(state_path, state)

    for seed in SEEDS:
        task_name = f"test_evaluate/seed_{seed}"
        report_path = output_dir / "test" / f"seed_{seed}" / "report.json"
        state["current_task"] = task_name
        state["updated_at"] = utc_now()
        write_json(state_path, state)
        if not report_is_complete(
            report_path,
            seed=seed,
            split="test",
            mode="evaluate",
            parameters=parameters,
        ):
            run_command(
                test_command(report_path.parent, seed, parameters),
                output_dir / "logs" / f"test_evaluate_seed_{seed}.log",
            )
        state["completed_tasks"].append(task_name)
        state["updated_at"] = utc_now()
        write_json(state_path, state)

    validation_summary = summarize_validation(output_dir, validation_reports, parameters)
    test_summary = summarize_test(output_dir)
    paired_tests = paired_test_summary(output_dir, args.bootstrap_samples)
    acceptance_result = acceptance(test_summary)
    pareto = create_pareto_summary(test_summary, acceptance_result)
    write_json(output_dir / "validation_summary.json", validation_summary)
    write_json(output_dir / "test_summary.json", test_summary)
    write_json(output_dir / "paired_tests.json", paired_tests)
    write_json(output_dir / "acceptance.json", acceptance_result)
    write_json(output_dir / "v2_pareto_summary.json", pareto)
    write_json(
        output_dir / "final_summary.json",
        {
            "created_at": utc_now(),
            "dataset": "McAuley-Lab/Amazon-Reviews-2023",
            "synthetic_data_used": False,
            "selection_protocol": {
                "validation_only": True,
                "test_used_for_selection": False,
                "selection_audit": str(lock_path.resolve()),
            },
            "locked_parameters": parameters,
            "validation": validation_summary,
            "test": test_summary,
            "paired_tests": paired_tests,
            "acceptance": acceptance_result,
            "pareto": pareto,
        },
    )
    state["status"] = "completed"
    state["current_task"] = None
    state["completed_at"] = utc_now()
    state["updated_at"] = utc_now()
    write_json(state_path, state)


def main() -> None:
    args = parse_args()
    error: BaseException | None = None
    try:
        run(args)
    except BaseException as exc:
        error = exc
        state_path = args.output_dir.resolve() / "state.json"
        try:
            state = (
                json.loads(state_path.read_text(encoding="utf-8"))
                if state_path.exists()
                else {}
            )
            state.update(
                {
                    "status": "failed",
                    "updated_at": utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            write_json(state_path, state)
        except Exception:
            pass
    if args.linger:
        while True:
            time.sleep(3600)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
