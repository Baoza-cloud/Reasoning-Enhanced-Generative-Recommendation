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
DEFAULT_OUTPUT = ROOT / "artifacts" / "candidate_reranking_v1"
SEEDS = (2026, 2027, 2028)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pilot tuning and locked multi-seed candidate-reranking evaluation"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pilot-examples", type=int, default=10000)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--linger", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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


def parameters_equal(left: dict[str, object], right: dict[str, object]) -> bool:
    return (
        math.isclose(float(left["retrieval_weight"]), float(right["retrieval_weight"]))
        and math.isclose(float(left["frequency_alpha"]), float(right["frequency_alpha"]))
        and str(left["quota_profile"]) == str(right["quota_profile"])
        and math.isclose(float(left["quota_strength"]), float(right["quota_strength"]))
    )


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
    if not (
        report.get("dataset") == "McAuley-Lab/Amazon-Reviews-2023"
        and report.get("synthetic_data_used") is False
        and int(report.get("seed", -1)) == seed
        and report.get("split") == split
        and report.get("mode") == mode
    ):
        return False
    if parameters is not None:
        return parameters_equal(report["selection"]["selected_parameters"], parameters)
    return True


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


def pilot_command(output_dir: Path, examples: int) -> list[str]:
    return [
        sys.executable,
        "scripts/analyze_candidate_reranking.py",
        "--config",
        "configs/v2_amazon2023_top10.yaml",
        "--artifact-dir",
        str(artifact_dir(2026)),
        "--seed",
        "2026",
        "--split",
        "validation",
        "--mode",
        "tune",
        "--max-examples",
        str(examples),
        "--selection-strategy",
        "reservoir",
        "--selection-seed",
        "90210",
        "--sweep-profile",
        "expanded",
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
        "0.97",
        "--log-every-batches",
        "25",
        "--output-dir",
        str(output_dir),
    ]


def formal_command(
    output_dir: Path,
    seed: int,
    split: str,
    parameters: dict[str, object],
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
        split,
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
        "100",
        "--output-dir",
        str(output_dir),
    ]


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


def metric_values(report: dict[str, object], section: str) -> dict[str, float]:
    metrics = report[section]["metrics"]
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
            metrics["strata_coverage"]["strata"]["observed_tail"][
                "recommendation_share"
            ]
        ),
    }


def summarize_split(output_dir: Path, split: str) -> dict[str, object]:
    per_seed: dict[str, object] = {}
    rows: dict[int, dict[str, dict[str, float]]] = {}
    for seed in SEEDS:
        report_path = output_dir / "formal" / split / f"seed_{seed}" / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        baseline = metric_values(report, "baseline")
        selected = metric_values(report, "selected")
        baseline["latency_ms_per_example"] = float(
            1000.0 * float(report["timing"]["generator_seconds"]) / int(report["examples"])
        ) + float(report["baseline"]["rerank_efficiency"]["milliseconds_per_example"])
        selected["latency_ms_per_example"] = float(
            report["timing"]["selected_total_milliseconds_per_example"]
        )
        rows[seed] = {"baseline": baseline, "selected": selected}
        per_seed[str(seed)] = {
            "baseline": baseline,
            "selected": selected,
            "delta": {
                metric: selected[metric] - baseline[metric] for metric in baseline
            },
            "candidate_oracle": report["candidate_oracle"],
            "evaluation_ablations": report["evaluation_ablations"],
            "report_path": str(report_path.resolve()),
        }
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


def bootstrap_ci(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
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
        indicators = (
            output_dir / "formal" / "test" / f"seed_{seed}" / "hit_indicators.jsonl"
        )
        rows = [json.loads(line) for line in indicators.read_text(encoding="utf-8").splitlines()]
        seed_result: dict[str, object] = {}
        for group_name, keep in (
            ("overall", np.ones(len(rows), dtype=bool)),
            (
                "observed_tail",
                np.asarray(
                    [row["target_group"] == "observed_tail" for row in rows], dtype=bool
                ),
            ),
        ):
            group_result: dict[str, object] = {}
            for cutoff in (5, 10):
                baseline = np.asarray(
                    [
                        row["baseline_rank"] is not None
                        and int(row["baseline_rank"]) <= cutoff
                        for row in rows
                    ],
                    dtype=np.int8,
                )[keep]
                selected = np.asarray(
                    [
                        row["selected_rank"] is not None
                        and int(row["selected_rank"]) <= cutoff
                        for row in rows
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


def run(args: argparse.Namespace) -> None:
    if args.pilot_examples <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("pilot-examples and bootstrap-samples must be positive")
    output_dir = args.output_dir.resolve()
    state_path = output_dir / "state.json"
    state: dict[str, object] = {
        "status": "running_pilot",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "completed_tasks": [],
        "current_task": "pilot/seed_2026/validation_10k",
        "selected_parameters": None,
        "error": None,
    }
    write_json(state_path, state)

    pilot_dir = output_dir / "pilot" / "seed_2026_validation_10k"
    pilot_report_path = pilot_dir / "report.json"
    if not report_is_complete(
        pilot_report_path, seed=2026, split="validation", mode="tune"
    ):
        run_command(
            pilot_command(pilot_dir, args.pilot_examples),
            output_dir / "logs" / "pilot_seed_2026_validation_10k.log",
        )
    pilot_report = json.loads(pilot_report_path.read_text(encoding="utf-8"))
    parameters = dict(pilot_report["selection"]["selected_parameters"])
    state["completed_tasks"].append("pilot/seed_2026/validation_10k")
    state["selected_parameters"] = parameters
    state["status"] = "running_formal"
    state["updated_at"] = utc_now()
    write_json(state_path, state)

    tasks = [(split, seed) for split in ("validation", "test") for seed in SEEDS]
    state["total_formal_tasks"] = len(tasks)
    write_json(state_path, state)
    for split, seed in tasks:
        task_name = f"formal/{split}/seed_{seed}"
        report_path = output_dir / "formal" / split / f"seed_{seed}" / "report.json"
        state["current_task"] = task_name
        state["updated_at"] = utc_now()
        write_json(state_path, state)
        if not report_is_complete(
            report_path,
            seed=seed,
            split=split,
            mode="evaluate",
            parameters=parameters,
        ):
            run_command(
                formal_command(report_path.parent, seed, split, parameters),
                output_dir / "logs" / f"formal_{split}_seed_{seed}.log",
            )
        state["completed_tasks"].append(task_name)
        state["updated_at"] = utc_now()
        write_json(state_path, state)

    validation_summary = summarize_split(output_dir, "validation")
    test_summary = summarize_split(output_dir, "test")
    paired_tests = paired_test_summary(output_dir, args.bootstrap_samples)
    write_json(output_dir / "validation_summary.json", validation_summary)
    write_json(output_dir / "test_summary.json", test_summary)
    write_json(output_dir / "paired_tests.json", paired_tests)
    write_json(
        output_dir / "final_summary.json",
        {
            "created_at": utc_now(),
            "dataset": "McAuley-Lab/Amazon-Reviews-2023",
            "synthetic_data_used": False,
            "pilot_report": str(pilot_report_path.resolve()),
            "locked_parameters": parameters,
            "validation": validation_summary,
            "test": test_summary,
            "paired_tests": paired_tests,
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
