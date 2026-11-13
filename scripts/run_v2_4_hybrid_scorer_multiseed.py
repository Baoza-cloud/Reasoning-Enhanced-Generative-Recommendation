from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from regenrec.metrics import mean_confidence_interval


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2026, 2027, 2028)
CONFIG = "configs/v2_4_adaptive_forgetting.yaml"
DYNAMIC_ROOT = Path("artifacts/v2_4_adaptive_forgetting")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate the V2.4 hybrid candidate scorer on disjoint Validation folds"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v2_4_hybrid_scorer_multiseed"),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def write_state(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"updated_at": datetime.now(timezone.utc).isoformat(), **payload},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def wait_for_dynamic(state_path: Path, poll_seconds: int, output_state: Path) -> None:
    while True:
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") == "completed":
                return
            if state.get("status") == "failed":
                raise RuntimeError(f"dynamic-forgetting prerequisite failed: {state}")
        write_state(
            output_state,
            status="waiting_for_dynamic_forgetting",
            prerequisite=str(state_path.resolve()),
        )
        time.sleep(poll_seconds)


def run(command: list[str], log_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["OMP_NUM_THREADS"] = "2"
    environment["MKL_NUM_THREADS"] = "2"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def summarize(output_dir: Path) -> None:
    reports = {
        seed: json.loads((output_dir / f"seed_{seed}/report.json").read_text())
        for seed in SEEDS
    }
    metric_paths = {
        "recall_at_5": ("ranking", "recall_at_5"),
        "ndcg_at_5": ("ranking", "ndcg_at_5"),
        "recall_at_10": ("ranking", "recall_at_10"),
        "ndcg_at_10": ("ranking", "ndcg_at_10"),
        "observed_tail_recall_at_10": (
            "target_groups",
            "observed_tail",
            "recall_at_10",
        ),
    }
    metrics: dict[str, object] = {}
    for name, path in metric_paths.items():
        baseline: list[float] = []
        learned: list[float] = []
        for seed in SEEDS:
            baseline_value: object = reports[seed]["baseline"]["metrics"]
            learned_value: object = reports[seed]["selected"]["metrics"]
            for key in path:
                baseline_value = baseline_value[key]
                learned_value = learned_value[key]
            baseline.append(float(baseline_value))
            learned.append(float(learned_value))
        delta = [right - left for left, right in zip(baseline, learned, strict=True)]
        metrics[name] = {
            "generator_baseline": mean_confidence_interval(baseline),
            "learned_hybrid": mean_confidence_interval(learned),
            "paired_delta": mean_confidence_interval(delta),
        }
    overall_ratio = (
        metrics["recall_at_10"]["learned_hybrid"]["mean"]
        / metrics["recall_at_10"]["generator_baseline"]["mean"]
    )
    tail_delta = metrics["observed_tail_recall_at_10"]["paired_delta"]
    acceptance = {
        "overall_recall_at_10_retention": overall_ratio,
        "overall_retention_gate": overall_ratio >= 0.97,
        "tail_delta_ci_low": tail_delta["ci95_low"],
        "stable_tail_gain_gate": tail_delta["ci95_low"] > 0,
    }
    acceptance["accepted_as_default"] = (
        acceptance["overall_retention_gate"] and acceptance["stable_tail_gain_gate"]
    )
    summary = {
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "category": "Musical_Instruments",
        "synthetic_data_used": False,
        "split": "validation_user_hash_selection_fold",
        "test_evaluated": False,
        "seeds": list(SEEDS),
        "metrics": metrics,
        "acceptance": acceptance,
        "decision_rule": (
            "default only if mean overall Recall@10 retains >=97% and the paired "
            "Student-t 95% CI lower bound for Observed-Tail Recall@10 is >0"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.poll_seconds <= 0:
        raise ValueError("workers and poll-seconds must be positive")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    state_path = output_dir / "state.json"
    completed: list[str] = []
    try:
        wait_for_dynamic(ROOT / DYNAMIC_ROOT / "state.json", args.poll_seconds, state_path)
        tasks: list[tuple[int, list[str], Path]] = []
        for seed in SEEDS:
            seed_dir = output_dir / f"seed_{seed}"
            report_path = seed_dir / "report.json"
            step = f"fit_evaluate_seed_{seed}"
            if report_path.is_file():
                completed.append(step)
                continue
            tasks.append(
                (
                    seed,
                    [
                        sys.executable,
                        "scripts/analyze_candidate_reranking.py",
                        "--config",
                        CONFIG,
                        "--artifact-dir",
                        str(ROOT / DYNAMIC_ROOT / f"seed_{seed}"),
                        "--seed",
                        str(seed),
                        "--split",
                        "validation",
                        "--mode",
                        "tune",
                        "--beam-size",
                        "20",
                        "--candidate-count",
                        "20",
                        "--retrieval-count",
                        "50",
                        "--top-k",
                        "10",
                        "--hybrid-scorer-mode",
                        "fit",
                        "--hybrid-fit-fraction",
                        "0.7",
                        "--hybrid-epochs",
                        "300",
                        "--hybrid-tail-positive-weight",
                        "1.5",
                        "--output-dir",
                        str(seed_dir),
                    ],
                    output_dir / "logs" / f"{step}.log",
                )
            )
        write_state(
            state_path,
            status="running",
            step="parallel_validation_fit_evaluation",
            completed=completed,
            pending=[f"fit_evaluate_seed_{seed}" for seed, *_ in tasks],
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run, command, log_path): seed
                for seed, command, log_path in tasks
            }
            for future in as_completed(futures):
                seed = futures[future]
                future.result()
                completed.append(f"fit_evaluate_seed_{seed}")
                write_state(
                    state_path,
                    status="running",
                    step="parallel_validation_fit_evaluation",
                    completed=completed,
                )
        summarize(output_dir)
        write_state(
            state_path,
            status="completed",
            step=None,
            completed=completed + ["summarize"],
        )
    except Exception as error:
        write_state(
            state_path,
            status="failed",
            completed=completed,
            error=repr(error),
        )
        raise


if __name__ == "__main__":
    main()
