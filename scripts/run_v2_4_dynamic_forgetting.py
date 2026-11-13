from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from regenrec.metrics import mean_confidence_interval


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2026, 2027, 2028)
CONFIG = "configs/v2_4_adaptive_forgetting.yaml"
BASELINE_ROOT = Path(
    "artifacts/amazon2023_musical_instruments_long_tail_training_v1/formal/token_loss_010"
)
METRIC_PATHS = {
    "recall_at_5": ("validation", "recall_at_5"),
    "ndcg_at_5": ("validation", "ndcg_at_5"),
    "recall_at_10": ("validation", "recall_at_10"),
    "ndcg_at_10": ("validation", "ndcg_at_10"),
    "legacy_tail_recall_at_10": ("validation", "target_groups", "tail", "recall_at_10"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train V2.4 learned history forgetting on Validation only"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v2_4_adaptive_forgetting"),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=2)
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


def run(command: list[str], log_path: Path, threads: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = str(threads)
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def nested_float(payload: dict[str, object], path: tuple[str, ...]) -> float:
    value: object = payload
    for key in path:
        if not isinstance(value, dict):
            raise ValueError(f"invalid report path: {'.'.join(path)}")
        value = value[key]
    return float(value)


def validate_report(path: Path, seed: int) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if int(report["seed"]) != seed:
        raise ValueError(f"seed mismatch: {path}")
    generator = report["generator_config"]
    if generator.get("history_forgetting_mode") != "learned_exponential":
        raise ValueError(f"wrong forgetting mode: {path}")
    if "test" in report:
        raise ValueError(f"V2.4 development run evaluated reused Test: {path}")
    audit = report["data_audit"]
    if audit.get("status") != "passed" or audit.get("synthetic_data_used") is not False:
        raise ValueError(f"real-data audit failed: {path}")
    return report


def summarize(output_dir: Path) -> None:
    dynamic_reports = {
        seed: validate_report(output_dir / f"seed_{seed}/report.json", seed)
        for seed in SEEDS
    }
    baseline_reports = {
        seed: json.loads((ROOT / BASELINE_ROOT / f"seed_{seed}/report.json").read_text())
        for seed in SEEDS
    }
    metrics: dict[str, object] = {}
    for name, path in METRIC_PATHS.items():
        dynamic = [nested_float(dynamic_reports[seed], path) for seed in SEEDS]
        baseline = [nested_float(baseline_reports[seed], path) for seed in SEEDS]
        delta = [left - right for left, right in zip(dynamic, baseline, strict=True)]
        metrics[name] = {
            "adaptive_forgetting": mean_confidence_interval(dynamic),
            "frozen_v2_baseline": mean_confidence_interval(baseline),
            "paired_delta": mean_confidence_interval(delta),
            "stable_positive": mean_confidence_interval(delta)["ci95_low"] > 0,
        }
    summary = {
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "category": "Musical_Instruments",
        "synthetic_data_used": False,
        "selection_split": "validation",
        "test_evaluated": False,
        "seeds": list(SEEDS),
        "interval": "two-sided 95% Student-t confidence interval",
        "metrics": metrics,
        "next_gate": (
            "Freeze only from Validation; confirm once on untouched Sports, Toys, and "
            "Beauty Test splits before any generalization claim."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.threads_per_worker <= 0:
        raise ValueError("worker counts must be positive")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    state_path = output_dir / "state.json"
    tasks: list[tuple[int, list[str], Path, Path]] = []
    completed: list[str] = []
    for seed in SEEDS:
        seed_dir = output_dir / f"seed_{seed}"
        report_path = seed_dir / "report.json"
        step = f"train_seed_{seed}"
        if report_path.is_file():
            validate_report(report_path, seed)
            completed.append(step)
            continue
        tasks.append(
            (
                seed,
                [
                    sys.executable,
                    "scripts/train_amazon2023.py",
                    "--config",
                    CONFIG,
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(seed_dir),
                ],
                output_dir / "logs" / f"{step}.log",
                report_path,
            )
        )
    try:
        write_state(
            state_path,
            status="running",
            phase="validation_only_training",
            completed=completed,
            pending=[f"train_seed_{seed}" for seed, *_ in tasks],
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run, command, log_path, args.threads_per_worker): (
                    seed,
                    report_path,
                )
                for seed, command, log_path, report_path in tasks
            }
            for future in as_completed(futures):
                seed, report_path = futures[future]
                future.result()
                validate_report(report_path, seed)
                completed.append(f"train_seed_{seed}")
                write_state(
                    state_path,
                    status="running",
                    phase="validation_only_training",
                    completed=completed,
                )
        summarize(output_dir)
        write_state(
            state_path,
            status="completed",
            phase=None,
            completed=completed + ["summarize"],
        )
    except Exception as error:
        write_state(
            state_path,
            status="failed",
            phase="validation_only_training",
            completed=completed,
            error=repr(error),
        )
        raise


if __name__ == "__main__":
    main()
