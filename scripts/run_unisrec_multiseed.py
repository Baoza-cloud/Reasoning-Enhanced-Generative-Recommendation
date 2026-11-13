from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from regenrec.metrics import mean_confidence_interval


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "external/UniSRec/.venv/bin/python"
DATASET_DIR = ROOT / "external/UniSRec/dataset/downstream/Amazon2023Musical"
SEEDS = (2026, 2027, 2028)
PREREQUISITES = (
    ROOT / "artifacts/v2_4_adaptive_forgetting/state.json",
    ROOT / "artifacts/v2_4_external_preparation/state.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official UniSRec on audited Amazon 2023 data across three seeds"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/unisrec_amazon2023_musical_multiseed"),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=512)
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


def wait_for_prerequisites(output_state: Path, poll_seconds: int) -> None:
    while True:
        waiting: list[str] = []
        for path in PREREQUISITES:
            if not path.is_file():
                waiting.append(str(path))
                continue
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("status") == "failed":
                raise RuntimeError(f"prerequisite failed: {path}: {state}")
            if state.get("status") != "completed":
                waiting.append(str(path))
        if not waiting:
            return
        write_state(
            output_state,
            status="waiting_for_prerequisites",
            waiting=waiting,
        )
        time.sleep(poll_seconds)


def run(command: list[str], log_path: Path, threads: int) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = str(threads)
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
    reports = [
        json.loads((output_dir / f"seed_{seed}/report.json").read_text())
        for seed in SEEDS
    ]
    summary: dict[str, object] = {
        "model": "official RUCAIBox/UniSRec from scratch",
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "category": "Musical_Instruments",
        "synthetic_data_used": False,
        "seeds": list(SEEDS),
        "interval": "two-sided 95% Student-t confidence interval",
        "selection_split": "validation",
        "test_used_for_selection": False,
    }
    for split, key in (
        ("validation", "best_valid_result"),
        ("test", "test_result"),
    ):
        summary[split] = {
            metric: mean_confidence_interval(
                [float(report["results"][key][metric]) for report in reports]
            )
            for metric in ("recall@5", "ndcg@5", "recall@10", "ndcg@10")
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if min(
        args.workers,
        args.threads_per_worker,
        args.poll_seconds,
        args.epochs,
        args.train_batch_size,
        args.eval_batch_size,
    ) <= 0:
        raise ValueError("worker, poll, and epoch settings must be positive")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    state_path = output_dir / "state.json"
    completed: list[str] = []
    try:
        wait_for_prerequisites(state_path, args.poll_seconds)
        tasks: list[tuple[int, list[str], Path]] = []
        for seed in SEEDS:
            seed_dir = output_dir / f"seed_{seed}"
            report_path = seed_dir / "report.json"
            step = f"train_evaluate_seed_{seed}"
            if report_path.is_file():
                completed.append(step)
                continue
            tasks.append(
                (
                    seed,
                    [
                        str(PYTHON),
                        "scripts/train_unisrec_baseline.py",
                        "--dataset-dir",
                        str(DATASET_DIR),
                        "--output-dir",
                        str(seed_dir),
                        "--seed",
                        str(seed),
                        "--epochs",
                        str(args.epochs),
                        "--stopping-step",
                        "3",
                        "--train-batch-size",
                        str(args.train_batch_size),
                        "--eval-batch-size",
                        str(args.eval_batch_size),
                    ],
                    output_dir / "logs" / f"{step}.log",
                )
            )
        write_state(
            state_path,
            status="running",
            step=(
                "sequential_unisrec_training"
                if args.workers == 1
                else "parallel_unisrec_training"
            ),
            completed=completed,
            pending=[f"train_evaluate_seed_{seed}" for seed, *_ in tasks],
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run, command, log_path, args.threads_per_worker): seed
                for seed, command, log_path in tasks
            }
            for future in as_completed(futures):
                seed = futures[future]
                future.result()
                completed.append(f"train_evaluate_seed_{seed}")
                write_state(
                    state_path,
                    status="running",
                    step=(
                        "sequential_unisrec_training"
                        if args.workers == 1
                        else "parallel_unisrec_training"
                    ),
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
