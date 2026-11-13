from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2026, 2027, 2028)
MODES = ("fixed_0p5_gate", "no_event_attention", "no_fusion_mlp")
CONFIG = "configs/v2_preference_reasoning_retraining.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently retrain and evaluate three fine-grained Preference "
            "Reasoning architecture ablations"
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/preference_reasoning_retraining_ablation_v1"),
    )
    parser.add_argument(
        "--stage", choices=("all", "train", "evaluate", "summarize"), default="all"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-workers", type=int, default=3)
    parser.add_argument("--train-threads-per-worker", type=int, default=2)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--torch-threads-per-worker", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
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


def run(command: list[str], log_path: Path, *, torch_threads: int = 0) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if torch_threads > 0:
        environment["OMP_NUM_THREADS"] = str(torch_threads)
        environment["MKL_NUM_THREADS"] = str(torch_threads)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def validate_training_report(path: Path, mode: str, seed: int) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if int(report["seed"]) != seed:
        raise ValueError(f"seed mismatch in {path}")
    if report["generator_config"].get("preference_reasoning_mode") != mode:
        raise ValueError(f"reasoning mode mismatch in {path}")
    audit = report.get("data_audit", {})
    if audit.get("status") != "passed" or audit.get("synthetic_data_used") is not False:
        raise ValueError(f"real-data audit failed in {path}")
    if "test" in report:
        raise ValueError(f"training report unexpectedly evaluated Test: {path}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.eval_workers <= 0:
        raise ValueError("eval-workers must be positive")
    if args.train_workers <= 0:
        raise ValueError("train-workers must be positive")
    if args.train_threads_per_worker <= 0:
        raise ValueError("train-threads-per-worker must be positive")
    if args.torch_threads_per_worker < 0:
        raise ValueError("torch-threads-per-worker must be non-negative")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    state_path = output_dir / "state.json"
    completed: list[str] = []
    step = "initialize"
    try:
        if args.stage in {"all", "train"}:
            training_tasks: list[tuple[str, str, int, list[str], Path, Path]] = []
            for mode in MODES:
                for seed in SEEDS:
                    seed_dir = output_dir / mode / f"seed_{seed}"
                    report_path = seed_dir / "report.json"
                    task_step = f"train_{mode}_seed_{seed}"
                    if report_path.is_file():
                        validate_training_report(report_path, mode, seed)
                        completed.append(task_step)
                        continue
                    command = [
                            sys.executable,
                            "scripts/train_amazon2023.py",
                            "--config",
                            CONFIG,
                            "--seed",
                            str(seed),
                            "--output-dir",
                            str(seed_dir),
                            "--preference-reasoning-mode",
                            mode,
                        ]
                    training_tasks.append(
                        (
                            task_step,
                            mode,
                            seed,
                            command,
                            output_dir / "logs" / f"{task_step}.log",
                            report_path,
                        )
                    )

            write_state(
                state_path,
                status="running",
                phase="parallel_training",
                step="parallel_training",
                completed=completed,
                pending=[task[0] for task in training_tasks],
                train_workers=args.train_workers,
                train_threads_per_worker=args.train_threads_per_worker,
            )
            with ThreadPoolExecutor(max_workers=args.train_workers) as executor:
                training_futures = {
                    executor.submit(
                        run,
                        command,
                        log_path,
                        torch_threads=args.train_threads_per_worker,
                    ): (task_step, mode, seed, report_path)
                    for task_step, mode, seed, command, log_path, report_path in training_tasks
                }
                for future in as_completed(training_futures):
                    task_step, mode, seed, report_path = training_futures[future]
                    future.result()
                    validate_training_report(report_path, mode, seed)
                    completed.append(task_step)
                    write_state(
                        state_path,
                        status="running",
                        phase="parallel_training",
                        step="parallel_training",
                        last_completed=task_step,
                        completed=completed,
                        pending=[
                            value[0]
                            for value in training_futures.values()
                            if value[0] not in completed
                        ],
                        train_workers=args.train_workers,
                        train_threads_per_worker=args.train_threads_per_worker,
                    )

        if args.stage in {"all", "evaluate"}:
            tasks: list[tuple[str, list[str], Path]] = []
            for mode in MODES:
                for seed in SEEDS:
                    seed_dir = output_dir / mode / f"seed_{seed}"
                    training_report = seed_dir / "report.json"
                    if not training_report.is_file():
                        raise FileNotFoundError(f"missing training report: {training_report}")
                    validate_training_report(training_report, mode, seed)
                    report_path = seed_dir / "beam20_test.json"
                    task_step = f"evaluate_{mode}_seed_{seed}"
                    if report_path.is_file():
                        completed.append(task_step)
                        continue
                    command = [
                        sys.executable,
                        "scripts/analyze_beam_coverage_efficiency.py",
                        "--config",
                        CONFIG,
                        "--seed",
                        str(seed),
                        "--artifact-dir",
                        str(seed_dir),
                        "--split",
                        "test",
                        "--beam-size",
                        "20",
                        "--batch-size",
                        str(args.batch_size),
                        "--preference-reasoning-mode",
                        mode,
                        "--hit-indicators-dir",
                        str(seed_dir / "hit_indicators"),
                        "--output",
                        str(report_path),
                    ]
                    tasks.append(
                        (task_step, command, output_dir / "logs" / f"{task_step}.log")
                    )

            write_state(
                state_path,
                status="running",
                phase="parallel_test_evaluation",
                step="parallel_test_evaluation",
                completed=completed,
                pending=[task[0] for task in tasks],
                eval_workers=args.eval_workers,
                torch_threads_per_worker=args.torch_threads_per_worker,
            )
            with ThreadPoolExecutor(max_workers=args.eval_workers) as executor:
                futures = {
                    executor.submit(
                        run,
                        command,
                        log_path,
                        torch_threads=args.torch_threads_per_worker,
                    ): task_step
                    for task_step, command, log_path in tasks
                }
                for future in as_completed(futures):
                    task_step = futures[future]
                    future.result()
                    completed.append(task_step)
                    write_state(
                        state_path,
                        status="running",
                        phase="parallel_test_evaluation",
                        step="parallel_test_evaluation",
                        last_completed=task_step,
                        completed=completed,
                        pending=[
                            value for value in futures.values() if value not in completed
                        ],
                        eval_workers=args.eval_workers,
                        torch_threads_per_worker=args.torch_threads_per_worker,
                    )

        if args.stage in {"all", "summarize"}:
            step = "summarize"
            write_state(
                state_path,
                status="running",
                phase="summarize",
                step=step,
                completed=completed,
            )
            run(
                [
                    sys.executable,
                    "scripts/analyze_preference_reasoning_retraining_ablation.py",
                    "--output-dir",
                    str(output_dir),
                    "--bootstrap-samples",
                    str(args.bootstrap_samples),
                ],
                output_dir / "logs" / f"{step}.log",
            )
            completed.append(step)

        write_state(
            state_path,
            status="completed",
            phase=None,
            step=None,
            completed=completed,
        )
    except BaseException as error:
        write_state(
            state_path,
            status="failed",
            phase=locals().get("phase"),
            step=step,
            completed=completed,
            error=repr(error),
        )
        raise


if __name__ == "__main__":
    main()
