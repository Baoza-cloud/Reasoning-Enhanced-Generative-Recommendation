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
MODES = (
    "fixed_0p5_gate",
    "long_term_only",
    "short_term_only",
    "no_event_attention",
    "no_fusion_mlp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable three-seed frozen-checkpoint reasoning interventions"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/preference_reasoning_mechanism_ablation_v1"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads-per-worker", type=int, default=0)
    parser.add_argument("--stage", choices=("all", "evaluate", "summarize"), default="all")
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


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.torch_threads_per_worker < 0:
        raise ValueError("torch-threads-per-worker must be non-negative")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    state_path = output_dir / "state.json"
    completed: list[str] = []
    step = "initialize"
    try:
        if args.stage in {"all", "evaluate"}:
            tasks: list[tuple[str, list[str], Path]] = []
            for mode in MODES:
                for seed in SEEDS:
                    seed_dir = output_dir / mode / f"seed_{seed}"
                    report_path = seed_dir / "beam20_test.json"
                    step = f"evaluate_{mode}_seed_{seed}"
                    if report_path.is_file():
                        completed.append(step)
                        continue
                    artifact = ROOT / (
                        "artifacts/amazon2023_musical_instruments_long_tail_training_v1/"
                        f"formal/token_loss_010/seed_{seed}"
                    )
                    command = [
                        sys.executable,
                        "scripts/analyze_beam_coverage_efficiency.py",
                        "--config",
                        "configs/v2_final_default.yaml",
                        "--seed",
                        str(seed),
                        "--artifact-dir",
                        str(artifact),
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
                    tasks.append((step, command, output_dir / "logs" / f"{step}.log"))

            write_state(
                state_path,
                status="running",
                step="parallel_evaluation",
                completed=completed,
                pending=[task[0] for task in tasks],
                workers=args.workers,
                torch_threads_per_worker=args.torch_threads_per_worker,
            )
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
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
                    pending = [value for value in futures.values() if value not in completed]
                    write_state(
                        state_path,
                        status="running",
                        step="parallel_evaluation",
                        last_completed=task_step,
                        completed=completed,
                        pending=pending,
                        workers=args.workers,
                        torch_threads_per_worker=args.torch_threads_per_worker,
                    )

        if args.stage in {"all", "summarize"}:
            step = "summarize"
            write_state(state_path, status="running", step=step, completed=completed)
            run(
                [
                    sys.executable,
                    "scripts/analyze_preference_reasoning_mechanisms.py",
                    "--output-dir",
                    str(output_dir),
                ],
                output_dir / "logs" / f"{step}.log",
            )
            completed.append(step)
        write_state(state_path, status="completed", step=None, completed=completed)
    except BaseException as error:
        write_state(
            state_path,
            status="failed",
            step=step,
            completed=completed,
            error=repr(error),
        )
        raise


if __name__ == "__main__":
    main()
