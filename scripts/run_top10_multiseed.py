from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Beam-20 Amazon generative recommendation across independent seeds"
    )
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/v2_amazon2023_top10.yaml"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(
            "artifacts/amazon2023_musical_instruments_blair_v2_top10_multiseed"
        ),
    )
    parser.add_argument("--model-name", default="V2.1 global-CE + Beam-20/return-10")
    parser.add_argument("--resume-seed", type=int, default=2026)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=Path(
            "artifacts/amazon2023_musical_instruments_blair_v2_intensity/"
            "generator_checkpoint.pt"
        ),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, **payload: object) -> None:
    path.write_text(
        json.dumps({"updated_at": utc_now(), **payload}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_command(command: list[str], log_path: Path, *, root: Path, environment: dict[str, str]) -> None:
    with log_path.open("a", encoding="utf-8") as log_handle:
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {command}")


def main() -> None:
    args = parse_args()
    seeds = args.seeds or [2026, 2027, 2028]
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("provide at least two unique random seeds")

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    config_path = args.config if args.config.is_absolute() else root / args.config
    artifact_root = (
        args.artifact_root if args.artifact_root.is_absolute() else root / args.artifact_root
    )
    resume_from = (
        args.resume_from if args.resume_from.is_absolute() else root / args.resume_from
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"config does not exist: {config_path}")
    log_dir = artifact_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = artifact_root / "state.json"
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = ".:src"

    completed: list[int] = []
    report_paths: list[Path] = []
    for seed in seeds:
        output_dir = artifact_root / f"seed_{seed}"
        report_path = output_dir / "report.json"
        report_paths.append(report_path)
        if report_path.is_file():
            completed.append(seed)
            continue
        command = [
            sys.executable,
            "scripts/train_amazon2023.py",
            "--config",
            str(config_path),
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir),
        ]
        if seed == args.resume_seed:
            if not resume_from.is_file():
                raise FileNotFoundError(f"resume checkpoint does not exist: {resume_from}")
            command.extend(["--resume-from", str(resume_from)])
        write_state(
            state_path,
            status="running",
            current_seed=seed,
            completed_seeds=completed,
            command=command,
        )
        try:
            run_command(
                command,
                log_dir / f"seed_{seed}.log",
                root=root,
                environment=environment,
            )
        except RuntimeError as exc:
            write_state(
                state_path,
                status="failed",
                current_seed=seed,
                completed_seeds=completed,
                error=str(exc),
            )
            raise
        if not report_path.is_file():
            raise RuntimeError(f"training completed without report: {report_path}")
        completed.append(seed)

    summary_path = artifact_root / "summary.json"
    summary_command = [
        sys.executable,
        "scripts/summarize_multiseed.py",
        "--output",
        str(summary_path),
        "--model-name",
        args.model_name,
    ]
    for report_path in report_paths:
        summary_command.extend(["--report", str(report_path)])
    run_command(
        summary_command,
        log_dir / "summary.log",
        root=root,
        environment=environment,
    )
    write_state(
        state_path,
        status="completed",
        current_seed=None,
        completed_seeds=completed,
        summary=str(summary_path),
    )


if __name__ == "__main__":
    main()
