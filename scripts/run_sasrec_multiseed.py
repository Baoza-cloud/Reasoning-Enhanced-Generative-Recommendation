from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SASRec across independent random seeds")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--poll-seconds", type=int, default=60)
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
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    top10_summary = (
        root
        / "artifacts"
        / "amazon2023_musical_instruments_blair_v2_top10_multiseed"
        / "summary.json"
    )
    artifact_root = root / "artifacts" / "amazon2023_musical_instruments_sasrec_multiseed"
    log_dir = artifact_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = artifact_root / "state.json"
    while not top10_summary.is_file():
        write_state(
            state_path,
            status="waiting_for_top10_multiseed",
            expected_summary=str(top10_summary),
            completed_seeds=[],
        )
        time.sleep(args.poll_seconds)

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = ".:src"
    completed: list[int] = []
    report_paths: list[Path] = []
    for seed in seeds:
        if seed == 2026:
            report_path = root / "artifacts" / "amazon2023_musical_instruments_sasrec" / "report.json"
        else:
            output_dir = artifact_root / f"seed_{seed}"
            report_path = output_dir / "report.json"
        report_paths.append(report_path)
        if report_path.is_file():
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            if int(payload.get("seed", -1)) == seed:
                completed.append(seed)
                continue
        if seed == 2026:
            raise RuntimeError("the existing SASRec seed-2026 report is missing its seed metadata")
        command = [
            sys.executable,
            "scripts/train_sasrec_baseline.py",
            "--config",
            "configs/baseline_sasrec_amazon2023.yaml",
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir),
        ]
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
        "SASRec full-catalog",
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
