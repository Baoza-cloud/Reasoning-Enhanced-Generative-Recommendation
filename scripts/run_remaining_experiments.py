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
    parser = argparse.ArgumentParser(description="Run the remaining formal experiments serially")
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def experiment_steps() -> list[dict[str, object]]:
    return [
        {
            "name": "v2_main",
            "report": Path(
                "artifacts/amazon2023_musical_instruments_blair_v2_intensity/report.json"
            ),
            "wait_only": True,
        },
        {
            "name": "v2_no_intensity_ablation",
            "report": Path(
                "artifacts/amazon2023_musical_instruments_blair_v2_no_intensity/report.json"
            ),
            "command": [
                sys.executable,
                "scripts/train_amazon2023.py",
                "--config",
                "configs/v2_amazon2023_ablation_no_intensity.yaml",
            ],
        },
        {
            "name": "sasrec_full",
            "report": Path("artifacts/amazon2023_musical_instruments_sasrec/report.json"),
            "command": [
                sys.executable,
                "scripts/train_sasrec_baseline.py",
                "--config",
                "configs/baseline_sasrec_amazon2023.yaml",
            ],
        },
        {
            "name": "criteo_uplift_full",
            "report": Path("artifacts/criteo_uplift_dr_full/report.json"),
            "command": [
                sys.executable,
                "scripts/train_criteo_uplift.py",
                "--config",
                "configs/v3_criteo_uplift.yaml",
            ],
        },
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, **payload: object) -> None:
    path.write_text(
        json.dumps({"updated_at": utc_now(), **payload}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    state_path = root / "artifacts" / "remaining_experiments_state.json"
    log_dir = root / "artifacts" / "experiment_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = ".:src"

    completed: list[str] = []
    for step in experiment_steps():
        name = str(step["name"])
        report_path = root / Path(step["report"])
        if bool(step.get("wait_only")):
            while not report_path.is_file():
                write_state(
                    state_path,
                    status="waiting",
                    current=name,
                    completed=completed,
                    expected_report=str(report_path),
                )
                time.sleep(args.poll_seconds)
            completed.append(name)
            continue
        if report_path.is_file():
            completed.append(name)
            continue

        command = [str(part) for part in step["command"]]
        log_path = log_dir / f"{name}.log"
        write_state(
            state_path,
            status="running",
            current=name,
            completed=completed,
            command=command,
            log=str(log_path),
        )
        with log_path.open("a", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0 or not report_path.is_file():
            write_state(
                state_path,
                status="failed",
                current=name,
                completed=completed,
                returncode=result.returncode,
                expected_report=str(report_path),
                log=str(log_path),
            )
            raise SystemExit(result.returncode or 1)
        completed.append(name)

    write_state(state_path, status="completed", current=None, completed=completed)


if __name__ == "__main__":
    main()
