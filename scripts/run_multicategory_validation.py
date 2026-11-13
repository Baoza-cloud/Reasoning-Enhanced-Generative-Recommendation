from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


CATEGORY = "Industrial_and_Scientific"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Industrial_and_Scientific external-validation pipeline"
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def prerequisite_paths() -> list[Path]:
    return [
        Path(
            "data/amazon2023/processed/Industrial_and_Scientific/last_out_v2/manifest.json"
        ),
        Path("artifacts/amazon2023_musical_instruments_long_tail_full/report.json"),
    ]


def validation_steps() -> list[dict[str, object]]:
    processed = Path(
        "data/amazon2023/processed/Industrial_and_Scientific/last_out_v2"
    )
    generator_artifacts = Path("artifacts/amazon2023_industrial_scientific_blair_v2_seed2026")
    return [
        {
            "name": "audit_real_data",
            "report": processed / "data_audit.json",
            "command": [
                sys.executable,
                "scripts/audit_amazon2023.py",
                "--processed-dir",
                str(processed),
            ],
        },
        {
            "name": "blair_embeddings",
            "report": processed / "item_embeddings_blair.pt",
            "command": [
                sys.executable,
                "scripts/embed_amazon_items.py",
                "--input",
                str(processed / "items.jsonl"),
                "--output",
                str(processed / "item_embeddings_blair.pt"),
                "--backend",
                "blair",
                "--device",
                "cpu",
                "--batch-size",
                "64",
                "--max-length",
                "512",
            ],
        },
        {
            "name": "popularity_markov_baselines",
            "report": generator_artifacts / "baseline_report.json",
            "command": [
                sys.executable,
                "scripts/evaluate_baselines.py",
                "--config",
                "configs/v2_amazon2023_industrial_scientific.yaml",
                "--output",
                str(generator_artifacts / "baseline_report.json"),
            ],
        },
        {
            "name": "rqvae_v2_beam20",
            "report": generator_artifacts / "report.json",
            "command": [
                sys.executable,
                "scripts/train_amazon2023.py",
                "--config",
                "configs/v2_amazon2023_industrial_scientific.yaml",
            ],
        },
        {
            "name": "sasrec",
            "report": Path(
                "artifacts/amazon2023_industrial_scientific_sasrec_seed2026/report.json"
            ),
            "command": [
                sys.executable,
                "scripts/train_sasrec_baseline.py",
                "--config",
                "configs/baseline_sasrec_amazon2023_industrial_scientific.yaml",
            ],
        },
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"updated_at": utc_now(), **payload}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    state_path = root / "artifacts" / "multicategory_validation_state.json"
    log_dir = root / "artifacts" / "multicategory_validation_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": ".:src:scripts",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )

    prerequisites = prerequisite_paths()
    while missing := [path for path in prerequisites if not path.is_file()]:
        write_state(
            state_path,
            status="waiting_for_prerequisites",
            category=CATEGORY,
            missing=[str(path) for path in missing],
        )
        time.sleep(args.poll_seconds)

    completed: list[str] = []
    for step in validation_steps():
        name = str(step["name"])
        report_path = root / Path(step["report"])
        if report_path.is_file():
            completed.append(name)
            continue
        command = [str(part) for part in step["command"]]
        log_path = log_dir / f"{name}.log"
        write_state(
            state_path,
            status="running",
            category=CATEGORY,
            current=name,
            completed=completed,
            command=command,
            expected_report=str(report_path),
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
                category=CATEGORY,
                current=name,
                completed=completed,
                returncode=result.returncode,
                expected_report=str(report_path),
                log=str(log_path),
            )
            raise SystemExit(result.returncode or 1)
        completed.append(name)

    write_state(
        state_path,
        status="completed",
        category=CATEGORY,
        current=None,
        completed=completed,
    )


if __name__ == "__main__":
    main()
