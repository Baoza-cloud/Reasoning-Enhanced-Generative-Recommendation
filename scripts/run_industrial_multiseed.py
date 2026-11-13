from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


GENERATOR_MODEL_NAME = "Industrial V2.1 global-CE + Beam-20/return-10"
SASREC_MODEL_NAME = "Industrial SASRec full-catalog"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Industrial & Scientific V2 and SASRec multi-seed evaluation"
    )
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"updated_at": utc_now(), **payload}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def run_command(
    command: list[str], log_path: Path, *, root: Path, environment: dict[str, str]
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def validate_report(path: Path, expected_seed: int) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("seed", -1)) != expected_seed:
        raise ValueError(f"report seed mismatch: expected {expected_seed}, path={path}")
    audit = payload.get("data_audit", {})
    synthetic = payload.get(
        "synthetic_data_used",
        audit.get("synthetic_data_used") if isinstance(audit, dict) else None,
    )
    if synthetic is not False:
        raise ValueError(f"report is not verified real data: {path}")
    return True


def summarize(
    report_paths: list[Path],
    output_path: Path,
    model_name: str,
    *,
    root: Path,
    environment: dict[str, str],
    log_path: Path,
) -> None:
    command = [
        sys.executable,
        "scripts/summarize_multiseed.py",
        "--output",
        str(output_path),
        "--model-name",
        model_name,
    ]
    for report_path in report_paths:
        command.extend(["--report", str(report_path)])
    run_command(command, log_path, root=root, environment=environment)


def main() -> None:
    args = parse_args()
    seeds = args.seeds or [2026, 2027, 2028]
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("provide at least two unique random seeds")
    seeds = sorted(seeds)
    if 2026 not in seeds:
        raise ValueError("seed 2026 is required because it is the completed reference run")

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    generator_root = root / "artifacts/amazon2023_industrial_scientific_blair_v2_multiseed"
    sasrec_root = root / "artifacts/amazon2023_industrial_scientific_sasrec_multiseed"
    comparison_root = root / "artifacts/amazon2023_industrial_scientific_multiseed_comparison"
    state_path = comparison_root / "state.json"
    log_dir = comparison_root / "logs"
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

    reference_generator_report = (
        root
        / "artifacts/amazon2023_industrial_scientific_blair_v2_seed2026/report.json"
    )
    reference_sasrec_report = (
        root / "artifacts/amazon2023_industrial_scientific_sasrec_seed2026/report.json"
    )
    if not validate_report(reference_generator_report, 2026):
        raise FileNotFoundError(reference_generator_report)
    if not validate_report(reference_sasrec_report, 2026):
        raise FileNotFoundError(reference_sasrec_report)

    generator_reports: list[Path] = []
    sasrec_reports: list[Path] = []
    completed: list[str] = []
    try:
        for seed in seeds:
            report_path = (
                reference_generator_report
                if seed == 2026
                else generator_root / f"seed_{seed}" / "report.json"
            )
            generator_reports.append(report_path)
            step = f"generator_seed_{seed}"
            if validate_report(report_path, seed):
                completed.append(step)
                continue
            command = [
                sys.executable,
                "scripts/train_amazon2023.py",
                "--config",
                "configs/v2_amazon2023_industrial_scientific_multiseed.yaml",
                "--seed",
                str(seed),
                "--output-dir",
                str(report_path.parent),
            ]
            write_state(
                state_path,
                status="running",
                current=step,
                completed=completed,
                command=command,
            )
            run_command(
                command,
                log_dir / f"{step}.log",
                root=root,
                environment=environment,
            )
            if not validate_report(report_path, seed):
                raise RuntimeError(f"training completed without a valid report: {report_path}")
            completed.append(step)

        generator_summary = generator_root / "summary.json"
        step = "generator_summary"
        write_state(state_path, status="running", current=step, completed=completed)
        summarize(
            generator_reports,
            generator_summary,
            GENERATOR_MODEL_NAME,
            root=root,
            environment=environment,
            log_path=log_dir / f"{step}.log",
        )
        completed.append(step)

        for seed in seeds:
            report_path = (
                reference_sasrec_report
                if seed == 2026
                else sasrec_root / f"seed_{seed}" / "report.json"
            )
            sasrec_reports.append(report_path)
            step = f"sasrec_seed_{seed}"
            if validate_report(report_path, seed):
                completed.append(step)
                continue
            command = [
                sys.executable,
                "scripts/train_sasrec_baseline.py",
                "--config",
                "configs/baseline_sasrec_amazon2023_industrial_scientific.yaml",
                "--seed",
                str(seed),
                "--output-dir",
                str(report_path.parent),
            ]
            write_state(
                state_path,
                status="running",
                current=step,
                completed=completed,
                command=command,
            )
            run_command(
                command,
                log_dir / f"{step}.log",
                root=root,
                environment=environment,
            )
            if not validate_report(report_path, seed):
                raise RuntimeError(f"training completed without a valid report: {report_path}")
            completed.append(step)

        sasrec_summary = sasrec_root / "summary.json"
        step = "sasrec_summary"
        write_state(state_path, status="running", current=step, completed=completed)
        summarize(
            sasrec_reports,
            sasrec_summary,
            SASREC_MODEL_NAME,
            root=root,
            environment=environment,
            log_path=log_dir / f"{step}.log",
        )
        completed.append(step)

        comparison_path = comparison_root / "paired_summary.json"
        step = "paired_comparison"
        command = [
            sys.executable,
            "scripts/compare_multiseed.py",
            "--left-summary",
            str(generator_summary),
            "--right-summary",
            str(sasrec_summary),
            "--output",
            str(comparison_path),
            "--left-name",
            GENERATOR_MODEL_NAME,
            "--right-name",
            SASREC_MODEL_NAME,
        ]
        write_state(
            state_path,
            status="running",
            current=step,
            completed=completed,
            command=command,
        )
        run_command(
            command,
            log_dir / f"{step}.log",
            root=root,
            environment=environment,
        )
        completed.append(step)
        write_state(
            state_path,
            status="completed",
            current=None,
            completed=completed,
            generator_summary=str(generator_summary),
            sasrec_summary=str(sasrec_summary),
            paired_summary=str(comparison_path),
        )
    except Exception as exc:
        write_state(
            state_path,
            status="failed",
            current=locals().get("step"),
            completed=completed,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
