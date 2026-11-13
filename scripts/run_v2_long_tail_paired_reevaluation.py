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
        description="Reevaluate original V2 and token-corrected models with one harness"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/v2_amazon2023_top10.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/v2_long_tail_paired_reevaluation"),
    )
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


def valid_analysis(path: Path, *, seed: int, split: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        payload.get("synthetic_data_used") is False
        and int(payload.get("seed", -1)) == seed
        and payload.get("split") == split
        and len(payload.get("results", [])) == 2
    )


def run_command(
    command: list[str],
    *,
    root: Path,
    environment: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {' '.join(command)}"
        )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_root = (
        args.output_root if args.output_root.is_absolute() else root / args.output_root
    )
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "state.json"
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = ".:src:scripts"
    model_roots = {
        "original_v2": root
        / "artifacts/amazon2023_musical_instruments_blair_v2_top10_multiseed",
        "token_loss_010": root
        / (
            "artifacts/amazon2023_musical_instruments_long_tail_training_v1/"
            "formal/token_loss_010"
        ),
    }
    completed: list[str] = []
    analyses: dict[str, list[Path]] = {name: [] for name in model_roots}
    total_jobs = len(model_roots) * 3 * 2
    for seed in (2026, 2027, 2028):
        for split in ("validation", "test"):
            for model_name, model_root in model_roots.items():
                artifact_dir = model_root / f"seed_{seed}"
                if not (artifact_dir / "generator.pt").is_file():
                    raise FileNotFoundError(f"missing generator artifact: {artifact_dir}")
                output_path = output_root / model_name / f"seed_{seed}" / f"{split}.json"
                analyses[model_name].append(output_path)
                job_name = f"{model_name}_seed_{seed}_{split}"
                if not valid_analysis(output_path, seed=seed, split=split):
                    command = [
                        sys.executable,
                        "scripts/analyze_beam_coverage_efficiency.py",
                        "--config",
                        str(config_path),
                        "--seed",
                        str(seed),
                        "--artifact-dir",
                        str(artifact_dir),
                        "--split",
                        split,
                        "--max-examples",
                        "0",
                        "--beam-size",
                        "20",
                        "--diversity-weight",
                        "0.0",
                        "--diversity-weight",
                        "0.1",
                        "--output",
                        str(output_path),
                    ]
                    write_state(
                        state_path,
                        status="running",
                        current=job_name,
                        completed_jobs=completed,
                        completed_count=len(completed),
                        total_jobs=total_jobs,
                        command=command,
                        log=str(output_root / "logs" / f"{job_name}.log"),
                    )
                    run_command(
                        command,
                        root=root,
                        environment=environment,
                        log_path=output_root / "logs" / f"{job_name}.log",
                    )
                    if not valid_analysis(output_path, seed=seed, split=split):
                        raise RuntimeError(f"invalid analysis output: {output_path}")
                completed.append(job_name)

    summary_path = output_root / "paired_summary_student_t_95.json"
    command = [
        sys.executable,
        "scripts/summarize_paired_long_tail_reevaluation.py",
        "--output",
        str(summary_path),
    ]
    for path in analyses["original_v2"]:
        command.extend(["--baseline-analysis", str(path)])
    for path in analyses["token_loss_010"]:
        command.extend(["--treatment-analysis", str(path)])
    write_state(
        state_path,
        status="summarizing",
        completed_jobs=completed,
        completed_count=len(completed),
        total_jobs=total_jobs,
        command=command,
    )
    run_command(
        command,
        root=root,
        environment=environment,
        log_path=output_root / "logs" / "summary.log",
    )
    write_state(
        state_path,
        status="completed",
        completed_jobs=completed,
        completed_count=len(completed),
        total_jobs=total_jobs,
        summary=str(summary_path),
    )


if __name__ == "__main__":
    main()
