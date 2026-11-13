from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2026, 2027, 2028)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the controlled three-seed explicit Preference Reasoning ablation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/preference_reasoning_ablation_v1"),
    )
    parser.add_argument(
        "--stage", choices=("all", "train", "evaluate", "summarize"), default="all"
    )
    return parser.parse_args()


def write_state(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
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
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    state_path = output_dir / "state.json"
    completed: list[str] = []
    try:
        if args.stage in {"all", "train"}:
            for seed in SEEDS:
                artifact = output_dir / f"mean_pool/seed_{seed}"
                report = artifact / "report.json"
                step = f"train_seed_{seed}"
                if not report.is_file():
                    write_state(state_path, status="running", step=step, completed=completed)
                    run(
                        [
                            sys.executable,
                            "scripts/train_amazon2023.py",
                            "--config",
                            "configs/v2_preference_reasoning_ablation.yaml",
                            "--seed",
                            str(seed),
                            "--output-dir",
                            str(artifact),
                        ],
                        output_dir / f"logs/{step}.log",
                    )
                completed.append(step)

        if args.stage in {"all", "evaluate"}:
            for seed in SEEDS:
                artifact = output_dir / f"mean_pool/seed_{seed}"
                report = artifact / "beam20_test.json"
                step = f"evaluate_seed_{seed}_beam20_test"
                if not report.is_file():
                    write_state(state_path, status="running", step=step, completed=completed)
                    run(
                        [
                            sys.executable,
                            "scripts/analyze_beam_coverage_efficiency.py",
                            "--config",
                            "configs/v2_preference_reasoning_ablation.yaml",
                            "--seed",
                            str(seed),
                            "--artifact-dir",
                            str(artifact),
                            "--split",
                            "test",
                            "--beam-size",
                            "20",
                            "--hit-indicators-dir",
                            str(artifact / "hit_indicators"),
                            "--output",
                            str(report),
                        ],
                        output_dir / f"logs/{step}.log",
                    )
                completed.append(step)

        if args.stage in {"all", "summarize"}:
            step = "summarize_and_explain"
            write_state(state_path, status="running", step=step, completed=completed)
            run(
                [
                    sys.executable,
                    "scripts/analyze_preference_reasoning_ablation.py",
                    "--output-dir",
                    str(output_dir),
                ],
                output_dir / f"logs/{step}.log",
            )
            completed.append(step)
        write_state(state_path, status="completed", step=None, completed=completed)
    except Exception as error:
        write_state(
            state_path,
            status="failed",
            step=locals().get("step"),
            completed=completed,
            error=repr(error),
        )
        raise


if __name__ == "__main__":
    main()
