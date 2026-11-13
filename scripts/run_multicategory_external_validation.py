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
        description="Run frozen V2 transport validation on Industrial & Scientific"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/multicategory_external_validation_v1"),
    )
    parser.add_argument(
        "--stage", choices=("all", "train", "evaluate", "summarize"), default="all"
    )
    return parser.parse_args()


def write_state(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {"updated_at": datetime.now(timezone.utc).isoformat(), **payload}
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


def original_artifact(seed: int) -> Path:
    if seed == 2026:
        return ROOT / "artifacts/amazon2023_industrial_scientific_blair_v2_seed2026"
    return ROOT / f"artifacts/amazon2023_industrial_scientific_blair_v2_multiseed/seed_{seed}"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    category_root = output_dir / "Industrial_and_Scientific"
    state_path = output_dir / "state.json"
    completed: list[str] = []
    try:
        if args.stage in {"all", "train"}:
            for seed in SEEDS:
                artifact = category_root / f"token_loss_010/seed_{seed}"
                report = artifact / "report.json"
                step = f"train_industrial_token_loss_seed_{seed}"
                if not report.is_file():
                    write_state(state_path, status="running", step=step, completed=completed)
                    run(
                        [
                            sys.executable,
                            "scripts/train_amazon2023.py",
                            "--config",
                            "configs/v2_industrial_token_loss_external.yaml",
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
                base_output = category_root / f"original_v2/seed_{seed}/beam20_test.json"
                step = f"evaluate_industrial_original_seed_{seed}"
                if not base_output.is_file():
                    write_state(state_path, status="running", step=step, completed=completed)
                    run(
                        [
                            sys.executable,
                            "scripts/analyze_beam_coverage_efficiency.py",
                            "--config",
                            "configs/v2_amazon2023_industrial_scientific_multiseed.yaml",
                            "--seed",
                            str(seed),
                            "--artifact-dir",
                            str(original_artifact(seed)),
                            "--split",
                            "test",
                            "--beam-size",
                            "20",
                            "--output",
                            str(base_output),
                        ],
                        output_dir / f"logs/{step}.log",
                    )
                completed.append(step)

                token_artifact = category_root / f"token_loss_010/seed_{seed}"
                rerank_dir = token_artifact / "candidate_reranking_test"
                rerank_report = rerank_dir / "report.json"
                step = f"evaluate_industrial_frozen_modes_seed_{seed}"
                if not rerank_report.is_file():
                    write_state(state_path, status="running", step=step, completed=completed)
                    run(
                        [
                            sys.executable,
                            "scripts/analyze_candidate_reranking.py",
                            "--config",
                            "configs/v2_industrial_token_loss_external.yaml",
                            "--artifact-dir",
                            str(token_artifact),
                            "--seed",
                            str(seed),
                            "--split",
                            "test",
                            "--mode",
                            "evaluate",
                            "--beam-size",
                            "20",
                            "--candidate-count",
                            "20",
                            "--retrieval-count",
                            "50",
                            "--top-k",
                            "10",
                            "--retrieval-weight",
                            "0.25",
                            "--frequency-alpha",
                            "0.10",
                            "--quota-profile",
                            "tail_20",
                            "--quota-strength",
                            "0.01",
                            "--evaluation-profile",
                            "frozen_v2_modes",
                            "--output-dir",
                            str(rerank_dir),
                        ],
                        output_dir / f"logs/{step}.log",
                    )
                completed.append(step)

        if args.stage in {"all", "summarize"}:
            step = "summarize_cross_category_transport"
            write_state(state_path, status="running", step=step, completed=completed)
            run(
                [
                    sys.executable,
                    "scripts/summarize_multicategory_external_validation.py",
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
