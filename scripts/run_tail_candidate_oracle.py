from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "observed_tail_candidate_oracle_v1"
MODELS = {
    "original_v2": ROOT
    / "artifacts"
    / "amazon2023_musical_instruments_blair_v2_top10_multiseed"
    / "seed_2026",
    "token_loss_010": ROOT
    / "artifacts"
    / "amazon2023_musical_instruments_long_tail_training_v1"
    / "formal"
    / "token_loss_010"
    / "seed_2026",
}
BEAM_SIZES = (20, 50, 100, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable Observed-Tail candidate-oracle diagnostics"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--linger",
        action="store_true",
        help="Remain alive after completion/failure so a launchd submit job is not restarted",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def report_is_complete(path: Path, beam_size: int) -> bool:
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        report.get("split") == "validation"
        and report.get("target_group_filter") == "observed_tail"
        and report.get("results", [{}])[0].get("beam_size") == beam_size
    )


def summarize(output_dir: Path) -> dict[str, object]:
    summary: dict[str, object] = {
        "created_at": utc_now(),
        "seed": 2026,
        "split": "validation",
        "target_group": "observed_tail",
        "beam_sizes": list(BEAM_SIZES),
        "models": {},
        "comparison_token_loss_010_minus_original_v2": {},
    }
    models_summary: dict[str, object] = {}
    for model_name in MODELS:
        beam_summary: dict[str, object] = {}
        for beam_size in BEAM_SIZES:
            report_path = output_dir / model_name / f"beam_{beam_size}" / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            result = report["results"][0]
            oracle = result["candidate_oracle"]["target_groups"]["observed_tail"]
            top_k = result["target_groups"]["observed_tail"]
            beam_summary[str(beam_size)] = {
                "examples": oracle["examples"],
                "candidate_hits": oracle["hits"],
                "candidate_recall": oracle["candidate_recall"],
                "mean_target_rank_when_hit": oracle["mean_target_rank_when_hit"],
                "top10_recall": top_k.get("recall_at_10", 0.0),
                "top10_hits": round(
                    float(top_k.get("recall_at_10", 0.0)) * int(oracle["examples"])
                ),
                "milliseconds_per_example": result["efficiency"][
                    "milliseconds_per_example"
                ],
                "report": str(report_path.resolve()),
            }
        models_summary[model_name] = beam_summary
    summary["models"] = models_summary

    comparison: dict[str, object] = {}
    baseline = models_summary["original_v2"]
    treatment = models_summary["token_loss_010"]
    for beam_size in BEAM_SIZES:
        key = str(beam_size)
        baseline_row = baseline[key]
        treatment_row = treatment[key]
        comparison[key] = {
            "candidate_hits_delta": (
                treatment_row["candidate_hits"] - baseline_row["candidate_hits"]
            ),
            "candidate_recall_delta": (
                treatment_row["candidate_recall"] - baseline_row["candidate_recall"]
            ),
            "top10_hits_delta": treatment_row["top10_hits"] - baseline_row["top10_hits"],
            "top10_recall_delta": (
                treatment_row["top10_recall"] - baseline_row["top10_recall"]
            ),
        }
    summary["comparison_token_loss_010_minus_original_v2"] = comparison
    return summary


def run(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    output_dir = args.output_dir.resolve()
    state_path = output_dir / "state.json"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (model_name, artifact_dir, beam_size)
        for model_name, artifact_dir in MODELS.items()
        for beam_size in BEAM_SIZES
    ]
    state: dict[str, object] = {
        "status": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "completed_tasks": [],
        "total_tasks": len(tasks),
        "current_task": None,
        "error": None,
    }
    write_json(state_path, state)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = "src:."

    for model_name, artifact_dir, beam_size in tasks:
        task_name = f"{model_name}/beam_{beam_size}"
        task_dir = output_dir / model_name / f"beam_{beam_size}"
        report_path = task_dir / "report.json"
        if report_is_complete(report_path, beam_size):
            state["completed_tasks"].append(task_name)
            state["updated_at"] = utc_now()
            write_json(state_path, state)
            continue
        task_dir.mkdir(parents=True, exist_ok=True)
        state["current_task"] = task_name
        state["updated_at"] = utc_now()
        write_json(state_path, state)
        command = [
            sys.executable,
            "scripts/analyze_beam_coverage_efficiency.py",
            "--config",
            "configs/v2_amazon2023_top10.yaml",
            "--seed",
            "2026",
            "--artifact-dir",
            str(artifact_dir),
            "--split",
            "validation",
            "--target-group",
            "observed_tail",
            "--beam-size",
            str(beam_size),
            "--batch-size",
            str(args.batch_size),
            "--warmup-batches",
            "0",
            "--log-every-batches",
            "100",
            "--hit-indicators-dir",
            str(task_dir / "hit_indicators"),
            "--output",
            str(report_path),
        ]
        with (log_dir / f"{model_name}_beam_{beam_size}.log").open(
            "a", encoding="utf-8"
        ) as log_handle:
            log_handle.write(f"\n[{utc_now()}] command={json.dumps(command)}\n")
            log_handle.flush()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 or not report_is_complete(report_path, beam_size):
            raise RuntimeError(
                f"task {task_name} failed with return code {completed.returncode}"
            )
        state["completed_tasks"].append(task_name)
        state["updated_at"] = utc_now()
        write_json(state_path, state)

    write_json(output_dir / "summary.json", summarize(output_dir))
    state["status"] = "completed"
    state["current_task"] = None
    state["completed_at"] = utc_now()
    state["updated_at"] = utc_now()
    write_json(state_path, state)


def main() -> None:
    args = parse_args()
    error: BaseException | None = None
    try:
        run(args)
    except BaseException as exc:
        error = exc
        state_path = args.output_dir.resolve() / "state.json"
        try:
            state = (
                json.loads(state_path.read_text(encoding="utf-8"))
                if state_path.exists()
                else {}
            )
            state.update(
                {
                    "status": "failed",
                    "updated_at": utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            write_json(state_path, state)
        except Exception:
            pass
    if args.linger:
        while True:
            time.sleep(3600)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
