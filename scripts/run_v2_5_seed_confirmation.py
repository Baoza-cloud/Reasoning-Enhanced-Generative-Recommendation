from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one independent V2.5 Train/Validation seed confirmation"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-examples", type=int, default=5000)
    parser.add_argument("--torch-threads", type=int, default=2)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def completed(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("status") == "completed"


def run(command: list[str], *, environment: dict[str, str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def main() -> None:
    args = parse_args()
    if args.seed not in {2027, 2028}:
        raise ValueError("confirmation runner is restricted to seeds 2027 and 2028")
    if args.train_examples <= 0 or args.torch_threads <= 0:
        raise ValueError("example and thread settings must be positive")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    if completed(state_path):
        print(f"seed {args.seed} confirmation already completed", flush=True)
        return
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = "src:."
    python = sys.executable
    train_dir = output_dir / "train_cache"
    validation_dir = output_dir / "validation_cache"
    fit_dir = output_dir / "fixed_fit"
    strict_report = output_dir / "strict_gate_report.json"
    try:
        write_json(
            state_path,
            {
                "status": "running",
                "phase": "train_cache",
                "seed": args.seed,
                "test_evaluated": False,
            },
        )
        if not completed(train_dir / "state.json"):
            run(
                [
                    python,
                    "scripts/analyze_v2_5_teacher_rescoring.py",
                    "--artifact-dir",
                    str(args.artifact_dir),
                    "--seed",
                    str(args.seed),
                    "--split",
                    "train",
                    "--max-examples",
                    str(args.train_examples),
                    "--selection-strategy",
                    "reservoir",
                    "--batch-size",
                    "128",
                    "--bootstrap-samples",
                    "5000",
                    "--torch-threads",
                    str(args.torch_threads),
                    "--skip-teacher-scoring",
                    "--output-dir",
                    str(train_dir),
                ],
                environment=environment,
            )
        write_json(
            state_path,
            {
                "status": "running",
                "phase": "full_validation_cache",
                "seed": args.seed,
                "test_evaluated": False,
            },
        )
        if not completed(validation_dir / "state.json"):
            run(
                [
                    python,
                    "scripts/analyze_v2_5_teacher_rescoring.py",
                    "--artifact-dir",
                    str(args.artifact_dir),
                    "--seed",
                    str(args.seed),
                    "--split",
                    "validation",
                    "--max-examples",
                    "0",
                    "--selection-strategy",
                    "reservoir",
                    "--batch-size",
                    "128",
                    "--bootstrap-samples",
                    "5000",
                    "--torch-threads",
                    str(args.torch_threads),
                    "--skip-teacher-scoring",
                    "--output-dir",
                    str(validation_dir),
                ],
                environment=environment,
            )
        write_json(
            state_path,
            {
                "status": "running",
                "phase": "frozen_hyperparameter_fit",
                "seed": args.seed,
                "test_evaluated": False,
            },
        )
        if not completed(fit_dir / "state.json"):
            run(
                [
                    python,
                    "scripts/train_v2_5_bounded_residual.py",
                    "--artifact-dir",
                    str(args.artifact_dir),
                    "--train-cache",
                    str(train_dir / "candidate_cache.pt"),
                    "--validation-cache",
                    str(validation_dir / "candidate_cache.pt"),
                    "--output-dir",
                    str(fit_dir),
                    "--seed",
                    str(args.seed),
                    "--epochs",
                    "200",
                    "--learning-rate",
                    "0.03",
                    "--max-negatives",
                    "20",
                    "--distill-weight",
                    "0.2",
                    "--gate-supervision-weight",
                    "1.0",
                    "--gate-positive-weight",
                    "20",
                    "--bootstrap-samples",
                    "5000",
                    "--bounds",
                    "0.5",
                    "--tail-positive-weights",
                    "8",
                    "--preserve-weights",
                    "4",
                    "--deployment-scales",
                    "1.0",
                    "--torch-threads",
                    str(args.torch_threads),
                ],
                environment=environment,
            )
        write_json(
            state_path,
            {
                "status": "running",
                "phase": "strict_single_seed_gate",
                "seed": args.seed,
                "test_evaluated": False,
            },
        )
        if not strict_report.is_file():
            run(
                [
                    python,
                    "scripts/analyze_v2_5_strict_gates.py",
                    "--artifact-dir",
                    str(args.artifact_dir),
                    "--validation-cache",
                    str(validation_dir / "candidate_cache.pt"),
                    "--fit-report",
                    str(fit_dir / "report.json"),
                    "--output",
                    str(strict_report),
                    "--bootstrap-samples",
                    "5000",
                    "--seed",
                    str(args.seed),
                    "--torch-threads",
                    str(args.torch_threads),
                ],
                environment=environment,
            )
        strict = json.loads(strict_report.read_text(encoding="utf-8"))
        write_json(
            state_path,
            {
                "status": "completed",
                "phase": None,
                "seed": args.seed,
                "test_evaluated": False,
                "frozen_hyperparameters": {
                    "training_bound": 0.5,
                    "deployment_scale": 1.0,
                    "tail_positive_weight": 8.0,
                    "preserve_weight": 4.0,
                    "distill_weight": 0.2,
                    "gate_supervision_weight": 1.0,
                    "gate_positive_weight": 20.0,
                },
                "passed_provisional_single_seed_gates": bool(
                    strict.get("selected_provisional_variant")
                ),
                "strict_gate_report": str(strict_report.resolve()),
            },
        )
    except Exception as error:
        write_json(
            state_path,
            {
                "status": "failed",
                "phase": "seed_confirmation",
                "seed": args.seed,
                "test_evaluated": False,
                "error": repr(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
