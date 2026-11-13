from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHASE1_ROOT = ROOT / "artifacts" / "observed_tail_candidate_oracle_v1"
DEFAULT_OUTPUT = ROOT / "artifacts" / "observed_tail_candidate_oracle_multiseed_v1"
SEEDS = (2026, 2027, 2028)
VALIDATION_BEAMS = (20, 50, 100, 200)
MODELS = ("original_v2", "token_loss_010")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue tail candidate-oracle diagnosis through multi-seed test confirmation"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--linger", action="store_true")
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


def artifact_dir(model_name: str, seed: int) -> Path:
    if model_name == "original_v2":
        return (
            ROOT
            / "artifacts"
            / "amazon2023_musical_instruments_blair_v2_top10_multiseed"
            / f"seed_{seed}"
        )
    return (
        ROOT
        / "artifacts"
        / "amazon2023_musical_instruments_long_tail_training_v1"
        / "formal"
        / "token_loss_010"
        / f"seed_{seed}"
    )


def phase1_report(model_name: str, beam_size: int) -> Path:
    return PHASE1_ROOT / model_name / f"beam_{beam_size}" / "report.json"


def report_path(
    output_dir: Path,
    split: str,
    seed: int,
    model_name: str,
    beam_size: int,
) -> Path:
    if split == "validation" and seed == 2026:
        return phase1_report(model_name, beam_size)
    return (
        output_dir
        / split
        / f"seed_{seed}"
        / model_name
        / f"beam_{beam_size}"
        / "report.json"
    )


def report_is_complete(path: Path, split: str, beam_size: int) -> bool:
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    results = report.get("results", [])
    return (
        report.get("split") == split
        and report.get("target_group_filter") == "observed_tail"
        and len(results) == 1
        and results[0].get("beam_size") == beam_size
    )


def wait_for_phase1(state: dict[str, object], state_path: Path) -> None:
    phase1_state_path = PHASE1_ROOT / "state.json"
    while True:
        if phase1_state_path.exists():
            phase1_state = json.loads(phase1_state_path.read_text(encoding="utf-8"))
            phase1_status = phase1_state.get("status")
            state["phase1_status"] = phase1_status
            state["updated_at"] = utc_now()
            write_json(state_path, state)
            if phase1_status == "completed":
                return
            if phase1_status == "failed":
                raise RuntimeError(f"phase 1 failed: {phase1_state.get('error')}")
        time.sleep(30)


def run_evaluation(
    *,
    output_dir: Path,
    split: str,
    seed: int,
    model_name: str,
    beam_size: int,
    batch_size: int,
    environment: dict[str, str],
) -> Path:
    output = report_path(output_dir, split, seed, model_name, beam_size)
    if report_is_complete(output, split, beam_size):
        return output
    task_dir = output.parent
    task_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/analyze_beam_coverage_efficiency.py",
        "--config",
        "configs/v2_amazon2023_top10.yaml",
        "--seed",
        str(seed),
        "--artifact-dir",
        str(artifact_dir(model_name, seed)),
        "--split",
        split,
        "--target-group",
        "observed_tail",
        "--beam-size",
        str(beam_size),
        "--batch-size",
        str(batch_size),
        "--warmup-batches",
        "0",
        "--log-every-batches",
        "100",
        "--hit-indicators-dir",
        str(task_dir / "hit_indicators"),
        "--output",
        str(output),
    ]
    log_path = (
        output_dir
        / "logs"
        / f"{split}_seed_{seed}_{model_name}_beam_{beam_size}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
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
    if completed.returncode != 0 or not report_is_complete(output, split, beam_size):
        raise RuntimeError(
            f"{split} seed={seed} model={model_name} beam={beam_size} failed "
            f"with return code {completed.returncode}"
        )
    return output


def result_row(path: Path) -> dict[str, float | int | str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    result = report["results"][0]
    oracle = result["candidate_oracle"]["target_groups"]["observed_tail"]
    top10 = result["target_groups"]["observed_tail"]
    examples = int(oracle["examples"])
    top10_recall = float(top10.get("recall_at_10", 0.0))
    return {
        "examples": examples,
        "candidate_hits": int(oracle["hits"]),
        "candidate_recall": float(oracle["candidate_recall"]),
        "mean_target_rank_when_hit": (
            float(oracle["mean_target_rank_when_hit"])
            if oracle["mean_target_rank_when_hit"] is not None
            else math.nan
        ),
        "top10_hits": round(top10_recall * examples),
        "top10_recall": top10_recall,
        "latency_ms_per_example": float(
            result["efficiency"]["milliseconds_per_example"]
        ),
        "hit_indicators_path": str(result["hit_indicators_path"]),
        "report_path": str(path.resolve()),
    }


def student_t_ci(values: list[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("Student-t CI requires at least two values")
    critical = {2: 4.303}.get(len(values) - 1, 1.96)
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values)
    margin = critical * standard_deviation / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "ci95_margin": margin,
    }


def summarize_grid(
    output_dir: Path,
    split: str,
    beams: tuple[int, ...],
) -> dict[str, object]:
    per_seed: dict[str, object] = {}
    aggregate: dict[str, object] = {}
    metrics = ("candidate_recall", "top10_recall", "latency_ms_per_example")
    for beam_size in beams:
        beam_rows: dict[str, dict[str, dict[str, float | int | str]]] = {}
        for seed in SEEDS:
            beam_rows[str(seed)] = {
                model_name: result_row(
                    report_path(output_dir, split, seed, model_name, beam_size)
                )
                for model_name in MODELS
            }
        per_seed[str(beam_size)] = beam_rows
        beam_aggregate: dict[str, object] = {}
        for model_name in MODELS:
            beam_aggregate[model_name] = {
                metric: student_t_ci(
                    [float(beam_rows[str(seed)][model_name][metric]) for seed in SEEDS]
                )
                for metric in metrics
            }
        beam_aggregate["token_loss_010_minus_original_v2"] = {
            metric: student_t_ci(
                [
                    float(beam_rows[str(seed)]["token_loss_010"][metric])
                    - float(beam_rows[str(seed)]["original_v2"][metric])
                    for seed in SEEDS
                ]
            )
            for metric in metrics
        }
        aggregate[str(beam_size)] = beam_aggregate
    return {
        "split": split,
        "target_group": "observed_tail",
        "seeds": list(SEEDS),
        "beams": list(beams),
        "per_seed": per_seed,
        "student_t_95": aggregate,
    }


def load_indicator_outcomes(path: str, outcome: str) -> tuple[list[str], np.ndarray]:
    targets: list[str] = []
    hits: list[bool] = []
    with Path(path).open(encoding="utf-8") as handle:
        for expected_index, line in enumerate(handle):
            row = json.loads(line)
            if int(row["example_index"]) != expected_index:
                raise ValueError(f"non-contiguous example index in {path}")
            targets.append(str(row["target_item_id"]))
            rank = row["candidate_rank"] if outcome == "candidate" else row["top_k_rank"]
            hits.append(rank is not None)
    return targets, np.asarray(hits, dtype=np.int8)


def exact_mcnemar(baseline: np.ndarray, treatment: np.ndarray) -> dict[str, float | int]:
    gains = int(np.sum((baseline == 0) & (treatment == 1)))
    losses = int(np.sum((baseline == 1) & (treatment == 0)))
    discordant = gains + losses
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(gains, losses)
        probability = sum(
            math.comb(discordant, index) for index in range(lower + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * probability)
    return {
        "treatment_only_hits": gains,
        "baseline_only_hits": losses,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
    }


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    batch_size = 100
    for start in range(0, samples, batch_size):
        current = min(batch_size, samples - start)
        indices = rng.integers(0, len(differences), size=(current, len(differences)))
        estimates.append(differences[indices].mean(axis=1))
    values = np.concatenate(estimates)
    return {
        "samples": samples,
        "mean_delta": float(differences.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def paired_primary_tests(
    test_summary: dict[str, object],
    selected_beam: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    beam_rows = test_summary["per_seed"][str(selected_beam)]
    result: dict[str, object] = {
        "beam_size": selected_beam,
        "unit": "user (one test example per unique user)",
        "per_seed": {},
    }
    hierarchical: dict[str, list[np.ndarray]] = {"candidate": [], "top10": []}
    for seed in SEEDS:
        seed_rows = beam_rows[str(seed)]
        seed_result: dict[str, object] = {}
        for outcome in ("candidate", "top10"):
            baseline_targets, baseline = load_indicator_outcomes(
                str(seed_rows["original_v2"]["hit_indicators_path"]), outcome
            )
            treatment_targets, treatment = load_indicator_outcomes(
                str(seed_rows["token_loss_010"]["hit_indicators_path"]), outcome
            )
            if baseline_targets != treatment_targets:
                raise ValueError(
                    f"paired targets differ for seed={seed}, outcome={outcome}"
                )
            differences = treatment.astype(np.float64) - baseline.astype(np.float64)
            hierarchical[outcome].append(differences)
            seed_result[outcome] = {
                "examples": len(differences),
                "mcnemar": exact_mcnemar(baseline, treatment),
                "paired_user_bootstrap": paired_bootstrap_ci(
                    differences,
                    samples=bootstrap_samples,
                    seed=seed + (0 if outcome == "candidate" else 10000),
                ),
            }
        result["per_seed"][str(seed)] = seed_result

    hierarchical_result: dict[str, object] = {}
    rng = np.random.default_rng(90210)
    batch_size = 50
    for outcome, seed_differences in hierarchical.items():
        matrix = np.stack(seed_differences)
        estimates: list[np.ndarray] = []
        for start in range(0, bootstrap_samples, batch_size):
            current = min(batch_size, bootstrap_samples - start)
            seed_indices = rng.integers(0, matrix.shape[0], size=(current, matrix.shape[0]))
            user_indices = rng.integers(0, matrix.shape[1], size=(current, matrix.shape[1]))
            batch_estimates = np.empty(current, dtype=np.float64)
            for row in range(current):
                sampled = matrix[seed_indices[row]][:, user_indices[row]]
                batch_estimates[row] = sampled.mean()
            estimates.append(batch_estimates)
        values = np.concatenate(estimates)
        hierarchical_result[outcome] = {
            "samples": bootstrap_samples,
            "mean_delta": float(matrix.mean()),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
    result["hierarchical_seed_user_bootstrap"] = hierarchical_result
    return result


def run(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("batch-size and bootstrap-samples must be positive")
    output_dir = args.output_dir.resolve()
    state_path = output_dir / "state.json"
    state: dict[str, object] = {
        "status": "waiting_for_seed_2026",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "phase1_status": None,
        "completed_tasks": [],
        "current_task": None,
        "selected_test_beam": None,
        "error": None,
    }
    write_json(state_path, state)
    wait_for_phase1(state, state_path)

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = "src:."
    validation_tasks = [
        (seed, model_name, beam_size)
        for seed in (2027, 2028)
        for model_name in MODELS
        for beam_size in VALIDATION_BEAMS
    ]
    state["status"] = "running_validation_multiseed"
    state["total_validation_tasks"] = len(validation_tasks)
    write_json(state_path, state)
    for seed, model_name, beam_size in validation_tasks:
        task_name = f"validation/seed_{seed}/{model_name}/beam_{beam_size}"
        state["current_task"] = task_name
        state["updated_at"] = utc_now()
        write_json(state_path, state)
        run_evaluation(
            output_dir=output_dir,
            split="validation",
            seed=seed,
            model_name=model_name,
            beam_size=beam_size,
            batch_size=args.batch_size,
            environment=environment,
        )
        state["completed_tasks"].append(task_name)
        state["updated_at"] = utc_now()
        write_json(state_path, state)

    validation_summary = summarize_grid(output_dir, "validation", VALIDATION_BEAMS)
    write_json(output_dir / "validation_summary.json", validation_summary)
    aggregate = validation_summary["student_t_95"]
    selected_beam = max(
        VALIDATION_BEAMS,
        key=lambda beam: (
            float(aggregate[str(beam)]["token_loss_010"]["candidate_recall"]["mean"]),
            -beam,
        ),
    )
    state["selected_test_beam"] = selected_beam
    state["selection_rule"] = (
        "maximize token_loss_010 mean validation Observed-Tail candidate recall; "
        "ties choose the smaller beam"
    )
    state["status"] = "running_locked_test"
    state["updated_at"] = utc_now()
    write_json(state_path, state)

    test_beams = tuple(dict.fromkeys((20, selected_beam)))
    test_tasks = [
        (seed, model_name, beam_size)
        for seed in SEEDS
        for model_name in MODELS
        for beam_size in test_beams
    ]
    state["total_test_tasks"] = len(test_tasks)
    write_json(state_path, state)
    for seed, model_name, beam_size in test_tasks:
        task_name = f"test/seed_{seed}/{model_name}/beam_{beam_size}"
        state["current_task"] = task_name
        state["updated_at"] = utc_now()
        write_json(state_path, state)
        run_evaluation(
            output_dir=output_dir,
            split="test",
            seed=seed,
            model_name=model_name,
            beam_size=beam_size,
            batch_size=args.batch_size,
            environment=environment,
        )
        state["completed_tasks"].append(task_name)
        state["updated_at"] = utc_now()
        write_json(state_path, state)

    test_summary = summarize_grid(output_dir, "test", test_beams)
    paired_tests = paired_primary_tests(
        test_summary,
        selected_beam,
        args.bootstrap_samples,
    )
    write_json(output_dir / "test_summary.json", test_summary)
    write_json(output_dir / "paired_primary_tests.json", paired_tests)
    write_json(
        output_dir / "final_summary.json",
        {
            "created_at": utc_now(),
            "dataset": "McAuley-Lab/Amazon-Reviews-2023",
            "synthetic_data_used": False,
            "target_group": "observed_tail",
            "validation_beam_selection": {
                "selected_beam": selected_beam,
                "rule": state["selection_rule"],
            },
            "validation_summary": validation_summary,
            "locked_test_summary": test_summary,
            "paired_primary_tests": paired_tests,
        },
    )
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
