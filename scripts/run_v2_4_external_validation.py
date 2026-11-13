from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from regenrec.metrics import mean_confidence_interval


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2026, 2027, 2028)
CATEGORIES = ("All_Beauty", "Toys_and_Games", "Sports_and_Outdoors")
PREREQUISITES = (
    ROOT / "artifacts/v2_4_external_preparation/state.json",
    ROOT / "artifacts/v2_4_hybrid_scorer_multiseed/state.json",
    ROOT / "artifacts/unisrec_amazon2023_musical_multiseed/state.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired original/dynamic V2 validation on three external categories"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v2_4_external_validation"),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--minimum-free-gib", type=float, default=3.0)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_state(path: Path, **payload: object) -> None:
    write_json(
        path,
        {"updated_at": datetime.now(timezone.utc).isoformat(), **payload},
    )


def wait_for_prerequisites(output_state: Path, poll_seconds: int) -> None:
    while True:
        waiting: list[str] = []
        for path in PREREQUISITES:
            if not path.is_file():
                waiting.append(str(path))
                continue
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("status") == "failed":
                raise RuntimeError(f"prerequisite failed: {path}: {state}")
            if state.get("status") != "completed":
                waiting.append(str(path))
        if not waiting:
            return
        write_state(output_state, status="waiting_for_prerequisites", waiting=waiting)
        time.sleep(poll_seconds)


def require_disk_space(minimum_free_gib: float) -> float:
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"only {free_gib:.2f} GiB free; external validation safety floor is "
            f"{minimum_free_gib:.2f} GiB"
        )
    return free_gib


def category_config(
    category: str,
    *,
    forgetting_mode: str,
    pretrained_artifact_dir: Path | None,
    codebook_size: int,
) -> dict[str, object]:
    processed = f"data/amazon2023/processed/{category}/last_out_v2"
    return {
        "seed": 2026,
        "device": "cpu",
        "data": {
            "category": category,
            "train_path": f"{processed}/train.jsonl",
            "valid_path": f"{processed}/valid.jsonl",
            "test_path": f"{processed}/test.jsonl",
            "item_embeddings_path": f"{processed}/item_embeddings_blair.pt",
            "data_audit_path": f"{processed}/data_audit.json",
            "max_train_examples": None,
            "max_eval_examples": None,
            "train_subset_strategy": "prefix",
            "train_subset_seed": 2026,
            "max_history": 50,
            "num_behaviors": 4,
            "num_time_buckets": 32,
        },
        "rqvae": {
            "hidden_dim": 512,
            "code_dim": 32,
            "num_levels": 3,
            "codebook_size": codebook_size,
            "commitment_weight": 0.25,
            "epochs": 50,
            "batch_size": 1024,
            "learning_rate": 0.0001,
            "kmeans_init": True,
            "kmeans_iters": 20,
            "kmeans_sample_size": 20000,
            "use_ema": True,
            "ema_decay": 0.99,
            "ema_epsilon": 0.00001,
            "dead_code_threshold": 2.0,
            "normalize_reconstruction": True,
            "pretrained_artifact_dir": str(pretrained_artifact_dir)
            if pretrained_artifact_dir
            else None,
        },
        "generator": {
            "model_dim": 128,
            "num_heads": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "feedforward_dim": 512,
            "dropout": 0.1,
            "short_term_window": 10,
            "use_behavior_intensity": True,
            "preference_reasoning_mode": "full",
            "history_forgetting_mode": forgetting_mode,
            "loss_type": "global_cross_entropy",
            "resume": True,
            "epochs": 10,
            "batch_size": 128,
            "learning_rate": 0.0003,
            "log_every_batches": 100,
        },
        "long_tail_training": {
            "sampling_alpha": 0.0,
            "item_loss_alpha": 0.0,
            "token_loss_alpha": 0.1,
            "max_weight": 5.0,
            "head_fraction": 0.2,
        },
        "quality_gate": {
            "min_level_usage_ratio": 0.5,
            "min_unique_tuple_ratio": 0.65,
        },
        "evaluation": {
            "beam_size": 20,
            "candidate_count": 20,
            "num_return_sequences": 10,
            "diversity_weight": 0.0,
            "renormalize_allowed_tokens": False,
            "ks": [5, 10],
            "splits": ["validation", "test"],
        },
        "output_dir": "overridden-by-runner",
    }


def run(command: list[str], log_path: Path, threads: int) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = str(threads)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def validate_report(path: Path, category: str, seed: int, mode: str) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["category"] != category or int(report["seed"]) != seed:
        raise ValueError(f"category/seed mismatch: {path}")
    if report["generator_config"].get("history_forgetting_mode", "none") != mode:
        raise ValueError(f"forgetting mode mismatch: {path}")
    audit = report["data_audit"]
    if audit.get("status") != "passed" or audit.get("synthetic_data_used") is not False:
        raise ValueError(f"real-data audit failed: {path}")


def nested_float(payload: dict[str, object], path: tuple[str, ...]) -> float:
    value: object = payload
    for key in path:
        value = value[key]
    return float(value)


def summarize_category(output_dir: Path, category: str) -> None:
    reports = {
        variant: {
            seed: json.loads(
                (output_dir / category / variant / f"seed_{seed}/report.json").read_text()
            )
            for seed in SEEDS
        }
        for variant in ("baseline", "adaptive_forgetting")
    }
    metric_paths = {
        "recall_at_5": ("recall_at_5",),
        "ndcg_at_5": ("ndcg_at_5",),
        "recall_at_10": ("recall_at_10",),
        "ndcg_at_10": ("ndcg_at_10",),
        "legacy_tail_recall_at_10": ("target_groups", "tail", "recall_at_10"),
    }
    summary: dict[str, object] = {
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "category": category,
        "synthetic_data_used": False,
        "seeds": list(SEEDS),
        "test_used_for_parameter_selection": False,
    }
    for split in ("validation", "test"):
        split_metrics: dict[str, object] = {}
        for name, suffix in metric_paths.items():
            path = (split, *suffix)
            baseline = [nested_float(reports["baseline"][seed], path) for seed in SEEDS]
            dynamic = [
                nested_float(reports["adaptive_forgetting"][seed], path) for seed in SEEDS
            ]
            delta = [right - left for left, right in zip(baseline, dynamic, strict=True)]
            split_metrics[name] = {
                "baseline": mean_confidence_interval(baseline),
                "adaptive_forgetting": mean_confidence_interval(dynamic),
                "paired_delta": mean_confidence_interval(delta),
            }
        summary[split] = split_metrics
    write_json(output_dir / category / "summary.json", summary)


def main() -> None:
    args = parse_args()
    if min(args.workers, args.threads_per_worker, args.poll_seconds) <= 0:
        raise ValueError("worker and poll settings must be positive")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    state_path = output_dir / "state.json"
    completed: list[str] = []
    try:
        wait_for_prerequisites(state_path, args.poll_seconds)
        for category in CATEGORIES:
            free_gib = require_disk_space(args.minimum_free_gib)
            manifest = json.loads(
                (
                    ROOT
                    / f"data/amazon2023/processed/{category}/last_out_v2/manifest.json"
                ).read_text()
            )
            item_count = int(manifest["items"]["total"])
            codebook_size = 64 if item_count < 4096 else 256
            baseline_seed_dir = output_dir / category / "baseline/seed_2026"
            baseline_seed_report = baseline_seed_dir / "report.json"
            bootstrap_config = category_config(
                category,
                forgetting_mode="none",
                pretrained_artifact_dir=None,
                codebook_size=codebook_size,
            )
            bootstrap_config_path = output_dir / "configs" / f"{category}_bootstrap.json"
            write_json(bootstrap_config_path, bootstrap_config)
            bootstrap_step = f"{category}:baseline_seed_2026_bootstrap_rqvae"
            if not baseline_seed_report.is_file():
                write_state(
                    state_path,
                    status="running",
                    step=bootstrap_step,
                    completed=completed,
                    free_gib=free_gib,
                )
                run(
                    [
                        sys.executable,
                        "scripts/train_amazon2023.py",
                        "--config",
                        str(bootstrap_config_path),
                        "--seed",
                        "2026",
                        "--output-dir",
                        str(baseline_seed_dir),
                    ],
                    output_dir / "logs" / f"{category}_baseline_seed_2026.log",
                    args.threads_per_worker,
                )
            validate_report(baseline_seed_report, category, 2026, "none")
            completed.append(bootstrap_step)

            configs: dict[str, Path] = {}
            for variant, mode in (
                ("baseline", "none"),
                ("adaptive_forgetting", "learned_exponential"),
            ):
                config = category_config(
                    category,
                    forgetting_mode=mode,
                    pretrained_artifact_dir=baseline_seed_dir,
                    codebook_size=codebook_size,
                )
                config_path = output_dir / "configs" / f"{category}_{variant}.json"
                write_json(config_path, config)
                configs[variant] = config_path

            tasks: list[tuple[str, str, int, list[str], Path, Path]] = []
            for variant, mode in (
                ("baseline", "none"),
                ("adaptive_forgetting", "learned_exponential"),
            ):
                for seed in SEEDS:
                    if variant == "baseline" and seed == 2026:
                        continue
                    seed_dir = output_dir / category / variant / f"seed_{seed}"
                    report_path = seed_dir / "report.json"
                    step = f"{category}:{variant}_seed_{seed}"
                    if report_path.is_file():
                        validate_report(report_path, category, seed, mode)
                        completed.append(step)
                        continue
                    command = [
                        sys.executable,
                        "scripts/train_amazon2023.py",
                        "--config",
                        str(configs[variant]),
                        "--seed",
                        str(seed),
                        "--output-dir",
                        str(seed_dir),
                    ]
                    tasks.append(
                        (
                            step,
                            mode,
                            seed,
                            command,
                            output_dir / "logs" / f"{category}_{variant}_seed_{seed}.log",
                            report_path,
                        )
                    )
            write_state(
                state_path,
                status="running",
                step=f"{category}:paired_training",
                completed=completed,
                pending=[task[0] for task in tasks],
            )
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        run,
                        command,
                        log_path,
                        args.threads_per_worker,
                    ): (step, mode, seed, report_path)
                    for step, mode, seed, command, log_path, report_path in tasks
                }
                for future in as_completed(futures):
                    step, mode, seed, report_path = futures[future]
                    future.result()
                    validate_report(report_path, category, seed, mode)
                    completed.append(step)
                    write_state(
                        state_path,
                        status="running",
                        step=f"{category}:paired_training",
                        completed=completed,
                    )
            summarize_category(output_dir, category)
            completed.append(f"{category}:summarize")
        write_state(
            state_path,
            status="completed",
            step=None,
            completed=completed,
        )
    except Exception as error:
        write_state(
            state_path,
            status="failed",
            completed=completed,
            error=repr(error),
        )
        raise


if __name__ == "__main__":
    main()
