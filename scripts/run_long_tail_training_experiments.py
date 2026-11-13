from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from regenrec.config import load_config


VARIANTS: dict[str, dict[str, float]] = {
    "baseline": {
        "sampling_alpha": 0.0,
        "item_loss_alpha": 0.0,
        "token_loss_alpha": 0.0,
    },
    "sampling_050": {
        "sampling_alpha": 0.5,
        "item_loss_alpha": 0.0,
        "token_loss_alpha": 0.0,
    },
    "item_loss_050": {
        "sampling_alpha": 0.0,
        "item_loss_alpha": 0.5,
        "token_loss_alpha": 0.0,
    },
    "token_loss_050": {
        "sampling_alpha": 0.0,
        "item_loss_alpha": 0.0,
        "token_loss_alpha": 0.5,
    },
    "hybrid_item025_token025": {
        "sampling_alpha": 0.0,
        "item_loss_alpha": 0.25,
        "token_loss_alpha": 0.25,
    },
    "sampling_010": {
        "sampling_alpha": 0.1,
        "item_loss_alpha": 0.0,
        "token_loss_alpha": 0.0,
    },
    "item_loss_010": {
        "sampling_alpha": 0.0,
        "item_loss_alpha": 0.1,
        "token_loss_alpha": 0.0,
    },
    "token_loss_010": {
        "sampling_alpha": 0.0,
        "item_loss_alpha": 0.0,
        "token_loss_alpha": 0.1,
    },
    "hybrid_item005_token005": {
        "sampling_alpha": 0.0,
        "item_loss_alpha": 0.05,
        "token_loss_alpha": 0.05,
    },
    "hybrid_item010_token010": {
        "sampling_alpha": 0.0,
        "item_loss_alpha": 0.1,
        "token_loss_alpha": 0.1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select long-tail training and diverse decoding on validation, then run "
            "the selected full-data method for seeds 2026/2027/2028"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/v2_amazon2023_long_tail_pilot.yaml"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/amazon2023_musical_instruments_long_tail_training_v1"),
    )
    parser.add_argument("--pilot-train-examples", type=int, default=50000)
    parser.add_argument("--pilot-eval-examples", type=int, default=10000)
    parser.add_argument("--pilot-epochs", type=int, default=3)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_state(path: Path, **payload: object) -> None:
    write_json(path, {"updated_at": utc_now(), **payload})


def run_command(
    command: list[str],
    log_path: Path,
    *,
    root: Path,
    environment: dict[str, str],
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
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {' '.join(command)}"
        )


def read_report(path: Path, *, require_test: bool = False) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    audit = payload.get("data_audit", {})
    if payload.get("synthetic_data_used", audit.get("synthetic_data_used")) is not False:
        raise ValueError(f"report is not verified real data: {path}")
    if "validation" not in payload or (require_test and "test" not in payload):
        raise ValueError(f"report is missing required evaluation splits: {path}")
    return payload


def metric(payload: dict[str, Any], *path: str) -> float:
    value: Any = payload
    for component in path:
        value = value[component]
    return float(value)


def select_training_variant(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = reports["baseline"]["validation"]
    recall_floor = 0.97 * metric(baseline, "recall_at_10")
    ndcg_floor = 0.97 * metric(baseline, "ndcg_at_10")
    rows: list[dict[str, Any]] = []
    for name, report in reports.items():
        validation = report["validation"]
        row = {
            "variant": name,
            "long_tail_training": report["long_tail_training"],
            "recall_at_10": metric(validation, "recall_at_10"),
            "ndcg_at_10": metric(validation, "ndcg_at_10"),
            "tail_recall_at_10": metric(
                validation, "target_groups", "tail", "recall_at_10"
            ),
            "catalog_coverage": metric(validation, "coverage", "catalog_coverage"),
            "tail_catalog_coverage": metric(
                validation, "coverage", "tail_catalog_coverage"
            ),
            "milliseconds_per_example": metric(
                validation, "efficiency", "milliseconds_per_example"
            ),
        }
        row["passes_accuracy_floor"] = (
            row["recall_at_10"] >= recall_floor and row["ndcg_at_10"] >= ndcg_floor
        )
        rows.append(row)
    eligible = [row for row in rows if row["passes_accuracy_floor"]]
    selected = max(
        eligible,
        key=lambda row: (
            row["tail_recall_at_10"],
            row["tail_catalog_coverage"],
            row["catalog_coverage"],
            row["ndcg_at_10"],
            -row["milliseconds_per_example"],
        ),
    )
    return {
        "selection_split": "validation",
        "test_split_accessed": False,
        "accuracy_constraint": {
            "relative_to_pilot_baseline": 0.97,
            "recall_at_10_floor": recall_floor,
            "ndcg_at_10_floor": ndcg_floor,
        },
        "objective_order": [
            "tail_recall_at_10",
            "tail_catalog_coverage",
            "catalog_coverage",
            "ndcg_at_10",
            "negative_latency",
        ],
        "candidates": rows,
        "selected_variant": selected["variant"],
    }


def select_diverse_decoding(grid: dict[str, Any]) -> dict[str, Any]:
    rows = list(grid["results"])
    baseline = next(
        row
        for row in rows
        if int(row["beam_size"]) == 20 and float(row["diversity_weight"]) == 0.0
    )
    recall_floor = 0.97 * metric(baseline, "ranking", "recall_at_10")
    ndcg_floor = 0.97 * metric(baseline, "ranking", "ndcg_at_10")
    eligible = [
        row
        for row in rows
        if metric(row, "ranking", "recall_at_10") >= recall_floor
        and metric(row, "ranking", "ndcg_at_10") >= ndcg_floor
    ]
    selected = max(
        eligible,
        key=lambda row: (
            metric(row, "target_groups", "tail", "recall_at_10"),
            metric(row, "coverage", "tail_catalog_coverage"),
            metric(row, "coverage", "catalog_coverage"),
            metric(row, "ranking", "ndcg_at_10"),
            -metric(row, "efficiency", "milliseconds_per_example"),
        ),
    )
    return {
        "selection_split": "validation",
        "test_split_accessed": False,
        "accuracy_constraint": {
            "relative_to_beam20_no_diversity": 0.97,
            "recall_at_10_floor": recall_floor,
            "ndcg_at_10_floor": ndcg_floor,
        },
        "selected": {
            "beam_size": int(selected["beam_size"]),
            "candidate_count": int(selected["candidate_count"]),
            "num_return_sequences": int(selected["return_count"]),
            "diversity_weight": float(selected["diversity_weight"]),
        },
        "selected_metrics": selected,
    }


def main() -> None:
    args = parse_args()
    if min(args.pilot_train_examples, args.pilot_eval_examples, args.pilot_epochs) <= 0:
        raise ValueError("pilot sizes and epochs must be positive")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    config_path = args.config if args.config.is_absolute() else root / args.config
    artifact_root = (
        args.artifact_root if args.artifact_root.is_absolute() else root / args.artifact_root
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    state_path = artifact_root / "state.json"
    log_dir = artifact_root / "logs"
    config_dir = artifact_root / "configs"
    base_config = load_config(config_path)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = ".:src:scripts"

    pilot_reports: dict[str, dict[str, Any]] = {}
    completed_pilots: list[str] = []
    for name, intervention in VARIANTS.items():
        variant_config = copy.deepcopy(base_config)
        variant_config["seed"] = 2026
        variant_config["data"]["max_train_examples"] = args.pilot_train_examples
        variant_config["data"]["max_eval_examples"] = args.pilot_eval_examples
        variant_config["data"]["train_subset_strategy"] = "reservoir"
        variant_config["data"]["train_subset_seed"] = 2026
        variant_config["generator"]["epochs"] = args.pilot_epochs
        variant_config["generator"]["resume"] = True
        variant_config["long_tail_training"].update(intervention)
        variant_config["evaluation"].update(
            {
                "beam_size": 20,
                "candidate_count": 20,
                "num_return_sequences": 10,
                "diversity_weight": 0.0,
                "splits": ["validation"],
            }
        )
        output_dir = artifact_root / "pilots" / name
        variant_config["output_dir"] = str(output_dir)
        generated_config_path = config_dir / f"pilot_{name}.json"
        write_json(generated_config_path, variant_config)
        report_path = output_dir / "report.json"
        if not report_path.is_file():
            command = [
                sys.executable,
                "scripts/train_amazon2023.py",
                "--config",
                str(generated_config_path),
            ]
            write_state(
                state_path,
                status="running_pilot",
                current=name,
                completed_pilots=completed_pilots,
                command=command,
                log=str(log_dir / f"pilot_{name}.log"),
            )
            run_command(
                command,
                log_dir / f"pilot_{name}.log",
                root=root,
                environment=environment,
            )
        pilot_reports[name] = read_report(report_path)
        completed_pilots.append(name)

    training_selection = select_training_variant(pilot_reports)
    training_selection["created_at"] = utc_now()
    training_selection["base_config_sha256"] = hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    training_selection_path = artifact_root / "training_selection.json"
    write_json(training_selection_path, training_selection)
    selected_name = str(training_selection["selected_variant"])
    selected_config_path = config_dir / f"pilot_{selected_name}.json"
    selected_artifact_dir = artifact_root / "pilots" / selected_name

    grid_path = artifact_root / f"validation_diversity_grid_{selected_name}.json"
    if not grid_path.is_file():
        grid_command = [
            sys.executable,
            "scripts/analyze_beam_coverage_efficiency.py",
            "--config",
            str(selected_config_path),
            "--artifact-dir",
            str(selected_artifact_dir),
            "--split",
            "validation",
            "--max-examples",
            str(args.pilot_eval_examples),
            "--beam-size",
            "20",
            "--beam-size",
            "50",
            "--diversity-weight",
            "0.0",
            "--diversity-weight",
            "0.05",
            "--diversity-weight",
            "0.1",
            "--diversity-weight",
            "0.2",
            "--output",
            str(grid_path),
        ]
        write_state(
            state_path,
            status="running_diversity_grid",
            current=selected_name,
            completed_pilots=completed_pilots,
            command=grid_command,
            log=str(log_dir / "validation_diversity_grid.log"),
        )
        run_command(
            grid_command,
            log_dir / "validation_diversity_grid.log",
            root=root,
            environment=environment,
        )
    decoding_selection = select_diverse_decoding(
        json.loads(grid_path.read_text(encoding="utf-8"))
    )
    decoding_selection["created_at"] = utc_now()
    decoding_selection_path = artifact_root / "decoding_selection.json"
    write_json(decoding_selection_path, decoding_selection)

    formal_config = load_config(selected_config_path)
    formal_config["data"]["max_train_examples"] = None
    formal_config["data"]["max_eval_examples"] = None
    formal_config["data"]["train_subset_strategy"] = "prefix"
    formal_config["generator"]["epochs"] = 10
    formal_config["generator"]["resume"] = True
    formal_config["evaluation"].update(decoding_selection["selected"])
    formal_config["evaluation"]["splits"] = ["validation", "test"]
    formal_root = artifact_root / "formal" / selected_name
    formal_config["output_dir"] = str(formal_root / "seed_2026")
    formal_config_path = config_dir / "formal_selected.json"
    write_json(formal_config_path, formal_config)

    completed_seeds: list[int] = []
    formal_reports: list[Path] = []
    for seed in (2026, 2027, 2028):
        output_dir = formal_root / f"seed_{seed}"
        report_path = output_dir / "report.json"
        formal_reports.append(report_path)
        if not report_path.is_file():
            command = [
                sys.executable,
                "scripts/train_amazon2023.py",
                "--config",
                str(formal_config_path),
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
            ]
            write_state(
                state_path,
                status="running_formal_seed",
                current_seed=seed,
                completed_seeds=completed_seeds,
                selected_training=selected_name,
                selected_decoding=decoding_selection["selected"],
                command=command,
                log=str(log_dir / f"formal_seed_{seed}.log"),
            )
            run_command(
                command,
                log_dir / f"formal_seed_{seed}.log",
                root=root,
                environment=environment,
            )
        read_report(report_path, require_test=True)
        completed_seeds.append(seed)

    summary_path = formal_root / "summary_student_t_95.json"
    summary_command = [
        sys.executable,
        "scripts/summarize_long_tail_multiseed.py",
        "--output",
        str(summary_path),
        "--model-name",
        f"long-tail {selected_name} + Semantic-ID MMR",
    ]
    for report_path in formal_reports:
        summary_command.extend(["--report", str(report_path)])
    baseline_root = root / (
        "artifacts/amazon2023_musical_instruments_blair_v2_top10_multiseed"
    )
    baseline_paths = [baseline_root / f"seed_{seed}" / "report.json" for seed in (2026, 2027, 2028)]
    if all(path.is_file() for path in baseline_paths):
        for path in baseline_paths:
            summary_command.extend(["--baseline-report", str(path)])
    write_state(
        state_path,
        status="summarizing",
        completed_pilots=completed_pilots,
        completed_seeds=completed_seeds,
        command=summary_command,
    )
    run_command(
        summary_command,
        log_dir / "summary.log",
        root=root,
        environment=environment,
    )
    write_state(
        state_path,
        status="completed",
        completed_pilots=completed_pilots,
        completed_seeds=completed_seeds,
        selected_training=selected_name,
        selected_decoding=decoding_selection["selected"],
        summary=str(summary_path),
    )


if __name__ == "__main__":
    main()
