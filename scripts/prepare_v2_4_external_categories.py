from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORIES = ("All_Beauty", "Toys_and_Games", "Sports_and_Outdoors")
UNISREC_NAMES = {
    "All_Beauty": "Amazon2023Beauty",
    "Toys_and_Games": "Amazon2023Toys",
    "Sports_and_Outdoors": "Amazon2023Sports",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, audit, embed, and export V2.4 external Amazon categories"
    )
    parser.add_argument("--category", action="append", choices=DEFAULT_CATEGORIES)
    parser.add_argument(
        "--state-dir", type=Path, default=Path("artifacts/v2_4_external_preparation")
    )
    parser.add_argument("--minimum-free-gib", type=float, default=4.0)
    return parser.parse_args()


def write_state(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"updated_at": datetime.now(timezone.utc).isoformat(), **payload},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(command: list[str], log_path: Path, *, offline: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    if offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
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


def require_disk_space(minimum_free_gib: float) -> float:
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"only {free_gib:.2f} GiB free; refusing external-category preparation below "
            f"the {minimum_free_gib:.2f} GiB safety floor"
        )
    return free_gib


def main() -> None:
    args = parse_args()
    if args.minimum_free_gib <= 0:
        raise ValueError("minimum-free-gib must be positive")
    categories = tuple(args.category or DEFAULT_CATEGORIES)
    state_dir = args.state_dir if args.state_dir.is_absolute() else ROOT / args.state_dir
    state_path = state_dir / "state.json"
    completed: list[str] = []
    step = "initialize"
    try:
        for category in categories:
            free_gib = require_disk_space(args.minimum_free_gib)
            processed = ROOT / f"data/amazon2023/processed/{category}/last_out_v2"
            steps = (
                (
                    "prepare",
                    processed / "manifest.json",
                    [
                        sys.executable,
                        "scripts/prepare_amazon2023_v2.py",
                        "--category",
                        category,
                        "--split-strategy",
                        "last_out",
                        "--data-dir",
                        "data/amazon2023/raw",
                        "--output-dir",
                        str(processed),
                        "--download",
                        "--download-timeout",
                        "600",
                    ],
                    False,
                ),
                (
                    "audit",
                    processed / "data_audit.json",
                    [
                        sys.executable,
                        "scripts/audit_amazon2023.py",
                        "--processed-dir",
                        str(processed),
                    ],
                    False,
                ),
                (
                    "embed_blair",
                    processed / "item_embeddings_blair.pt",
                    [
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
                        "128",
                    ],
                    True,
                ),
                (
                    "export_unisrec",
                    ROOT
                    / "external/UniSRec/dataset/downstream"
                    / UNISREC_NAMES[category]
                    / "manifest.json",
                    [
                        sys.executable,
                        "scripts/export_unisrec_dataset.py",
                        "--processed-dir",
                        str(processed),
                        "--embedding-path",
                        str(processed / "item_embeddings_blair.pt"),
                        "--output-root",
                        "external/UniSRec/dataset/downstream",
                        "--dataset-name",
                        UNISREC_NAMES[category],
                    ],
                    False,
                ),
            )
            for action, expected, command, offline in steps:
                step = f"{category}:{action}"
                if expected.is_file() and expected.stat().st_size > 0:
                    completed.append(step)
                    continue
                write_state(
                    state_path,
                    status="running",
                    step=step,
                    completed=completed,
                    categories=list(categories),
                    free_gib_before_step=free_gib,
                )
                run(
                    command,
                    state_dir / "logs" / f"{category}_{action}.log",
                    offline=offline,
                )
                if not expected.is_file() or expected.stat().st_size == 0:
                    raise RuntimeError(f"step did not create expected artifact: {expected}")
                completed.append(step)
        write_state(
            state_path,
            status="completed",
            step=None,
            completed=completed,
            categories=list(categories),
            free_gib=require_disk_space(args.minimum_free_gib),
        )
    except Exception as error:
        write_state(
            state_path,
            status="failed",
            step=step,
            completed=completed,
            categories=list(categories),
            error=repr(error),
        )
        raise


if __name__ == "__main__":
    main()
