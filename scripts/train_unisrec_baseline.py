from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = ROOT / "external/UniSRec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official RUCAIBox UniSRec implementation on exported data"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument(
        "--train-stage",
        choices=("inductive_ft", "transductive_ft"),
        default="transductive_ft",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stopping-step", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("synthetic_data_used") is not False:
        raise ValueError("UniSRec baseline requires an audited real-data export")
    if args.pretrained_checkpoint is not None and not args.pretrained_checkpoint.is_file():
        raise FileNotFoundError(args.pretrained_checkpoint)
    if args.epochs <= 0 or args.stopping_step <= 0:
        raise ValueError("epochs and stopping-step must be positive")

    sys.path.insert(0, str(OFFICIAL_ROOT))
    from finetune import finetune

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(manifest["dataset_name"])
    embedding_dimension = int(manifest["embedding_dimension"])
    pretrained = (
        str(args.pretrained_checkpoint.resolve()) if args.pretrained_checkpoint else ""
    )
    original_directory = Path.cwd()
    try:
        os.chdir(OFFICIAL_ROOT)
        model, dataset, results = finetune(
            dataset_name,
            pretrained_file=pretrained,
            fix_enc=args.pretrained_checkpoint is not None,
            data_path=str(dataset_dir.parent) + "/",
            benchmark_filename=["train", "valid", "test"],
            plm_size=embedding_dimension,
            adaptor_layers=[embedding_dimension, 300],
            train_stage=args.train_stage,
            seed=args.seed,
            reproducibility=True,
            use_gpu=False,
            epochs=args.epochs,
            stopping_step=args.stopping_step,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            topk=[5, 10],
            metrics=["Recall", "NDCG"],
            valid_metric="NDCG@10",
            checkpoint_dir=str(output_dir),
            show_progress=False,
        )
    finally:
        os.chdir(original_directory)
    report = {
        "model": model,
        "official_implementation": "RUCAIBox/UniSRec",
        "official_commit": (
            OFFICIAL_ROOT / ".git/refs/heads/master"
        ).read_text(encoding="utf-8").strip()
        if (OFFICIAL_ROOT / ".git/refs/heads/master").is_file()
        else None,
        "dataset": dataset,
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_manifest_sha256": sha256(manifest_path),
        "synthetic_data_used": False,
        "seed": args.seed,
        "train_stage": args.train_stage,
        "pretrained_checkpoint": str(args.pretrained_checkpoint.resolve())
        if args.pretrained_checkpoint
        else None,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "configuration": {
            "epochs": args.epochs,
            "stopping_step": args.stopping_step,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "embedding_dimension": embedding_dimension,
            "hidden_size": 300,
        },
        "results": results,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
