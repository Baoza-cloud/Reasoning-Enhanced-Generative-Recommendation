from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Protocol

from regenrec.baselines import FirstOrderMarkovRecommender, PopularityRecommender
from regenrec.config import load_config
from regenrec.data import SequenceExample, iter_sequence_examples_jsonl
from regenrec.metrics import ranking_metrics


class Recommender(Protocol):
    def recommend(
        self,
        example: SequenceExample,
        *,
        k: int,
        exclude_history: bool = False,
    ) -> list[str]: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deterministic V1 baselines")
    parser.add_argument("--config", default="configs/v1_amazon2023.yaml")
    parser.add_argument(
        "--output",
        help="Defaults to <output_dir>/baseline_report.json from the training config",
    )
    parser.add_argument(
        "--exclude-history",
        action="store_true",
        help="Filter previously interacted items (off by default to match the generative run)",
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_model(
    model: Recommender,
    path: str | Path,
    *,
    ks: tuple[int, ...],
    exclude_history: bool,
) -> dict[str, float | int]:
    predictions: list[list[str]] = []
    targets: list[str] = []
    recommended_items: set[str] = set()
    max_k = max(ks)
    for example in iter_sequence_examples_jsonl(path):
        ranked_items = model.recommend(
            example,
            k=max_k,
            exclude_history=exclude_history,
        )
        predictions.append(ranked_items)
        targets.append(example.target_item_id)
        recommended_items.update(ranked_items)
    metrics: dict[str, float | int] = ranking_metrics(predictions, targets, ks=ks)
    metrics["examples"] = len(targets)
    metrics[f"unique_recommended_at_{max_k}"] = len(recommended_items)
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    audit_path = Path(data_config["data_audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("synthetic_data_used") is not False:
        raise ValueError("baseline evaluation requires a passed real-data provenance audit")

    train_path = Path(data_config["train_path"])
    train_examples = list(iter_sequence_examples_jsonl(train_path))
    models: dict[str, Recommender] = {
        "popularity": PopularityRecommender().fit(train_examples),
        "first_order_markov": FirstOrderMarkovRecommender().fit(train_examples),
    }
    evaluation_config = config.get("evaluation", {})
    ks = tuple(sorted(set(int(k) for k in evaluation_config.get("ks", [5, 10]))))
    split_paths = {
        "validation": Path(data_config["valid_path"]),
        "test": Path(data_config["test_path"]),
    }
    report: dict[str, object] = {
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "category": data_config["category"],
        "synthetic_data_used": False,
        "fit_split": "train",
        "exclude_history": args.exclude_history,
        "ks": list(ks),
        "provenance": {
            "audit_path": str(audit_path.resolve()),
            "audit_sha256": sha256_file(audit_path),
            "train_path": str(train_path.resolve()),
            "train_sha256": sha256_file(train_path),
            **{f"{split}_sha256": sha256_file(path) for split, path in split_paths.items()},
        },
        "train_examples": len(train_examples),
        "models": {},
    }
    model_reports = report["models"]
    assert isinstance(model_reports, dict)
    for name, model in models.items():
        print(f"evaluate baseline={name}")
        model_reports[name] = {
            split: evaluate_model(
                model,
                path,
                ks=ks,
                exclude_history=args.exclude_history,
            )
            for split, path in split_paths.items()
        }

    output_path = (
        Path(args.output) if args.output else Path(config["output_dir"]) / "baseline_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
