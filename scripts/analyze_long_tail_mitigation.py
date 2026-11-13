from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import torch

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from regenrec.data import SequenceExample, iter_sequence_examples_jsonl
from regenrec.metrics import (
    popularity_calibrated_rerank,
    popularity_partition,
    ranking_metrics,
    recommendation_coverage_metrics,
)
from regenrec.models.generative import GenerativeRecommender
from train_amazon2023 import make_loader


DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune a train-popularity calibration on validation and audit errors on test"
    )
    parser.add_argument("--config", default="configs/v2_amazon2023_top10.yaml")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(
            "artifacts/amazon2023_musical_instruments_blair_v2_top10_multiseed/seed_2026"
        ),
    )
    parser.add_argument("--candidate-beam-size", type=int, default=20)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", action="append", type=float)
    parser.add_argument("--minimum-recall-ratio", type=float, default=0.97)
    parser.add_argument("--head-fraction", type=float, default=0.2)
    parser.add_argument("--max-validation-examples", type=int, default=0)
    parser.add_argument("--max-test-examples", type=int, default=0)
    parser.add_argument("--sample-errors", type=int, default=20)
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(
    config: dict[str, object],
    catalog: SemanticIdCatalog,
    artifact_dir: Path,
    device: torch.device,
) -> tuple[GenerativeRecommender, Path]:
    data_config = config["data"]
    generator_config = config["generator"]
    model = GenerativeRecommender(
        vocab_size=catalog.vocab_size,
        pad_token_id=catalog.special_tokens.pad,
        num_behaviors=data_config["num_behaviors"],
        num_time_buckets=data_config["num_time_buckets"],
        model_dim=generator_config["model_dim"],
        num_heads=generator_config["num_heads"],
        num_encoder_layers=generator_config["num_encoder_layers"],
        num_decoder_layers=generator_config["num_decoder_layers"],
        feedforward_dim=generator_config["feedforward_dim"],
        dropout=generator_config["dropout"],
        short_term_window=generator_config["short_term_window"],
        max_history_length=data_config["max_history"],
        max_target_length=catalog.sequence_length + 2,
        use_behavior_intensity=generator_config.get("use_behavior_intensity", False),
        preference_reasoning_mode=generator_config.get(
            "preference_reasoning_mode", "full"
        ),
        history_forgetting_mode=generator_config.get("history_forgetting_mode", "none"),
    ).to(device)
    model_path = artifact_dir / "generator.pt"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, model_path


def generate_candidates(
    model: GenerativeRecommender,
    split_path: str,
    catalog: SemanticIdCatalog,
    *,
    device: torch.device,
    batch_size: int,
    beam_size: int,
    candidate_count: int,
    limit: int | None,
    log_every_batches: int,
    split_name: str,
) -> tuple[list[list[str]], list[list[float]], list[str], list[SequenceExample], float]:
    loader, count = make_loader(
        split_path,
        catalog,
        batch_size=batch_size,
        limit=limit,
        shuffle=False,
    )
    examples = []
    for index, example in enumerate(iter_sequence_examples_jsonl(split_path)):
        if limit is not None and index >= limit:
            break
        examples.append(example)
    if len(examples) != count:
        raise RuntimeError("example metadata does not align with the evaluation loader")

    candidates: list[list[str]] = []
    scores: list[list[float]] = []
    targets: list[str] = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = {name: tensor.to(device) for name, tensor in raw_batch.items()}
            batch_items, batch_scores = model.generate_beam(
                history_tokens=batch["history_tokens"],
                behavior_types=batch["behavior_types"],
                time_gap_buckets=batch["time_gap_buckets"],
                history_mask=batch["history_mask"],
                catalog=catalog,
                beam_size=beam_size,
                num_return_sequences=candidate_count,
                renormalize_allowed_tokens=False,
                behavior_intensities=batch["behavior_intensities"],
            )
            batch_targets = [
                catalog.resolve(row.tolist()) for row in batch["target_labels"][:, :-1]
            ]
            if any(target is None for target in batch_targets):
                raise RuntimeError("evaluation target could not be resolved in the catalog")
            candidates.extend(batch_items)
            scores.extend(batch_scores)
            targets.extend(str(target) for target in batch_targets)
            if batch_index % log_every_batches == 0:
                print(
                    f"split={split_name} batches={batch_index} examples={len(targets)} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    if len(targets) != count:
        raise RuntimeError(f"evaluated {len(targets)} examples but expected {count}")
    return candidates, scores, targets, examples, elapsed


def group_metrics(
    predictions: Sequence[Sequence[str]],
    targets: Sequence[str],
    group: frozenset[str],
) -> dict[str, float | int]:
    selected = [
        (prediction, target)
        for prediction, target in zip(predictions, targets, strict=True)
        if target in group
    ]
    if not selected:
        return {"examples": 0}
    group_predictions, group_targets = zip(*selected, strict=True)
    return {
        "examples": len(selected),
        **ranking_metrics(group_predictions, group_targets, ks=(5, 10)),
    }


def evaluate_predictions(
    predictions: list[list[str]],
    targets: list[str],
    *,
    catalog: SemanticIdCatalog,
    head_items: frozenset[str],
    tail_items: frozenset[str],
    train_frequencies: Counter[str],
) -> dict[str, object]:
    return {
        "ranking": ranking_metrics(predictions, targets, ks=(5, 10)),
        "target_groups": {
            "head": group_metrics(predictions, targets, head_items),
            "tail": group_metrics(predictions, targets, tail_items),
        },
        "coverage": recommendation_coverage_metrics(
            predictions,
            catalog_items=catalog.item_ids,
            tail_items=tail_items,
            item_frequencies=train_frequencies,
        ),
    }


def evaluate_alpha(
    candidates: list[list[str]],
    scores: list[list[float]],
    targets: list[str],
    *,
    alpha: float,
    top_k: int,
    catalog: SemanticIdCatalog,
    head_items: frozenset[str],
    tail_items: frozenset[str],
    train_frequencies: Counter[str],
) -> tuple[list[list[str]], dict[str, object]]:
    predictions = popularity_calibrated_rerank(
        candidates,
        scores,
        train_frequencies,
        alpha=alpha,
        top_k=top_k,
    )
    return predictions, {
        "alpha": alpha,
        **evaluate_predictions(
            predictions,
            targets,
            catalog=catalog,
            head_items=head_items,
            tail_items=tail_items,
            train_frequencies=train_frequencies,
        ),
    }


def choose_alpha(
    results: list[dict[str, object]], minimum_recall_ratio: float
) -> tuple[float, dict[str, object]]:
    baseline = next(result for result in results if math.isclose(float(result["alpha"]), 0.0))
    baseline_recall = float(baseline["ranking"]["recall_at_10"])
    threshold = baseline_recall * minimum_recall_ratio
    feasible = [
        result for result in results if float(result["ranking"]["recall_at_10"]) >= threshold
    ]
    selected = max(
        feasible,
        key=lambda result: (
            float(result["target_groups"]["tail"]["recall_at_10"]),
            float(result["coverage"]["tail_catalog_coverage"]),
            float(result["ranking"]["ndcg_at_10"]),
            -float(result["alpha"]),
        ),
    )
    return float(selected["alpha"]), {
        "selection_split": "validation",
        "minimum_recall_ratio": minimum_recall_ratio,
        "minimum_recall_at_10": threshold,
        "objective": [
            "maximize tail-target Recall@10",
            "then tail catalog coverage",
            "then overall NDCG@10",
        ],
        "selected_alpha": float(selected["alpha"]),
    }


def history_bucket(length: int) -> str:
    if length <= 5:
        return "01_1-5"
    if length <= 10:
        return "02_6-10"
    if length <= 20:
        return "03_11-20"
    return "04_21-50"


def frequency_bucket(frequency: int) -> str:
    if frequency == 0:
        return "01_zero"
    if frequency <= 5:
        return "02_1-5"
    if frequency <= 20:
        return "03_6-20"
    if frequency <= 100:
        return "04_21-100"
    return "05_101+"


def segmented_metrics(
    predictions: list[list[str]],
    targets: list[str],
    labels: list[str],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    output: dict[str, dict[str, float | int]] = {}
    for label, indices in sorted(grouped.items()):
        output[label] = {
            "examples": len(indices),
            **ranking_metrics(
                [predictions[index] for index in indices],
                [targets[index] for index in indices],
                ks=(5, 10),
            ),
        }
    return output


def load_titles(path: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            text = str(payload.get("text") or payload["item_id"])
            titles[str(payload["item_id"])] = text.split(" [SEP] ", 1)[0][:160]
    return titles


def item_record(
    item_id: str,
    train_frequencies: Counter[str],
    tail_items: frozenset[str],
    titles: dict[str, str],
) -> dict[str, object]:
    return {
        "item_id": item_id,
        "title": titles.get(item_id, item_id),
        "train_frequency": train_frequencies[item_id],
        "popularity_group": "tail" if item_id in tail_items else "head",
    }


def error_analysis(
    candidates: list[list[str]],
    baseline: list[list[str]],
    calibrated: list[list[str]],
    targets: list[str],
    examples: list[SequenceExample],
    *,
    train_frequencies: Counter[str],
    tail_items: frozenset[str],
    titles: dict[str, str],
    sample_limit: int,
) -> dict[str, object]:
    transition_counts: Counter[str] = Counter()
    false_positive_counts: Counter[str] = Counter()
    samples: dict[str, list[dict[str, object]]] = defaultdict(list)
    for index, (pool, before, after, target, example) in enumerate(
        zip(candidates, baseline, calibrated, targets, examples, strict=True)
    ):
        before_hit = target in before
        after_hit = target in after
        pool_hit = target in pool
        transition = (
            "kept_hit"
            if before_hit and after_hit
            else "recovered"
            if not before_hit and after_hit
            else "lost"
            if before_hit and not after_hit
            else "still_missed"
        )
        transition_counts[transition] += 1
        for item_id in after:
            if item_id != target:
                false_positive_counts[item_id] += 1
        target_group = "tail" if target in tail_items else "head"
        sample_group = f"{transition}_{target_group}"
        if not pool_hit:
            sample_group = f"candidate_pool_miss_{target_group}"
        if len(samples[sample_group]) < sample_limit:
            samples[sample_group].append(
                {
                    "row_index": index,
                    "user_hash": hashlib.sha256(example.user_id.encode()).hexdigest()[:12],
                    "history_length": len(example.history),
                    "target": item_record(
                        target, train_frequencies, tail_items, titles
                    ),
                    "target_in_candidate_pool": pool_hit,
                    "baseline_top10": [
                        item_record(item, train_frequencies, tail_items, titles)
                        for item in before
                    ],
                    "calibrated_top10": [
                        item_record(item, train_frequencies, tail_items, titles)
                        for item in after
                    ],
                }
            )

    return {
        "candidate_pool": {
            **ranking_metrics(candidates, targets, ks=(10, len(candidates[0]))),
            "size": len(candidates[0]),
        },
        "transitions": dict(sorted(transition_counts.items())),
        "by_history_length": {
            "baseline": segmented_metrics(
                baseline, targets, [history_bucket(len(example.history)) for example in examples]
            ),
            "calibrated": segmented_metrics(
                calibrated,
                targets,
                [history_bucket(len(example.history)) for example in examples],
            ),
        },
        "by_target_train_frequency": {
            "baseline": segmented_metrics(
                baseline,
                targets,
                [frequency_bucket(train_frequencies[target]) for target in targets],
            ),
            "calibrated": segmented_metrics(
                calibrated,
                targets,
                [frequency_bucket(train_frequencies[target]) for target in targets],
            ),
        },
        "most_common_false_positives": [
            {
                **item_record(item_id, train_frequencies, tail_items, titles),
                "false_positive_slots": count,
            }
            for item_id, count in false_positive_counts.most_common(25)
        ],
        "sample_policy": "first rows in deterministic test order; user IDs are SHA-256 truncated",
        "samples": dict(sorted(samples.items())),
    }


def main() -> None:
    args = parse_args()
    if args.candidate_count < args.top_k or args.candidate_beam_size < args.candidate_count:
        raise ValueError("candidate-beam-size >= candidate-count >= top-k is required")
    if not 0.0 < args.minimum_recall_ratio <= 1.0:
        raise ValueError("minimum-recall-ratio must be in (0, 1]")
    if not 0.0 < args.head_fraction < 1.0:
        raise ValueError("head-fraction must be in (0, 1)")
    alphas = sorted(set(args.alpha or DEFAULT_ALPHAS))
    if 0.0 not in alphas or any(alpha < 0 or not math.isfinite(alpha) for alpha in alphas):
        raise ValueError("alpha grid must include 0 and contain finite non-negative values")

    config = load_config(args.config)
    data_config = config["data"]
    generator_config = config["generator"]
    audit_path = Path(data_config["data_audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("synthetic_data_used") is not False:
        raise ValueError("long-tail analysis requires a passed real-data audit")

    catalog_path = args.artifact_dir / "catalog.json"
    catalog = SemanticIdCatalog.load(catalog_path)
    device = torch.device(config["device"])
    model, model_path = load_model(config, catalog, args.artifact_dir, device)

    train_frequencies: Counter[str] = Counter(
        example.target_item_id
        for example in iter_sequence_examples_jsonl(data_config["train_path"])
    )
    head_items, tail_items = popularity_partition(
        catalog.item_ids,
        train_frequencies,
        head_fraction=args.head_fraction,
    )

    validation = generate_candidates(
        model,
        data_config["valid_path"],
        catalog,
        device=device,
        batch_size=generator_config["batch_size"],
        beam_size=args.candidate_beam_size,
        candidate_count=args.candidate_count,
        limit=args.max_validation_examples or None,
        log_every_batches=args.log_every_batches,
        split_name="validation",
    )
    validation_candidates, validation_scores, validation_targets, _, validation_seconds = validation
    validation_results: list[dict[str, object]] = []
    for alpha in alphas:
        _, metrics = evaluate_alpha(
            validation_candidates,
            validation_scores,
            validation_targets,
            alpha=alpha,
            top_k=args.top_k,
            catalog=catalog,
            head_items=head_items,
            tail_items=tail_items,
            train_frequencies=train_frequencies,
        )
        validation_results.append(metrics)
    selected_alpha, selection = choose_alpha(
        validation_results,
        args.minimum_recall_ratio,
    )
    print(json.dumps({"selection": selection}, ensure_ascii=False, indent=2), flush=True)

    test = generate_candidates(
        model,
        data_config["test_path"],
        catalog,
        device=device,
        batch_size=generator_config["batch_size"],
        beam_size=args.candidate_beam_size,
        candidate_count=args.candidate_count,
        limit=args.max_test_examples or None,
        log_every_batches=args.log_every_batches,
        split_name="test",
    )
    test_candidates, test_scores, test_targets, test_examples, test_seconds = test
    baseline_predictions, baseline_metrics = evaluate_alpha(
        test_candidates,
        test_scores,
        test_targets,
        alpha=0.0,
        top_k=args.top_k,
        catalog=catalog,
        head_items=head_items,
        tail_items=tail_items,
        train_frequencies=train_frequencies,
    )
    calibrated_predictions, calibrated_metrics = evaluate_alpha(
        test_candidates,
        test_scores,
        test_targets,
        alpha=selected_alpha,
        top_k=args.top_k,
        catalog=catalog,
        head_items=head_items,
        tail_items=tail_items,
        train_frequencies=train_frequencies,
    )

    items_path = Path(data_config["train_path"]).parent / "items.jsonl"
    titles = load_titles(items_path)
    errors = error_analysis(
        test_candidates,
        baseline_predictions,
        calibrated_predictions,
        test_targets,
        test_examples,
        train_frequencies=train_frequencies,
        tail_items=tail_items,
        titles=titles,
        sample_limit=args.sample_errors,
    )
    report = {
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "category": data_config["category"],
        "synthetic_data_used": False,
        "data_audit_sha256": sha256(audit_path),
        "model_sha256": sha256(model_path),
        "catalog_sha256": sha256(catalog_path),
        "protocol": {
            "candidate_generation": "global-log-softmax Catalog Trie beam search",
            "candidate_beam_size": args.candidate_beam_size,
            "candidate_count": args.candidate_count,
            "top_k": args.top_k,
            "popularity_source": "training target-item frequency only",
            "popularity_penalty": "score - alpha * log1p(train_frequency) / log1p(max_train_frequency)",
            "head_fraction_of_catalog": args.head_fraction,
            "validation_examples": len(validation_targets),
            "test_examples": len(test_targets),
            "test_used_for_selection": False,
        },
        "selection": selection,
        "validation_alpha_sweep": validation_results,
        "test": {
            "baseline": baseline_metrics,
            "selected_calibration": calibrated_metrics,
        },
        "timing": {
            "validation_candidate_generation_seconds": validation_seconds,
            "test_candidate_generation_seconds": test_seconds,
            "device": str(device),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    error_path = args.output_dir / "error_analysis.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    error_path.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {report_path} and {error_path}", flush=True)


if __name__ == "__main__":
    main()
