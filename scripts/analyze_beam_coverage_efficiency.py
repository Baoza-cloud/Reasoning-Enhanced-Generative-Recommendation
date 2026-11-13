from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

import psutil
import torch
from torch.utils.data import DataLoader

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from regenrec.data import (
    RecommendationDataset,
    SequenceCollator,
    iter_sequence_examples_jsonl,
    load_sequence_examples_jsonl,
)
from regenrec.long_tail import semantic_mmr_rerank
from regenrec.metrics import (
    popularity_strata,
    ranking_metrics,
    recommendation_coverage_metrics,
    recommendation_strata_coverage_metrics,
)
from regenrec.models.generative import GenerativeRecommender
from train_amazon2023 import make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure full-catalog coverage, long-tail reach, and beam efficiency"
    )
    parser.add_argument("--config", default="configs/v2_amazon2023_top10.yaml")
    parser.add_argument("--seed", type=int, help="Independent-run seed for report provenance")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(
            "artifacts/amazon2023_musical_instruments_blair_v2_top10_multiseed/seed_2026"
        ),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--preference-reasoning-mode",
        choices=(
            "full",
            "mean_pool",
            "long_term_only",
            "short_term_only",
            "fixed_0p5_gate",
            "no_event_attention",
            "no_fusion_mlp",
        ),
        help=(
            "Frozen-checkpoint reasoning intervention. When omitted, use the mode "
            "from the training config."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="0 evaluates the complete split",
    )
    parser.add_argument("--beam-size", action="append", type=int, required=True)
    parser.add_argument(
        "--diversity-weight",
        action="append",
        type=float,
        help="Semantic-ID MMR penalty in [0, 1); repeat to evaluate a grid (default: 0)",
    )
    parser.add_argument("--head-fraction", type=float, default=0.2)
    parser.add_argument("--torso-fraction", type=float, default=0.3)
    parser.add_argument(
        "--target-group",
        choices=("head", "torso", "observed_tail", "cold_start"),
        help="Evaluate only examples whose target belongs to this popularity stratum",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Evaluation batch size override (use a smaller value for wide beams)",
    )
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument(
        "--hit-indicators-dir",
        type=Path,
        help=(
            "Optional directory for per-example candidate and Top-K ranks; "
            "these records support paired McNemar/bootstrap tests"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def generate_batch(
    model: GenerativeRecommender,
    batch: dict[str, torch.Tensor],
    *,
    catalog: SemanticIdCatalog,
    beam_size: int,
    candidate_count: int,
) -> tuple[list[list[str]], list[list[float]]]:
    predicted, scores = model.generate_beam(
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
    return predicted, scores


def target_ids(batch: dict[str, torch.Tensor], catalog: SemanticIdCatalog) -> list[str]:
    resolved = [catalog.resolve(row.tolist()) for row in batch["target_labels"][:, :-1]]
    if any(item_id is None for item_id in resolved):
        raise RuntimeError("evaluation target could not be resolved in the semantic catalog")
    return [str(item_id) for item_id in resolved]


def group_metrics(
    predictions: list[list[str]],
    targets: list[str],
    group_items: frozenset[str],
    *,
    ks: tuple[int, ...],
) -> dict[str, float | int]:
    selected = [
        (prediction, target)
        for prediction, target in zip(predictions, targets, strict=True)
        if target in group_items
    ]
    if not selected:
        return {
            "examples": 0,
            **{
                name: 0.0
                for k in ks
                for name in (f"recall_at_{k}", f"ndcg_at_{k}")
            },
        }
    group_predictions, group_targets = zip(*selected, strict=True)
    return {
        "examples": len(selected),
        **ranking_metrics(group_predictions, group_targets, ks=ks),
    }


def candidate_oracle_metrics(
    candidates: list[list[str]],
    targets: list[str],
    group_items: frozenset[str] | None = None,
) -> dict[str, float | int | None]:
    """Measure whether the relevant item exists anywhere in the generated candidates."""
    selected = [
        (ranked_items, target)
        for ranked_items, target in zip(candidates, targets, strict=True)
        if group_items is None or target in group_items
    ]
    ranks: list[int] = []
    returned_counts: list[int] = []
    for ranked_items, target in selected:
        returned_counts.append(len(ranked_items))
        try:
            ranks.append(ranked_items.index(target) + 1)
        except ValueError:
            pass
    examples = len(selected)
    hits = len(ranks)
    return {
        "examples": examples,
        "hits": hits,
        "candidate_recall": hits / examples if examples else 0.0,
        "mean_target_rank_when_hit": sum(ranks) / hits if hits else None,
        "mean_returned_candidates": (
            sum(returned_counts) / examples if examples else 0.0
        ),
        "min_returned_candidates": min(returned_counts) if returned_counts else 0,
        "max_returned_candidates": max(returned_counts) if returned_counts else 0,
    }


def item_rank(ranked_items: list[str], target: str) -> int | None:
    try:
        return ranked_items.index(target) + 1
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    if args.max_examples < 0:
        raise ValueError("max-examples must be non-negative")
    if not 0.0 < args.head_fraction < 1.0:
        raise ValueError("head-fraction must be in (0, 1)")
    if not 0.0 <= args.torso_fraction < 1.0:
        raise ValueError("torso-fraction must be in [0, 1)")
    if args.head_fraction + args.torso_fraction >= 1.0:
        raise ValueError("head-fraction + torso-fraction must be below 1")
    if args.warmup_batches < 0:
        raise ValueError("warmup-batches must be non-negative")
    if args.log_every_batches <= 0:
        raise ValueError("log-every-batches must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    beam_sizes = list(dict.fromkeys(args.beam_size))
    if any(beam_size <= 0 for beam_size in beam_sizes):
        raise ValueError("beam sizes must be positive")
    diversity_weights = list(dict.fromkeys(args.diversity_weight or [0.0]))
    if any(not 0.0 <= weight < 1.0 for weight in diversity_weights):
        raise ValueError("diversity weights must be in [0, 1)")

    config = load_config(args.config)
    data_config = config["data"]
    generator_config = config["generator"]
    audit_path = Path(data_config["data_audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("synthetic_data_used") is not False:
        raise ValueError("coverage analysis requires a passed real-data audit")

    catalog_path = args.artifact_dir / "catalog.json"
    model_path = args.artifact_dir / "generator.pt"
    catalog = SemanticIdCatalog.load(catalog_path)
    device = torch.device(config["device"])
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
        preference_reasoning_mode=(
            args.preference_reasoning_mode
            or generator_config.get("preference_reasoning_mode", "full")
        ),
        history_forgetting_mode=generator_config.get("history_forgetting_mode", "none"),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    train_frequencies: Counter[str] = Counter()
    train_interactions = 0
    for example in iter_sequence_examples_jsonl(data_config["train_path"]):
        train_frequencies[example.target_item_id] += 1
        train_interactions += 1
    strata = popularity_strata(
        catalog.item_ids,
        train_frequencies,
        head_fraction=args.head_fraction,
        torso_fraction=args.torso_fraction,
    )
    observed_tail_items = strata["observed_tail"]
    cold_start_items = strata["cold_start"]
    legacy_tail_items = frozenset(
        set(strata["torso"]) | set(observed_tail_items) | set(cold_start_items)
    )
    stratum_interactions = {
        name: sum(train_frequencies[item_id] for item_id in items)
        for name, items in strata.items()
    }

    split_path = data_config["valid_path"] if args.split == "validation" else data_config["test_path"]
    if args.target_group is None:
        loader, example_count = make_loader(
            split_path,
            catalog,
            batch_size=args.batch_size or generator_config["batch_size"],
            limit=args.max_examples or None,
            shuffle=False,
        )
    else:
        group_items = strata[args.target_group]
        examples = [
            example
            for example in load_sequence_examples_jsonl(split_path)
            if example.target_item_id in group_items
        ]
        if args.max_examples:
            examples = examples[: args.max_examples]
        if not examples:
            raise ValueError(
                f"split {args.split} has no examples in target group {args.target_group}"
            )
        example_count = len(examples)
        loader = DataLoader(
            RecommendationDataset(examples),
            batch_size=args.batch_size or generator_config["batch_size"],
            shuffle=False,
            collate_fn=SequenceCollator(catalog),
        )
    process = psutil.Process(os.getpid())
    results: list[dict[str, object]] = []
    for beam_size in beam_sizes:
        candidate_count = beam_size
        return_count = min(beam_size, 10)
        ks = (1,) if return_count == 1 else tuple(k for k in (5, 10) if k <= return_count)
        if args.warmup_batches:
            with torch.no_grad():
                for batch_index, raw_batch in enumerate(loader, start=1):
                    batch = move_batch(raw_batch, device)
                    generate_batch(
                        model,
                        batch,
                        catalog=catalog,
                        beam_size=beam_size,
                        candidate_count=candidate_count,
                    )
                    if batch_index >= args.warmup_batches:
                        break
            synchronize(device)

        gc.collect()
        initial_rss = process.memory_info().rss
        peak_rss = initial_rss
        candidate_items: list[list[str]] = []
        candidate_scores: list[list[float]] = []
        targets: list[str] = []
        started = time.perf_counter()
        with torch.no_grad():
            for batch_index, raw_batch in enumerate(loader, start=1):
                batch = move_batch(raw_batch, device)
                batch_items, batch_scores = generate_batch(
                    model,
                    batch,
                    catalog=catalog,
                    beam_size=beam_size,
                    candidate_count=candidate_count,
                )
                candidate_items.extend(batch_items)
                candidate_scores.extend(batch_scores)
                targets.extend(target_ids(batch, catalog))
                peak_rss = max(peak_rss, process.memory_info().rss)
                if batch_index % args.log_every_batches == 0:
                    print(
                        f"beam={beam_size} batches={batch_index} "
                        f"examples={len(targets)} elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
        synchronize(device)
        decode_elapsed = time.perf_counter() - started
        if len(targets) != example_count:
            raise RuntimeError(
                f"evaluated {len(targets)} examples but expected {example_count}"
            )
        total_candidates = sum(len(items) for items in candidate_items)
        for diversity_weight in diversity_weights:
            rerank_started = time.perf_counter()
            predictions = semantic_mmr_rerank(
                candidate_items,
                candidate_scores,
                catalog,
                diversity_weight=diversity_weight,
                top_k=return_count,
            )
            rerank_elapsed = time.perf_counter() - rerank_started
            total_elapsed = decode_elapsed + rerank_elapsed
            result: dict[str, object] = {
                "beam_size": beam_size,
                "candidate_count": candidate_count,
                "return_count": return_count,
                "diversity_weight": diversity_weight,
                "ranking": ranking_metrics(predictions, targets, ks=ks),
                "target_groups": {
                    name: group_metrics(predictions, targets, items, ks=ks)
                    for name, items in strata.items()
                },
                "candidate_oracle": {
                    "overall": candidate_oracle_metrics(candidate_items, targets),
                    "target_groups": {
                        name: candidate_oracle_metrics(
                            candidate_items,
                            targets,
                            items,
                        )
                        for name, items in strata.items()
                    },
                },
                "coverage": recommendation_coverage_metrics(
                    predictions,
                    catalog_items=catalog.item_ids,
                    tail_items=legacy_tail_items,
                    item_frequencies=train_frequencies,
                ),
                "strata_coverage": recommendation_strata_coverage_metrics(
                    predictions,
                    catalog_items=catalog.item_ids,
                    strata=strata,
                ),
                "efficiency": {
                    "decode_seconds": decode_elapsed,
                    "rerank_seconds": rerank_elapsed,
                    "wall_seconds": total_elapsed,
                    "examples_per_second": example_count / total_elapsed,
                    "milliseconds_per_example": 1000.0 * total_elapsed / example_count,
                    "decode_milliseconds_per_example": (
                        1000.0 * decode_elapsed / example_count
                    ),
                    "rerank_milliseconds_per_example": (
                        1000.0 * rerank_elapsed / example_count
                    ),
                    "candidates_per_decode_second": total_candidates / decode_elapsed,
                    "initial_rss_mb": initial_rss / 1024**2,
                    "peak_rss_mb": peak_rss / 1024**2,
                    "peak_rss_delta_mb": (peak_rss - initial_rss) / 1024**2,
                },
            }
            results.append(result)
            if args.hit_indicators_dir is not None:
                args.hit_indicators_dir.mkdir(parents=True, exist_ok=True)
                group_by_item = {
                    item_id: name for name, items in strata.items() for item_id in items
                }
                weight_label = str(diversity_weight).replace(".", "p")
                indicators_path = args.hit_indicators_dir / (
                    f"beam_{beam_size}_diversity_{weight_label}.jsonl"
                )
                with indicators_path.open("w", encoding="utf-8") as handle:
                    for example_index, (candidates, prediction, target) in enumerate(
                        zip(candidate_items, predictions, targets, strict=True)
                    ):
                        record = {
                            "example_index": example_index,
                            "target_item_id": target,
                            "target_group": group_by_item[target],
                            "candidate_rank": item_rank(candidates, target),
                            "top_k_rank": item_rank(prediction, target),
                            "returned_candidates": len(candidates),
                            "returned_top_k": len(prediction),
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                result["hit_indicators_path"] = str(indicators_path.resolve())
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    report = {
        "seed": args.seed,
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "synthetic_data_used": False,
        "data_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "split": args.split,
        "preference_reasoning_mode": model.preference_reasoning_mode,
        "intervention_scope": (
            "inference-time frozen-checkpoint mechanism intervention"
            if args.preference_reasoning_mode is not None
            else "mode from training configuration"
        ),
        "target_group_filter": args.target_group,
        "examples": example_count,
        "batch_size": args.batch_size or generator_config["batch_size"],
        "timing_scope": (
            "data collation, host-to-device transfer, trie-constrained decoding, "
            "and Semantic-ID MMR reranking"
        ),
        "normalization": "global vocabulary log-softmax before trie masking",
        "popularity_definition": {
            "source": "training target-item frequency",
            "fractions_apply_to": "items with positive training-target frequency",
            "head_fraction_of_observed_items": args.head_fraction,
            "torso_fraction_of_observed_items": args.torso_fraction,
            "observed_tail_fraction_of_observed_items": (
                1.0 - args.head_fraction - args.torso_fraction
            ),
            "strata": {
                name: {
                    "items": len(items),
                    "training_targets": stratum_interactions[name],
                    "training_target_share": (
                        stratum_interactions[name] / train_interactions
                    ),
                }
                for name, items in strata.items()
            },
            "training_targets": train_interactions,
            "zero_frequency_catalog_items": sum(
                train_frequencies[item_id] == 0 for item_id in catalog.item_ids
            ),
        },
        "model": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "logical_cpu_count": psutil.cpu_count(logical=True),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
