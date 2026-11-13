from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from regenrec.data import (
    RecommendationDataset,
    SequenceCollator,
    SequenceExample,
    iter_sequence_examples_jsonl,
    load_sequence_examples_jsonl,
)
from regenrec.embeddings import load_embedding_bundle
from regenrec.metrics import (
    popularity_strata,
    ranking_metrics,
    recommendation_coverage_metrics,
    recommendation_strata_coverage_metrics,
)
from regenrec.models.generative import GenerativeRecommender
from regenrec.reranking import (
    CandidateFeatureRow,
    LearnedHybridScorer,
    build_blair_user_profiles,
    build_candidate_features,
    exact_blair_retrieve,
    fit_pairwise_hybrid_scorer,
    fused_soft_quota_rerank,
    learned_hybrid_rerank,
    score_item_rows,
)


QUOTA_PROFILES: dict[str, dict[str, float] | None] = {
    "none": None,
    "mild": {"head": 0.70, "torso": 0.20, "observed_tail": 0.10},
    "balanced": {"head": 0.60, "torso": 0.25, "observed_tail": 0.15},
    "tail_20": {"head": 0.60, "torso": 0.20, "observed_tail": 0.20},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune or evaluate BLaIR retrieval and long-tail candidate reranking"
    )
    parser.add_argument("--config", default="configs/v2_amazon2023_top10.yaml")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--mode", choices=("tune", "evaluate"), default="tune")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--selection-strategy", choices=("prefix", "reservoir"), default="prefix"
    )
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--retrieval-count", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--minimum-recall-ratio", type=float, default=0.97)
    parser.add_argument(
        "--sweep-profile",
        choices=("compact", "expanded", "mild"),
        default="compact",
        help=(
            "Mild is the pre-registered validation-only search over BLaIR weight "
            "0.15-0.25, frequency alpha 0.05-0.10, and quota strength <=0.01"
        ),
    )
    parser.add_argument("--retrieval-weight", type=float, default=0.5)
    parser.add_argument("--frequency-alpha", type=float, default=0.0)
    parser.add_argument(
        "--quota-profile", choices=tuple(QUOTA_PROFILES), default="none"
    )
    parser.add_argument("--quota-strength", type=float, default=0.0)
    parser.add_argument(
        "--hybrid-scorer-mode",
        choices=("linear", "fit", "evaluate"),
        default="linear",
        help="Fit/evaluate a validation-trained pairwise generator+BLaIR scorer",
    )
    parser.add_argument("--hybrid-scorer-path", type=Path)
    parser.add_argument("--hybrid-fit-fraction", type=float, default=0.7)
    parser.add_argument("--hybrid-epochs", type=int, default=300)
    parser.add_argument("--hybrid-tail-positive-weight", type=float, default=1.5)
    parser.add_argument(
        "--evaluation-profile",
        choices=("standard", "frozen_v2_modes"),
        default="standard",
        help=(
            "Append the pre-registered Musical-Instruments mild and aggressive "
            "V2 modes without using the current split for parameter selection"
        ),
    )
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


def load_examples(args: argparse.Namespace, split_path: str) -> list[SequenceExample]:
    return load_sequence_examples_jsonl(
        split_path,
        limit=args.max_examples or None,
        selection_strategy=args.selection_strategy,
        seed=args.selection_seed,
    )


def generate_candidates(
    model: GenerativeRecommender,
    examples: list[SequenceExample],
    catalog: SemanticIdCatalog,
    *,
    device: torch.device,
    batch_size: int,
    beam_size: int,
    candidate_count: int,
    log_every_batches: int,
) -> tuple[list[list[str]], list[list[float]], float]:
    loader = DataLoader(
        RecommendationDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=SequenceCollator(catalog),
    )
    candidates: list[list[str]] = []
    scores: list[list[float]] = []
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
            candidates.extend(batch_items)
            scores.extend(batch_scores)
            if batch_index % log_every_batches == 0:
                print(
                    f"beam batches={batch_index} examples={len(candidates)} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    return candidates, scores, time.perf_counter() - started


def group_ranking_metrics(
    predictions: Sequence[Sequence[str]],
    targets: Sequence[str],
    items: frozenset[str],
) -> dict[str, float | int]:
    selected = [
        (prediction, target)
        for prediction, target in zip(predictions, targets, strict=True)
        if target in items
    ]
    if not selected:
        return {
            "examples": 0,
            "hits_at_5": 0,
            "hits_at_10": 0,
            "recall_at_5": 0.0,
            "ndcg_at_5": 0.0,
            "recall_at_10": 0.0,
            "ndcg_at_10": 0.0,
        }
    group_predictions, group_targets = zip(*selected, strict=True)
    metrics = ranking_metrics(group_predictions, group_targets, ks=(5, 10))
    return {
        "examples": len(selected),
        "hits_at_5": round(metrics["recall_at_5"] * len(selected)),
        "hits_at_10": round(metrics["recall_at_10"] * len(selected)),
        **metrics,
    }


def evaluate_predictions(
    predictions: list[list[str]],
    targets: list[str],
    *,
    catalog: SemanticIdCatalog,
    strata: dict[str, frozenset[str]],
    item_frequencies: Counter[str],
) -> dict[str, object]:
    ranking = ranking_metrics(predictions, targets, ks=(5, 10))
    legacy_tail = frozenset(
        set(strata["torso"]) | set(strata["observed_tail"]) | set(strata["cold_start"])
    )
    return {
        "ranking": ranking,
        "target_groups": {
            name: group_ranking_metrics(predictions, targets, items)
            for name, items in strata.items()
        },
        "coverage": recommendation_coverage_metrics(
            predictions,
            catalog_items=catalog.item_ids,
            tail_items=legacy_tail,
            item_frequencies=item_frequencies,
        ),
        "strata_coverage": recommendation_strata_coverage_metrics(
            predictions,
            catalog_items=catalog.item_ids,
            strata=strata,
        ),
    }


def candidate_oracle(
    candidate_rows: Sequence[Sequence[str]],
    targets: Sequence[str],
    strata: dict[str, frozenset[str]],
) -> dict[str, object]:
    def metrics(items: frozenset[str] | None) -> dict[str, float | int]:
        selected = [
            (row, target)
            for row, target in zip(candidate_rows, targets, strict=True)
            if items is None or target in items
        ]
        hits = sum(target in row for row, target in selected)
        return {
            "examples": len(selected),
            "hits": hits,
            "recall": hits / len(selected) if selected else 0.0,
        }

    return {
        "overall": metrics(None),
        "target_groups": {name: metrics(items) for name, items in strata.items()},
    }


def variant_key(result: dict[str, object]) -> tuple[float, float, float, float, float]:
    tail = result["metrics"]["target_groups"]["observed_tail"]
    overall = result["metrics"]["ranking"]
    coverage = result["metrics"]["strata_coverage"]["strata"]["observed_tail"]
    return (
        float(tail["recall_at_10"]),
        float(tail["ndcg_at_10"]),
        float(overall["recall_at_10"]),
        float(overall["ndcg_at_10"]),
        float(coverage["catalog_coverage"]),
    )


def evaluate_variant(
    features: list[CandidateFeatureRow],
    targets: list[str],
    *,
    catalog: SemanticIdCatalog,
    strata: dict[str, frozenset[str]],
    item_frequencies: Counter[str],
    retrieval_weight: float,
    frequency_alpha: float,
    quota_profile: str,
    quota_strength: float,
    top_k: int,
) -> tuple[list[list[str]], dict[str, object]]:
    started = time.perf_counter()
    predictions = fused_soft_quota_rerank(
        features,
        retrieval_weight=retrieval_weight,
        frequency_alpha=frequency_alpha,
        quota_shares=QUOTA_PROFILES[quota_profile],
        quota_strength=quota_strength,
        top_k=top_k,
    )
    elapsed = time.perf_counter() - started
    return predictions, {
        "parameters": {
            "retrieval_weight": retrieval_weight,
            "frequency_alpha": frequency_alpha,
            "quota_profile": quota_profile,
            "quota_strength": quota_strength,
        },
        "metrics": evaluate_predictions(
            predictions,
            targets,
            catalog=catalog,
            strata=strata,
            item_frequencies=item_frequencies,
        ),
        "rerank_efficiency": {
            "wall_seconds": elapsed,
            "milliseconds_per_example": 1000.0 * elapsed / len(targets),
        },
    }


def select_feasible(
    results: list[dict[str, object]],
    *,
    minimum_recall: float,
) -> dict[str, object]:
    feasible = [
        result
        for result in results
        if float(result["metrics"]["ranking"]["recall_at_10"]) >= minimum_recall
    ]
    if not feasible:
        raise RuntimeError("no reranking variant satisfies the main Recall@10 gate")
    return max(feasible, key=variant_key)


def rank_of(items: Sequence[str], target: str) -> int | None:
    try:
        return list(items).index(target) + 1
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    if args.max_examples < 0:
        raise ValueError("max-examples must be non-negative")
    if args.candidate_count < args.top_k or args.beam_size < args.candidate_count:
        raise ValueError("beam-size >= candidate-count >= top-k is required")
    if args.retrieval_count <= 0 or args.batch_size <= 0 or args.retrieval_batch_size <= 0:
        raise ValueError("retrieval-count and batch sizes must be positive")
    if not 0 < args.minimum_recall_ratio <= 1:
        raise ValueError("minimum-recall-ratio must be in (0, 1]")
    if args.mode == "tune" and args.split != "validation":
        raise ValueError("hyperparameter tuning is restricted to the validation split")
    if not 0 < args.hybrid_fit_fraction < 1:
        raise ValueError("hybrid-fit-fraction must be in (0, 1)")
    if args.hybrid_scorer_mode == "fit" and args.split != "validation":
        raise ValueError("hybrid scorer fitting is restricted to the validation split")
    if args.hybrid_scorer_mode == "evaluate" and args.hybrid_scorer_path is None:
        raise ValueError("hybrid-scorer-path is required for hybrid evaluation")

    config = load_config(args.config)
    data_config = config["data"]
    audit_path = Path(data_config["data_audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("synthetic_data_used") is not False:
        raise ValueError("reranking analysis requires a passed real-data audit")

    catalog_path = args.artifact_dir / "catalog.json"
    catalog = SemanticIdCatalog.load(catalog_path)
    device = torch.device(config["device"])
    model, model_path = load_model(config, catalog, args.artifact_dir, device)
    embedding_path = Path(data_config["item_embeddings_path"])
    embedding_ids, item_embeddings, embedding_metadata = load_embedding_bundle(
        embedding_path
    )
    if embedding_metadata.get("backend") != "blair":
        raise ValueError("candidate reranking requires real BLaIR embeddings")
    if set(embedding_ids) != set(catalog.item_ids):
        raise ValueError("BLaIR embedding items and catalog items must match")
    item_to_embedding = {item_id: index for index, item_id in enumerate(embedding_ids)}

    train_frequencies: Counter[str] = Counter()
    for example in iter_sequence_examples_jsonl(data_config["train_path"]):
        train_frequencies[example.target_item_id] += 1
    strata = popularity_strata(catalog.item_ids, train_frequencies)
    item_groups = {
        item_id: name for name, items in strata.items() for item_id in items
    }
    observed_tail_ids = [
        item_id for item_id in catalog.item_ids if item_id in strata["observed_tail"]
    ]
    split_path = (
        data_config["valid_path"] if args.split == "validation" else data_config["test_path"]
    )
    examples = load_examples(args, split_path)
    targets = [example.target_item_id for example in examples]
    histories = [[event.item_id for event in example.history] for example in examples]

    generator_items, generator_scores, generator_seconds = generate_candidates(
        model,
        examples,
        catalog,
        device=device,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        candidate_count=args.candidate_count,
        log_every_batches=args.log_every_batches,
    )
    started = time.perf_counter()
    profiles = build_blair_user_profiles(examples, item_embeddings, item_to_embedding)
    profile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    retrieval_items, retrieval_scores = exact_blair_retrieve(
        profiles,
        observed_tail_ids,
        item_embeddings,
        item_to_embedding,
        histories,
        top_k=args.retrieval_count,
        batch_size=args.retrieval_batch_size,
    )
    retrieval_seconds = time.perf_counter() - started
    started = time.perf_counter()
    generator_semantic_scores = score_item_rows(
        profiles,
        generator_items,
        item_embeddings,
        item_to_embedding,
        batch_size=args.retrieval_batch_size,
    )
    features = build_candidate_features(
        generator_items,
        generator_scores,
        generator_semantic_scores,
        retrieval_items,
        retrieval_scores,
        train_frequencies,
        item_groups,
    )
    hybrid_scorer: LearnedHybridScorer | None = None
    hybrid_metadata: dict[str, object] | None = None
    if args.hybrid_scorer_mode == "fit":
        threshold = round(args.hybrid_fit_fraction * 10_000)
        fit_indices = [
            index
            for index, example in enumerate(examples)
            if int(hashlib.sha256(example.user_id.encode()).hexdigest()[:16], 16) % 10_000
            < threshold
        ]
        fit_index_set = set(fit_indices)
        selection_indices = [
            index for index in range(len(examples)) if index not in fit_index_set
        ]
        hybrid_scorer, fit_diagnostics = fit_pairwise_hybrid_scorer(
            [features[index] for index in fit_indices],
            [targets[index] for index in fit_indices],
            seed=args.seed,
            epochs=args.hybrid_epochs,
            tail_positive_weight=args.hybrid_tail_positive_weight,
        )
        scorer_path = args.hybrid_scorer_path or args.output_dir / "hybrid_scorer.json"
        scorer_path.parent.mkdir(parents=True, exist_ok=True)
        scorer_path.write_text(
            json.dumps(hybrid_scorer.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        hybrid_metadata = {
            "mode": "fit",
            "scorer_path": str(scorer_path.resolve()),
            "fit_fraction": args.hybrid_fit_fraction,
            "split_assignment": "sha256(user_id) modulo 10000",
            "fit_examples": len(fit_indices),
            "selection_examples": len(selection_indices),
            "fit_diagnostics": fit_diagnostics,
            "selection_labels_used_for_fitting": False,
            "test_labels_used_for_fitting": False,
        }
        features = [features[index] for index in selection_indices]
        targets = [targets[index] for index in selection_indices]
        examples = [examples[index] for index in selection_indices]
        generator_items = [generator_items[index] for index in selection_indices]
        retrieval_items = [retrieval_items[index] for index in selection_indices]
    elif args.hybrid_scorer_mode == "evaluate":
        assert args.hybrid_scorer_path is not None
        hybrid_scorer = LearnedHybridScorer.from_dict(
            json.loads(args.hybrid_scorer_path.read_text(encoding="utf-8"))
        )
        hybrid_metadata = {
            "mode": "evaluate",
            "scorer_path": str(args.hybrid_scorer_path.resolve()),
            "fit_fraction": None,
            "fit_examples": None,
            "selection_examples": len(examples),
            "selection_labels_used_for_fitting": False,
            "test_labels_used_for_fitting": False,
        }
    feature_seconds = time.perf_counter() - started
    union_items = [list(row.item_ids) for row in features]
    source_oracle = {
        "generator": candidate_oracle(generator_items, targets, strata),
        "blair_observed_tail_retrieval": candidate_oracle(
            retrieval_items, targets, strata
        ),
        "union": candidate_oracle(union_items, targets, strata),
    }

    baseline_predictions, baseline = evaluate_variant(
        features,
        targets,
        catalog=catalog,
        strata=strata,
        item_frequencies=train_frequencies,
        retrieval_weight=0.0,
        frequency_alpha=0.0,
        quota_profile="none",
        quota_strength=0.0,
        top_k=args.top_k,
    )
    stage_a: list[dict[str, object]] = []
    stage_b: list[dict[str, object]] = []
    evaluation_ablations: list[dict[str, object]] = []
    selected_predictions: list[list[str]]
    selection: dict[str, object]
    if hybrid_scorer is not None:
        rerank_started = time.perf_counter()
        selected_predictions = learned_hybrid_rerank(
            features,
            hybrid_scorer,
            top_k=args.top_k,
        )
        rerank_elapsed = time.perf_counter() - rerank_started
        selected = {
            "parameters": hybrid_scorer.to_dict(),
            "metrics": evaluate_predictions(
                selected_predictions,
                targets,
                catalog=catalog,
                strata=strata,
                item_frequencies=train_frequencies,
            ),
            "rerank_efficiency": {
                "wall_seconds": rerank_elapsed,
                "milliseconds_per_example": 1000.0 * rerank_elapsed / len(targets),
            },
        }
        selection = {
            "selection_split": args.split,
            "test_used_for_selection": False,
            "objective": "pairwise validation fit; evaluated on disjoint user-hash fold",
            "selected_parameters": hybrid_scorer.to_dict(),
        }
    elif args.mode == "tune":
        if args.sweep_profile == "mild":
            retrieval_weights = (0.15, 0.20, 0.25)
            frequency_alphas = (0.05, 0.075, 0.10)
        elif args.sweep_profile == "expanded":
            retrieval_weights = (0.0, 0.025, 0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.0)
            frequency_alphas = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
        else:
            retrieval_weights = (0.0, 0.25, 0.50, 0.75, 1.0)
            frequency_alphas = (0.0, 0.05, 0.10, 0.20)
        for retrieval_weight in retrieval_weights:
            for frequency_alpha in frequency_alphas:
                _, result = evaluate_variant(
                    features,
                    targets,
                    catalog=catalog,
                    strata=strata,
                    item_frequencies=train_frequencies,
                    retrieval_weight=retrieval_weight,
                    frequency_alpha=frequency_alpha,
                    quota_profile="none",
                    quota_strength=0.0,
                    top_k=args.top_k,
                )
                stage_a.append(result)
        baseline_recall = float(baseline["metrics"]["ranking"]["recall_at_10"])
        minimum_recall = baseline_recall * args.minimum_recall_ratio
        selected_stage_a = select_feasible(stage_a, minimum_recall=minimum_recall)
        selected_parameters = selected_stage_a["parameters"]
        if args.sweep_profile == "mild":
            parameter_pairs = {
                (retrieval_weight, frequency_alpha)
                for retrieval_weight in retrieval_weights
                for frequency_alpha in frequency_alphas
            }
            quota_profiles = ("mild", "balanced", "tail_20")
            quota_strengths = (0.005, 0.01)
        elif args.sweep_profile == "expanded":
            feasible_stage_a = sorted(
                (
                    result
                    for result in stage_a
                    if float(result["metrics"]["ranking"]["recall_at_10"])
                    >= minimum_recall
                ),
                key=variant_key,
                reverse=True,
            )
            parameter_pairs = {
                (
                    float(result["parameters"]["retrieval_weight"]),
                    float(result["parameters"]["frequency_alpha"]),
                )
                for result in feasible_stage_a[:6]
            }
            parameter_pairs.update(
                {
                    (retrieval_weight, frequency_alpha)
                    for retrieval_weight in (0.0, 0.025, 0.05, 0.10, 0.15)
                    for frequency_alpha in (0.0, 0.02, 0.05)
                }
            )
            quota_strengths = (0.05, 0.10, 0.20, 0.40, 0.80)
            quota_profiles = ("mild", "balanced", "tail_20")
        else:
            parameter_pairs = {
                (
                    float(selected_parameters["retrieval_weight"]),
                    float(selected_parameters["frequency_alpha"]),
                )
            }
            quota_strengths = (0.02, 0.05, 0.10, 0.20)
            quota_profiles = ("mild", "balanced", "tail_20")
        for retrieval_weight, frequency_alpha in sorted(parameter_pairs):
            for quota_profile in quota_profiles:
                for quota_strength in quota_strengths:
                    _, result = evaluate_variant(
                        features,
                        targets,
                        catalog=catalog,
                        strata=strata,
                        item_frequencies=train_frequencies,
                        retrieval_weight=retrieval_weight,
                        frequency_alpha=frequency_alpha,
                        quota_profile=quota_profile,
                        quota_strength=quota_strength,
                        top_k=args.top_k,
                    )
                    stage_b.append(result)
        selected = select_feasible(stage_a + stage_b, minimum_recall=minimum_recall)
        parameters = selected["parameters"]
        selected_predictions, selected = evaluate_variant(
            features,
            targets,
            catalog=catalog,
            strata=strata,
            item_frequencies=train_frequencies,
            retrieval_weight=float(parameters["retrieval_weight"]),
            frequency_alpha=float(parameters["frequency_alpha"]),
            quota_profile=str(parameters["quota_profile"]),
            quota_strength=float(parameters["quota_strength"]),
            top_k=args.top_k,
        )
        selection = {
            "selection_split": args.split,
            "test_used_for_selection": args.split == "test",
            "minimum_recall_ratio": args.minimum_recall_ratio,
            "baseline_recall_at_10": baseline_recall,
            "minimum_recall_at_10": minimum_recall,
            "objective": [
                "maximize Observed-Tail Recall@10",
                "then Observed-Tail NDCG@10",
                "then overall Recall@10 and NDCG@10",
                "then Observed-Tail catalog coverage",
            ],
            "selected_parameters": parameters,
        }
    else:
        selected_predictions, selected = evaluate_variant(
            features,
            targets,
            catalog=catalog,
            strata=strata,
            item_frequencies=train_frequencies,
            retrieval_weight=args.retrieval_weight,
            frequency_alpha=args.frequency_alpha,
            quota_profile=args.quota_profile,
            quota_strength=args.quota_strength,
            top_k=args.top_k,
        )
        ablation_parameters = [
            ("generator_only", 0.0, 0.0, "none", 0.0),
            ("blair_fusion_only", args.retrieval_weight, 0.0, "none", 0.0),
            (
                "blair_plus_frequency",
                args.retrieval_weight,
                args.frequency_alpha,
                "none",
                0.0,
            ),
            (
                "full_with_soft_quota",
                args.retrieval_weight,
                args.frequency_alpha,
                args.quota_profile,
                args.quota_strength,
            ),
        ]
        if args.evaluation_profile == "frozen_v2_modes":
            ablation_parameters.extend(
                [
                    ("frozen_mild", 0.25, 0.10, "tail_20", 0.01),
                    ("frozen_aggressive", 0.25, 0.20, "balanced", 0.05),
                ]
            )
        for label, retrieval_weight, frequency_alpha, quota_profile, quota_strength in (
            ablation_parameters
        ):
            _, ablation = evaluate_variant(
                features,
                targets,
                catalog=catalog,
                strata=strata,
                item_frequencies=train_frequencies,
                retrieval_weight=retrieval_weight,
                frequency_alpha=frequency_alpha,
                quota_profile=quota_profile,
                quota_strength=quota_strength,
                top_k=args.top_k,
            )
            ablation["label"] = label
            evaluation_ablations.append(ablation)
        selection = {
            "selection_split": None,
            "test_used_for_selection": False,
            "selected_parameters": selected["parameters"],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    indicators_path = args.output_dir / "hit_indicators.jsonl"
    with indicators_path.open("w", encoding="utf-8") as handle:
        for index, (example, target, baseline_row, selected_row, gen, ret, union) in enumerate(
            zip(
                examples,
                targets,
                baseline_predictions,
                selected_predictions,
                generator_items,
                retrieval_items,
                union_items,
                strict=True,
            )
        ):
            handle.write(
                json.dumps(
                    {
                        "example_index": index,
                        "user_hash": hashlib.sha256(example.user_id.encode()).hexdigest()[:16],
                        "target_item_id": target,
                        "target_group": item_groups[target],
                        "baseline_rank": rank_of(baseline_row, target),
                        "selected_rank": rank_of(selected_row, target),
                        "target_in_generator_candidates": target in gen,
                        "target_in_blair_retrieval": target in ret,
                        "target_in_union": target in union,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = {
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "synthetic_data_used": False,
        "seed": args.seed,
        "split": args.split,
        "mode": args.mode,
        "evaluation_profile": args.evaluation_profile,
        "examples": len(examples),
        "sampling": {
            "max_examples": args.max_examples or None,
            "strategy": args.selection_strategy,
            "seed": args.selection_seed,
        },
        "sweep_profile": args.sweep_profile,
        "hybrid_scorer": hybrid_metadata,
        "provenance": {
            "data_audit_sha256": sha256(audit_path),
            "model_sha256": sha256(model_path),
            "catalog_sha256": sha256(catalog_path),
            "embedding_sha256": sha256(embedding_path),
            "embedding_metadata": embedding_metadata,
        },
        "protocol": {
            "generator": {
                "beam_size": args.beam_size,
                "candidate_count": args.candidate_count,
            },
            "retrieval": {
                "source": "exact inner-product search over real BLaIR item embeddings",
                "candidate_stratum": "observed_tail",
                "candidate_count": args.retrieval_count,
                "history_only_profile": True,
                "seen_history_items_excluded": True,
                "negative_review_positive_interest_weight": 0.0,
                "neutral_review_positive_interest_weight": 0.25,
                "positive_review_positive_interest_weight": 1.0,
                "recency_decay": 0.12,
            },
            "frequency_adjustment": (
                "fused_score - alpha * log1p(training_target_frequency) / "
                "log1p(max_training_target_frequency)"
            ),
            "soft_quota": (
                "iterative proportional deficit bonus; no hard group insertion"
            ),
            "test_labels_used_for_candidate_or_score_construction": False,
            "validation_fit_labels_used_for_hybrid_scorer": (
                args.hybrid_scorer_mode == "fit"
            ),
            "frozen_v2_modes": (
                {
                    "source_category": "Musical_Instruments",
                    "selection_split": "validation",
                    "current_category_used_for_selection": False,
                    "mild": {
                        "retrieval_weight": 0.25,
                        "frequency_alpha": 0.10,
                        "quota_profile": "tail_20",
                        "quota_strength": 0.01,
                    },
                    "aggressive": {
                        "retrieval_weight": 0.25,
                        "frequency_alpha": 0.20,
                        "quota_profile": "balanced",
                        "quota_strength": 0.05,
                    },
                }
                if args.evaluation_profile == "frozen_v2_modes"
                else None
            ),
        },
        "popularity_strata": {name: len(items) for name, items in strata.items()},
        "candidate_oracle": source_oracle,
        "baseline": baseline,
        "stage_a_fusion_frequency_sweep": stage_a,
        "stage_b_soft_quota_sweep": stage_b,
        "evaluation_ablations": evaluation_ablations,
        "selection": selection,
        "selected": selected,
        "timing": {
            "generator_seconds": generator_seconds,
            "profile_seconds": profile_seconds,
            "retrieval_seconds": retrieval_seconds,
            "feature_seconds": feature_seconds,
            "fixed_pipeline_milliseconds_per_example": 1000.0
            * (generator_seconds + profile_seconds + retrieval_seconds + feature_seconds)
            / len(examples),
            "selected_total_milliseconds_per_example": 1000.0
            * (
                generator_seconds
                + profile_seconds
                + retrieval_seconds
                + feature_seconds
                + float(selected["rerank_efficiency"]["wall_seconds"])
            )
            / len(examples),
        },
        "hit_indicators_path": str(indicators_path.resolve()),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selection": selection, "selected": selected}, ensure_ascii=False, indent=2))
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
