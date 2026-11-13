from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from regenrec.data import RecommendationDataset, SequenceCollator, SequenceExample
from regenrec.embeddings import load_embedding_bundle
from regenrec.metrics import popularity_strata
from regenrec.reranking import (
    build_blair_user_profiles,
    exact_blair_retrieve,
    score_item_rows,
)
from scripts.analyze_candidate_reranking import (
    candidate_oracle,
    evaluate_predictions,
    generate_candidates,
    load_examples,
    load_model,
)


ROOT = Path(__file__).resolve().parents[1]
GROUP_TO_ID = {"head": 0, "torso": 1, "observed_tail": 2, "cold_start": 3}
RESIDUAL_WEIGHTS = (0.01, 0.025, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/Validation-only all-candidate teacher-forced scoring diagnostic"
    )
    parser.add_argument("--config", default="configs/v2_final_default.yaml")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
        help="Test is deliberately unavailable in this V2.5 development tool.",
    )
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based prefix offset used by resumable Train census shards.",
    )
    parser.add_argument(
        "--selection-strategy", choices=("prefix", "reservoir"), default="reservoir"
    )
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--retrieval-count", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--teacher-history-batch-size", type=int, default=32)
    parser.add_argument("--teacher-pair-batch-size", type=int, default=1024)
    parser.add_argument(
        "--skip-teacher-scoring",
        action="store_true",
        help="Build a Beam+BLaIR cache without the diagnostic-only union teacher pass.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--log-every-batches", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def percentile_ranks(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: (values[index], -index), reverse=True)
    if len(order) == 1:
        return [1.0]
    output = [0.0] * len(order)
    denominator = len(order) - 1
    for rank, index in enumerate(order):
        output[index] = 1.0 - rank / denominator
    return output


def bootstrap_mean(values: Sequence[float], *, samples: int, seed: int) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"n": 0, "mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    for start in range(0, samples, 100):
        current = min(100, samples - start)
        indices = rng.integers(0, len(array), size=(current, len(array)))
        estimates.append(array[indices].mean(axis=1))
    distribution = np.concatenate(estimates)
    return {
        "n": len(array),
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "bootstrap_samples": samples,
    }


def union_rows(
    generator_items: Sequence[Sequence[str]],
    retrieval_items: Sequence[Sequence[str]],
) -> list[list[str]]:
    output: list[list[str]] = []
    for generated, retrieved in zip(generator_items, retrieval_items, strict=True):
        row: list[str] = []
        for item in (*generated, *retrieved):
            normalized = str(item)
            if normalized not in row:
                row.append(normalized)
        output.append(row)
    return output


def score_teacher_rows(
    model: torch.nn.Module,
    examples: list[SequenceExample],
    item_rows: list[list[str]],
    catalog: SemanticIdCatalog,
    *,
    device: torch.device,
    history_batch_size: int,
    pair_batch_size: int,
    log_every_batches: int,
) -> tuple[list[list[float]], float]:
    loader = DataLoader(
        RecommendationDataset(examples),
        batch_size=history_batch_size,
        shuffle=False,
        collate_fn=SequenceCollator(catalog),
    )
    output: list[list[float]] = []
    started = time.perf_counter()
    example_offset = 0
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = {name: tensor.to(device) for name, tensor in raw_batch.items()}
            rows = item_rows[example_offset : example_offset + len(batch["history_tokens"])]
            maximum = max(len(row) for row in rows)
            candidate_tokens = torch.full(
                (len(rows), maximum, catalog.sequence_length),
                catalog.special_tokens.pad,
                dtype=torch.long,
                device=device,
            )
            candidate_mask = torch.zeros(
                (len(rows), maximum), dtype=torch.bool, device=device
            )
            for row_index, row in enumerate(rows):
                indices = torch.tensor(
                    [catalog.item_to_index[item] for item in row], dtype=torch.long
                )
                candidate_tokens[row_index, : len(row)] = catalog.semantic_tokens[
                    indices
                ].to(device)
                candidate_mask[row_index, : len(row)] = True
            scores = model.score_semantic_candidates(
                history_tokens=batch["history_tokens"],
                behavior_types=batch["behavior_types"],
                time_gap_buckets=batch["time_gap_buckets"],
                history_mask=batch["history_mask"],
                behavior_intensities=batch["behavior_intensities"],
                candidate_tokens=candidate_tokens,
                candidate_mask=candidate_mask,
                pair_batch_size=pair_batch_size,
            )
            for row_index, row in enumerate(rows):
                output.append(scores[row_index, : len(row)].cpu().tolist())
            example_offset += len(rows)
            if batch_index % log_every_batches == 0:
                print(
                    f"teacher batches={batch_index} examples={example_offset} "
                    f"pairs={sum(len(row) for row in item_rows[:example_offset])} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    return output, time.perf_counter() - started


def rank_predictions(
    item_rows: Sequence[Sequence[str]], score_rows: Sequence[Sequence[float]], top_k: int
) -> list[list[str]]:
    output: list[list[str]] = []
    for items, scores in zip(item_rows, score_rows, strict=True):
        ranked = sorted(
            range(len(items)), key=lambda index: (scores[index], -index), reverse=True
        )[:top_k]
        output.append([str(items[index]) for index in ranked])
    return output


def retrieval_only_diagnostics(
    item_rows: list[list[str]],
    teacher_rows: list[list[float]],
    semantic_rows: list[list[float]],
    generator_items: list[list[str]],
    retrieval_items: list[list[str]],
    targets: list[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    teacher_auc: list[float] = []
    semantic_auc: list[float] = []
    teacher_ranks: list[int] = []
    semantic_ranks: list[int] = []
    for items, teacher, semantic, generated, retrieved, target in zip(
        item_rows,
        teacher_rows,
        semantic_rows,
        generator_items,
        retrieval_items,
        targets,
        strict=True,
    ):
        if target not in retrieved or target in generated:
            continue
        position = items.index(target)
        negatives = [index for index in range(len(items)) if index != position]
        target_teacher = teacher[position]
        target_semantic = semantic[position]
        teacher_auc.append(
            sum(
                float(target_teacher > teacher[index])
                + 0.5 * float(target_teacher == teacher[index])
                for index in negatives
            )
            / len(negatives)
        )
        semantic_auc.append(
            sum(
                float(target_semantic > semantic[index])
                + 0.5 * float(target_semantic == semantic[index])
                for index in negatives
            )
            / len(negatives)
        )
        teacher_order = sorted(range(len(items)), key=lambda index: teacher[index], reverse=True)
        semantic_order = sorted(range(len(items)), key=lambda index: semantic[index], reverse=True)
        teacher_ranks.append(teacher_order.index(position) + 1)
        semantic_ranks.append(semantic_order.index(position) + 1)
    return {
        "examples": len(teacher_auc),
        "teacher_pairwise_auc": bootstrap_mean(
            teacher_auc, samples=bootstrap_samples, seed=seed
        ),
        "semantic_pairwise_auc": bootstrap_mean(
            semantic_auc, samples=bootstrap_samples, seed=seed + 1
        ),
        "teacher_rank": {
            "mean": float(np.mean(teacher_ranks)) if teacher_ranks else 0.0,
            "median": float(np.median(teacher_ranks)) if teacher_ranks else 0.0,
            "top_10_rate": float(np.mean(np.asarray(teacher_ranks) <= 10))
            if teacher_ranks
            else 0.0,
            "top_20_rate": float(np.mean(np.asarray(teacher_ranks) <= 20))
            if teacher_ranks
            else 0.0,
        },
        "semantic_rank": {
            "mean": float(np.mean(semantic_ranks)) if semantic_ranks else 0.0,
            "median": float(np.median(semantic_ranks)) if semantic_ranks else 0.0,
            "top_10_rate": float(np.mean(np.asarray(semantic_ranks) <= 10))
            if semantic_ranks
            else 0.0,
            "top_20_rate": float(np.mean(np.asarray(semantic_ranks) <= 20))
            if semantic_ranks
            else 0.0,
        },
    }


def save_cache(
    path: Path,
    *,
    catalog: SemanticIdCatalog,
    examples: list[SequenceExample],
    item_rows: list[list[str]],
    teacher_rows: list[list[float]],
    semantic_rows: list[list[float]],
    generator_items: list[list[str]],
    generator_scores: list[list[float]],
    train_frequencies: Counter[str],
    item_groups: dict[str, str],
    metadata: dict[str, object],
) -> None:
    rows = len(item_rows)
    width = max(len(row) for row in item_rows)
    indices = torch.full((rows, width), -1, dtype=torch.int32)
    mask = torch.zeros((rows, width), dtype=torch.bool)
    teacher = torch.full((rows, width), float("-inf"), dtype=torch.float32)
    semantic = torch.full((rows, width), float("-inf"), dtype=torch.float32)
    generator = torch.full((rows, width), float("-inf"), dtype=torch.float32)
    groups = torch.full((rows, width), 255, dtype=torch.uint8)
    popularity = torch.zeros((rows, width), dtype=torch.float32)
    target_indices = torch.empty(rows, dtype=torch.int32)
    target_positions = torch.full((rows,), -1, dtype=torch.int16)
    history_lengths = torch.empty(rows, dtype=torch.int16)
    maximum_frequency = max(train_frequencies.values(), default=0)
    denominator = math.log1p(maximum_frequency)
    for row_index, (example, items, teacher_scores, semantic_scores, generated, gen_scores) in enumerate(
        zip(
            examples,
            item_rows,
            teacher_rows,
            semantic_rows,
            generator_items,
            generator_scores,
            strict=True,
        )
    ):
        generator_map = dict(zip(generated, gen_scores, strict=True))
        for column, item in enumerate(items):
            indices[row_index, column] = catalog.item_to_index[item]
            mask[row_index, column] = True
            teacher[row_index, column] = teacher_scores[column]
            semantic[row_index, column] = semantic_scores[column]
            generator[row_index, column] = generator_map.get(item, float("-inf"))
            groups[row_index, column] = GROUP_TO_ID[item_groups[item]]
            popularity[row_index, column] = (
                math.log1p(train_frequencies.get(item, 0)) / denominator
                if denominator > 0
                else 0.0
            )
        target_indices[row_index] = catalog.item_to_index[example.target_item_id]
        if example.target_item_id in items:
            target_positions[row_index] = items.index(example.target_item_id)
        history_lengths[row_index] = len(example.history)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": metadata,
            "candidate_catalog_indices": indices,
            "candidate_mask": mask,
            "teacher_scores": teacher,
            "semantic_scores": semantic,
            "beam_scores": generator,
            "groups": groups,
            "popularity": popularity,
            "target_catalog_indices": target_indices,
            "target_positions": target_positions,
            "history_lengths": history_lengths,
        },
        path,
    )


def hard_positive_census(
    examples: Sequence[SequenceExample],
    generator_items: Sequence[Sequence[str]],
    retrieval_items: Sequence[Sequence[str]],
    *,
    catalog: SemanticIdCatalog,
    train_frequencies: Counter[str],
    item_groups: dict[str, str],
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    row_indices: list[int] = []
    user_hashes: list[int] = []
    target_indices: list[int] = []
    group_ids: list[int] = []
    retrieval_ranks: list[int] = []
    history_lengths: list[int] = []
    target_frequencies: list[int] = []
    for row_index, (example, generated, retrieved) in enumerate(
        zip(examples, generator_items, retrieval_items, strict=True)
    ):
        target = example.target_item_id
        if target in generated or target not in retrieved:
            continue
        row_indices.append(row_index)
        digest = hashlib.sha256(example.user_id.encode("utf-8")).digest()
        user_hashes.append(int.from_bytes(digest[:8], "big") & ((1 << 63) - 1))
        target_indices.append(catalog.item_to_index[target])
        group_ids.append(GROUP_TO_ID[item_groups[target]])
        retrieval_ranks.append(list(retrieved).index(target) + 1)
        history_lengths.append(len(example.history))
        target_frequencies.append(train_frequencies.get(target, 0))
    tensors = {
        "row_indices": torch.tensor(row_indices, dtype=torch.int64),
        "user_hashes": torch.tensor(user_hashes, dtype=torch.int64),
        "target_catalog_indices": torch.tensor(target_indices, dtype=torch.int32),
        "groups": torch.tensor(group_ids, dtype=torch.uint8),
        "retrieval_ranks": torch.tensor(retrieval_ranks, dtype=torch.int16),
        "history_lengths": torch.tensor(history_lengths, dtype=torch.int16),
        "target_train_frequencies": torch.tensor(
            target_frequencies, dtype=torch.int32
        ),
    }
    group_counts = Counter(item_groups[examples[index].target_item_id] for index in row_indices)
    summary = {
        "definition": "target in BLaIR Observed-Tail-50 and not in Beam-20",
        "rows": len(row_indices),
        "unique_users_within_shard": len(set(user_hashes)),
        "unique_target_items_within_shard": len(set(target_indices)),
        "target_groups": {
            name: int(group_counts.get(name, 0)) for name in GROUP_TO_ID
        },
    }
    return summary, tensors


def main() -> None:
    args = parse_args()
    if args.max_examples < 0 or args.start_index < 0:
        raise ValueError("max-examples and start-index must be non-negative")
    if args.start_index and args.selection_strategy != "prefix":
        raise ValueError("start-index is only valid with prefix selection")
    if args.beam_size < args.candidate_count or args.candidate_count < args.top_k:
        raise ValueError("beam_size >= candidate_count >= top_k is required")
    if min(
        args.batch_size,
        args.retrieval_batch_size,
        args.teacher_history_batch_size,
        args.teacher_pair_batch_size,
        args.bootstrap_samples,
        args.torch_threads,
    ) <= 0:
        raise ValueError("batch/thread/bootstrap settings must be positive")
    torch.set_num_threads(args.torch_threads)
    config_path = Path(args.config)
    config = load_config(config_path)
    data_config = config["data"]
    audit_path = Path(data_config["data_audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("synthetic_data_used") is not False:
        raise ValueError("V2.5 requires passed real-data audit")
    catalog_path = args.artifact_dir / "catalog.json"
    model_path = args.artifact_dir / "generator.pt"
    catalog = SemanticIdCatalog.load(catalog_path)
    device = torch.device(config["device"])
    model, loaded_model_path = load_model(config, catalog, args.artifact_dir, device)
    if loaded_model_path != model_path:
        raise ValueError("loaded generator path mismatch")
    embedding_path = Path(data_config["item_embeddings_path"])
    embedding_ids, item_embeddings, embedding_metadata = load_embedding_bundle(embedding_path)
    if embedding_metadata.get("backend") != "blair":
        raise ValueError("V2.5 requires real BLaIR embeddings")
    item_to_embedding = {item: index for index, item in enumerate(embedding_ids)}
    if set(item_to_embedding) != set(catalog.item_ids):
        raise ValueError("embedding/catalog item mismatch")
    train_frequencies: Counter[str] = Counter()
    from regenrec.data import iter_sequence_examples_jsonl

    for example in iter_sequence_examples_jsonl(data_config["train_path"]):
        train_frequencies[example.target_item_id] += 1
    strata = popularity_strata(catalog.item_ids, train_frequencies)
    item_groups = {item: name for name, items in strata.items() for item in items}
    observed_tail_ids = [item for item in catalog.item_ids if item in strata["observed_tail"]]

    split_path = {
        "train": data_config["train_path"],
        "validation": data_config["valid_path"],
    }[args.split]
    if args.start_index:
        stop = (
            args.start_index + args.max_examples if args.max_examples else None
        )
        examples = list(
            islice(
                iter_sequence_examples_jsonl(split_path),
                args.start_index,
                stop,
            )
        )
        if not examples:
            raise ValueError(
                f"no examples found at start-index {args.start_index} in {split_path}"
            )
    else:
        class LoadArgs:
            max_examples = args.max_examples
            selection_strategy = args.selection_strategy
            selection_seed = args.seed

        examples = load_examples(LoadArgs, split_path)
    targets = [example.target_item_id for example in examples]
    histories = [[event.item_id for event in example.history] for example in examples]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    write_json(
        state_path,
        {
            "status": "running",
            "phase": "beam_and_blair_candidates",
            "seed": args.seed,
            "split": args.split,
            "start_index": args.start_index,
            "test_evaluated": False,
            "examples": len(examples),
        },
    )
    started_total = time.perf_counter()
    try:
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
        items = union_rows(generator_items, retrieval_items)
        semantic_rows = score_item_rows(
            profiles, items, item_embeddings, item_to_embedding, batch_size=args.retrieval_batch_size
        )
        if args.skip_teacher_scoring:
            teacher_rows = []
            for row, generated, generated_scores in zip(
                items, generator_items, generator_scores, strict=True
            ):
                score_by_item = dict(zip(generated, generated_scores, strict=True))
                teacher_rows.append(
                    [score_by_item.get(item, float("-inf")) for item in row]
                )
            teacher_seconds = 0.0
        else:
            write_json(
                state_path,
                {
                    "status": "running",
                    "phase": "teacher_forced_union_scoring",
                    "seed": args.seed,
                    "split": args.split,
                    "test_evaluated": False,
                    "examples": len(examples),
                },
            )
            teacher_rows, teacher_seconds = score_teacher_rows(
                model,
                examples,
                items,
                catalog,
                device=device,
                history_batch_size=args.teacher_history_batch_size,
                pair_batch_size=args.teacher_pair_batch_size,
                log_every_batches=args.log_every_batches,
            )
        baseline_predictions = [row[: args.top_k] for row in generator_items]
        variants: dict[str, list[list[str]]] = {
            "v2_default_beam": baseline_predictions,
        }
        if not args.skip_teacher_scoring:
            variants["teacher_union"] = rank_predictions(
                items, teacher_rows, args.top_k
            )
            for weight in RESIDUAL_WEIGHTS:
                residual_rows = []
                for teacher, semantic in zip(teacher_rows, semantic_rows, strict=True):
                    teacher_rank = percentile_ranks(teacher)
                    semantic_rank = percentile_ranks(semantic)
                    residual_rows.append(
                        [
                            left + weight * right
                            for left, right in zip(teacher_rank, semantic_rank)
                        ]
                    )
                variants[f"bounded_semantic_{weight:g}"] = rank_predictions(
                    items, residual_rows, args.top_k
                )
        metrics = {
            name: evaluate_predictions(
                predictions,
                targets,
                catalog=catalog,
                strata=strata,
                item_frequencies=train_frequencies,
            )
            for name, predictions in variants.items()
        }
        baseline_r10 = metrics["v2_default_beam"]["ranking"]["recall_at_10"]
        feasible = [
            name
            for name in variants
            if metrics[name]["ranking"]["recall_at_10"] >= 0.99 * baseline_r10
        ]
        selected = max(
            feasible,
            key=lambda name: (
                metrics[name]["target_groups"]["observed_tail"]["recall_at_10"],
                metrics[name]["ranking"]["recall_at_10"],
                metrics[name]["ranking"]["ndcg_at_10"],
            ),
        )
        oracle = {
            "generator": candidate_oracle(generator_items, targets, strata),
            "blair_observed_tail_retrieval": candidate_oracle(
                retrieval_items, targets, strata
            ),
            "union": candidate_oracle(items, targets, strata),
        }
        census_summary, census_tensors = hard_positive_census(
            examples,
            generator_items,
            retrieval_items,
            catalog=catalog,
            train_frequencies=train_frequencies,
            item_groups=item_groups,
        )
        diagnostics = (
            {
                "status": "skipped",
                "reason": "final scorer does not use union teacher scores",
            }
            if args.skip_teacher_scoring
            else retrieval_only_diagnostics(
                items,
                teacher_rows,
                semantic_rows,
                generator_items,
                retrieval_items,
                targets,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
        )
        provenance = {
            "config_sha256": sha256_file(config_path),
            "data_audit_sha256": sha256_file(audit_path),
            "model_sha256": sha256_file(model_path),
            "catalog_sha256": sha256_file(catalog_path),
            "embedding_sha256": sha256_file(embedding_path),
        }
        cache_path = output_dir / "candidate_cache.pt"
        save_cache(
            cache_path,
            catalog=catalog,
            examples=examples,
            item_rows=items,
            teacher_rows=teacher_rows,
            semantic_rows=semantic_rows,
            generator_items=generator_items,
            generator_scores=generator_scores,
            train_frequencies=train_frequencies,
            item_groups=item_groups,
            metadata={
                "seed": args.seed,
                "split": args.split,
                "test_evaluated": False,
                "examples": len(examples),
                "selection_strategy": args.selection_strategy,
                "start_index": args.start_index,
                "teacher_scoring_skipped": args.skip_teacher_scoring,
                "provenance": provenance,
            },
        )
        census_path = output_dir / "hard_positive_rows.pt"
        torch.save(
            {
                "metadata": {
                    "seed": args.seed,
                    "split": args.split,
                    "start_index": args.start_index,
                    "examples": len(examples),
                    "test_evaluated": False,
                    "provenance": provenance,
                },
                **census_tensors,
            },
            census_path,
        )
        report = {
            "status": "completed",
            "dataset": "McAuley-Lab/Amazon-Reviews-2023",
            "category": data_config["category"],
            "split": args.split,
            "test_evaluated": False,
            "synthetic_data_used": False,
            "seed": args.seed,
            "examples": len(examples),
            "selection": {
                "strategy": args.selection_strategy,
                "seed": args.seed,
                "max_examples": args.max_examples or None,
                "start_index": args.start_index,
            },
            "protocol": {
                "beam_size": args.beam_size,
                "candidate_count": args.candidate_count,
                "retrieval_count": args.retrieval_count,
                "top_k": args.top_k,
                "teacher_score": "sum global-vocabulary token log-probability",
                "teacher_scoring_skipped": args.skip_teacher_scoring,
                "residual_weights": list(RESIDUAL_WEIGHTS),
                "minimum_recall_retention": 0.99,
            },
            "candidate_oracle": oracle,
            "hard_positive_census": census_summary,
            "retrieval_only_diagnostics": diagnostics,
            "metrics": metrics,
            "selected_validation_variant": (
                selected if args.split == "validation" else None
            ),
            "selected_split_variant": selected,
            "selected_recall_at_10_retention": (
                metrics[selected]["ranking"]["recall_at_10"] / baseline_r10
            ),
            "timing": {
                "generator_seconds": generator_seconds,
                "profile_seconds": profile_seconds,
                "retrieval_seconds": retrieval_seconds,
                "teacher_seconds": teacher_seconds,
                "total_seconds": time.perf_counter() - started_total,
            },
            "provenance": provenance,
            "candidate_cache": {
                "path": str(cache_path.resolve()),
                "sha256": sha256_file(cache_path),
                "bytes": cache_path.stat().st_size,
            },
            "hard_positive_rows": {
                "path": str(census_path.resolve()),
                "sha256": sha256_file(census_path),
                "bytes": census_path.stat().st_size,
            },
        }
        write_json(output_dir / "report.json", report)
        write_json(
            state_path,
            {
                "status": "completed",
                "phase": None,
                "seed": args.seed,
                "split": args.split,
                "start_index": args.start_index,
                "test_evaluated": False,
                "examples": len(examples),
                "selected_validation_variant": (
                    selected if args.split == "validation" else None
                ),
                "selected_split_variant": selected,
            },
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "split": report["split"],
                    "examples": report["examples"],
                    "selected_split_variant": report["selected_split_variant"],
                    "selected_recall_at_10_retention": report[
                        "selected_recall_at_10_retention"
                    ],
                    "report": str((output_dir / "report.json").resolve()),
                    "candidate_cache": report["candidate_cache"],
                    "hard_positive_rows": report["hard_positive_rows"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    except Exception as error:
        write_json(
            state_path,
            {
                "status": "failed",
                "phase": "v2_5_teacher_rescoring",
                "seed": args.seed,
                "split": args.split,
                "test_evaluated": False,
                "error": repr(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
