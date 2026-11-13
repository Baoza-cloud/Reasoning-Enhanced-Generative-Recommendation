from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from regenrec.data import SequenceExample, iter_sequence_examples_jsonl, load_sequence_examples_jsonl
from regenrec.embeddings import load_embedding_bundle
from regenrec.metrics import popularity_strata
from regenrec.reranking import build_blair_user_profiles, exact_blair_retrieve


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2026, 2027, 2028)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit why real BLaIR tail candidates rarely become final Top-10 hits"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/v2_amazon2023_top10.yaml"))
    parser.add_argument(
        "--mild-root",
        type=Path,
        default=Path("artifacts/candidate_reranking_mild_v1"),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(
            "artifacts/amazon2023_musical_instruments_long_tail_training_v1/"
            "formal/token_loss_010/seed_2026/catalog.json"
        ),
    )
    parser.add_argument("--retrieval-count", type=int, default=50)
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--cases-per-type", type=int, default=6)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/v2_final_report")
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_titles(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            title = str(payload.get("text", payload["item_id"])).split(" [SEP] ", 1)[0]
            output[str(payload["item_id"])] = title[:180]
    return output


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {
        "n": len(materialized),
        "mean": statistics.mean(materialized),
        "median": statistics.median(materialized),
        "p25": percentile(materialized, 0.25),
        "p75": percentile(materialized, 0.75),
    }


def history_features(example: SequenceExample) -> dict[str, float | int]:
    positive = sum(event.behavior == 3 for event in example.history)
    return {
        "history_length": len(example.history),
        "positive_event_share": positive / len(example.history),
        "mean_behavior_intensity": statistics.mean(
            event.behavior_intensity for event in example.history
        ),
        "mean_time_gap_bucket": statistics.mean(
            event.time_gap_bucket for event in example.history
        ),
    }


def source_group(row: dict[str, object]) -> str:
    generator = bool(row["target_in_generator_candidates"])
    blair = bool(row["target_in_blair_retrieval"])
    if generator and blair:
        return "generator_and_blair"
    if generator:
        return "generator_only"
    if blair:
        return "blair_only"
    return "neither"


def error_group(row: dict[str, object]) -> str:
    if row["selected_rank"] is not None and int(row["selected_rank"]) <= 10:
        return "selected_top10_hit"
    if bool(row["target_in_union"]):
        return "candidate_present_top10_miss"
    return "candidate_miss"


def pair_outcome(row: dict[str, object]) -> str:
    baseline = row["baseline_rank"] is not None and int(row["baseline_rank"]) <= 10
    selected = row["selected_rank"] is not None and int(row["selected_rank"]) <= 10
    if baseline and selected:
        return "both_hit"
    if baseline:
        return "baseline_only_hit"
    if selected:
        return "selected_only_hit"
    return "both_miss"


def frequency_bucket(frequency: int) -> str:
    if frequency <= 1:
        return "1"
    if frequency == 2:
        return "2"
    if frequency <= 5:
        return "3-5"
    if frequency <= 10:
        return "6-10"
    return "11+"


def retrieval_bucket(rank: int | None) -> str:
    if rank is None:
        return "not_in_blair_top50"
    if rank <= 5:
        return "1-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    return "21-50"


def case_record(
    record: dict[str, object],
    *,
    example: SequenceExample,
    seed: int,
    titles: dict[str, str],
    train_frequencies: Counter[str],
    retrieved_items: Sequence[str],
    retrieved_scores: Sequence[float],
) -> dict[str, object]:
    target = example.target_item_id
    target_rank = (
        list(retrieved_items).index(target) + 1 if target in retrieved_items else None
    )
    target_score = (
        float(retrieved_scores[target_rank - 1]) if target_rank is not None else None
    )
    recent_history = []
    for event in example.history[-5:]:
        recent_history.append(
            {
                "item_id": event.item_id,
                "title": titles.get(event.item_id, event.item_id),
                "behavior": event.behavior,
                "behavior_intensity": event.behavior_intensity,
                "time_gap_bucket": event.time_gap_bucket,
            }
        )
    competitors = [
        {
            "retrieval_rank": index + 1,
            "item_id": item_id,
            "title": titles.get(item_id, item_id),
            "train_target_frequency": int(train_frequencies.get(item_id, 0)),
            "blair_similarity": float(score),
        }
        for index, (item_id, score) in enumerate(
            zip(retrieved_items[:5], retrieved_scores[:5], strict=True)
        )
    ]
    return {
        "seed": seed,
        "example_index": int(record["example_index"]),
        "user_hash": record["user_hash"],
        "error_group": error_group(record),
        "pair_outcome": pair_outcome(record),
        "candidate_source": source_group(record),
        "target": {
            "item_id": target,
            "title": titles.get(target, target),
            "train_target_frequency": int(train_frequencies.get(target, 0)),
            "blair_retrieval_rank": target_rank,
            "blair_similarity": target_score,
            "baseline_rank": record["baseline_rank"],
            "selected_rank": record["selected_rank"],
        },
        "history": {**history_features(example), "recent_events": recent_history},
        "top_blair_tail_competitors": competitors,
    }


def select_cases(cases: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen_examples: set[tuple[str, int]] = set()
    for group in (
        "candidate_miss",
        "candidate_present_top10_miss",
        "selected_top10_hit",
    ):
        group_cases = [case for case in cases if case["error_group"] == group]
        if group == "candidate_present_top10_miss":
            group_cases.sort(
                key=lambda case: (
                    case["target"]["blair_retrieval_rank"] is None,
                    case["target"]["blair_retrieval_rank"] or 999,
                    case["target"]["train_target_frequency"],
                    case["seed"],
                )
            )
        elif group == "selected_top10_hit":
            group_cases.sort(
                key=lambda case: (
                    case["target"]["selected_rank"] or 999,
                    case["target"]["train_target_frequency"],
                    case["seed"],
                )
            )
        else:
            group_cases.sort(
                key=lambda case: (
                    case["target"]["train_target_frequency"],
                    case["history"]["history_length"],
                    case["seed"],
                )
            )
        added = 0
        for case in group_cases:
            identity = (group, int(case["example_index"]))
            if identity in seen_examples:
                continue
            seen_examples.add(identity)
            selected.append(case)
            added += 1
            if added >= count:
                break
    return selected


def main() -> None:
    args = parse_args()
    if min(args.retrieval_count, args.retrieval_batch_size, args.cases_per_type) <= 0:
        raise ValueError("retrieval and case counts must be positive")
    config_path = resolve(args.config)
    mild_root = resolve(args.mild_root)
    catalog_path = resolve(args.catalog)
    output_dir = resolve(args.output_dir)
    config = load_config(config_path)
    data = config["data"]
    audit_path = resolve(Path(data["data_audit_path"]))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("synthetic_data_used") is not False:
        raise ValueError("final V2 error analysis requires a passed real-data audit")

    train_path = resolve(Path(data["train_path"]))
    test_path = resolve(Path(data["test_path"]))
    items_path = test_path.parent / "items.jsonl"
    embedding_path = resolve(Path(data["item_embeddings_path"]))
    examples = load_sequence_examples_jsonl(test_path)
    targets = [example.target_item_id for example in examples]
    histories = [[event.item_id for event in example.history] for example in examples]
    titles = load_titles(items_path)
    train_frequencies: Counter[str] = Counter(
        example.target_item_id for example in iter_sequence_examples_jsonl(train_path)
    )
    embedding_ids, embeddings, embedding_metadata = load_embedding_bundle(embedding_path)
    if embedding_metadata.get("backend") != "blair":
        raise ValueError("final V2 error analysis requires BLaIR embeddings")
    catalog = SemanticIdCatalog.load(catalog_path)
    if set(embedding_ids) != set(catalog.item_ids):
        raise ValueError("BLaIR embedding items and Semantic-ID catalog items do not match")
    strata = popularity_strata(catalog.item_ids, train_frequencies)
    tail_items = strata["observed_tail"]
    item_to_embedding = {item_id: index for index, item_id in enumerate(embedding_ids)}
    profiles = build_blair_user_profiles(examples, embeddings, item_to_embedding)
    retrieved_items, retrieved_scores = exact_blair_retrieve(
        profiles,
        [item_id for item_id in catalog.item_ids if item_id in tail_items],
        embeddings,
        item_to_embedding,
        histories,
        top_k=args.retrieval_count,
        batch_size=args.retrieval_batch_size,
    )

    mild_final_path = mild_root / "final_summary.json"
    mild_final = json.loads(mild_final_path.read_text(encoding="utf-8"))
    if mild_final.get("synthetic_data_used") is not False:
        raise ValueError("mild final summary is not verified real data")
    source_counts: Counter[str] = Counter()
    source_error_counts: dict[str, Counter[str]] = defaultdict(Counter)
    error_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    retrieval_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    frequency_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    group_features: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_seed: dict[str, object] = {}
    all_cases: list[dict[str, object]] = []
    input_files: list[Path] = [
        config_path,
        catalog_path,
        audit_path,
        train_path,
        test_path,
        items_path,
        embedding_path,
        mild_final_path,
        mild_root / "locked_validation_selection.json",
        mild_root / "acceptance.json",
        mild_root / "paired_tests.json",
        mild_root / "v2_pareto_summary.json",
    ]

    for seed in SEEDS:
        report_path = mild_root / "test" / f"seed_{seed}" / "report.json"
        indicators_path = mild_root / "test" / f"seed_{seed}" / "hit_indicators.jsonl"
        input_files.extend([report_path, indicators_path])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("synthetic_data_used") is not False:
            raise ValueError(f"seed {seed} report is not verified real data")
        records = [
            json.loads(line)
            for line in indicators_path.read_text(encoding="utf-8").splitlines()
        ]
        if len(records) != len(examples):
            raise ValueError(f"seed {seed} indicators do not align with test examples")
        seed_sources: Counter[str] = Counter()
        seed_errors: Counter[str] = Counter()
        seed_pairs: Counter[str] = Counter()
        seed_tail = 0
        for index, (record, example) in enumerate(zip(records, examples, strict=True)):
            expected_hash = hashlib.sha256(example.user_id.encode()).hexdigest()[:16]
            if (
                int(record["example_index"]) != index
                or record["target_item_id"] != example.target_item_id
                or record["user_hash"] != expected_hash
            ):
                raise ValueError(f"seed {seed} indicator alignment failed at index {index}")
            if record["target_group"] != "observed_tail":
                continue
            seed_tail += 1
            target = targets[index]
            recomputed_blair = target in retrieved_items[index]
            if recomputed_blair != bool(record["target_in_blair_retrieval"]):
                raise ValueError(
                    f"seed {seed} BLaIR retrieval audit failed at test index {index}"
                )
            target_rank = (
                retrieved_items[index].index(target) + 1 if recomputed_blair else None
            )
            source = source_group(record)
            error = error_group(record)
            outcome = pair_outcome(record)
            source_counts[source] += 1
            source_error_counts[source][error] += 1
            error_counts[error] += 1
            pair_counts[outcome] += 1
            seed_sources[source] += 1
            seed_errors[error] += 1
            seed_pairs[outcome] += 1
            retrieval_buckets[retrieval_bucket(target_rank)][error] += 1
            bucket = frequency_bucket(int(train_frequencies[target]))
            frequency_buckets[bucket][error] += 1
            features = history_features(example)
            features["target_train_frequency"] = int(train_frequencies[target])
            if target_rank is not None:
                features["blair_retrieval_rank"] = target_rank
            for name, value in features.items():
                group_features[error][name].append(float(value))
            if error != "candidate_miss" or len(all_cases) < args.cases_per_type * 20:
                all_cases.append(
                    case_record(
                        record,
                        example=example,
                        seed=seed,
                        titles=titles,
                        train_frequencies=train_frequencies,
                        retrieved_items=retrieved_items[index],
                        retrieved_scores=retrieved_scores[index],
                    )
                )
        per_seed[str(seed)] = {
            "observed_tail_examples": seed_tail,
            "candidate_sources": dict(seed_sources),
            "error_groups": dict(seed_errors),
            "paired_top10_outcomes": dict(seed_pairs),
            "candidate_oracle_from_report": report["candidate_oracle"]["union"][
                "target_groups"
            ]["observed_tail"],
            "selected_tail_metrics": report["selected"]["metrics"]["target_groups"][
                "observed_tail"
            ],
        }

    tail_pairs = sum(error_counts.values())
    union_hits = error_counts["candidate_present_top10_miss"] + error_counts[
        "selected_top10_hit"
    ]
    top10_hits = error_counts["selected_top10_hit"]
    retrieval_table: dict[str, object] = {}
    for bucket, counts in sorted(retrieval_buckets.items()):
        total = sum(counts.values())
        retrieval_table[bucket] = {
            "examples": total,
            **dict(counts),
            "top10_conversion": counts["selected_top10_hit"] / total if total else 0.0,
        }
    frequency_table: dict[str, object] = {}
    for bucket in ("1", "2", "3-5", "6-10", "11+"):
        counts = frequency_buckets[bucket]
        total = sum(counts.values())
        candidate_count = counts["candidate_present_top10_miss"] + counts[
            "selected_top10_hit"
        ]
        frequency_table[bucket] = {
            "examples": total,
            **dict(counts),
            "candidate_recall": candidate_count / total if total else 0.0,
            "top10_recall": counts["selected_top10_hit"] / total if total else 0.0,
        }
    feature_table = {
        group: {name: describe(values) for name, values in features.items()}
        for group, features in group_features.items()
    }
    selected_cases = select_cases(all_cases, args.cases_per_type)
    consistency = {
        "test_examples_per_seed": len(examples),
        "observed_tail_examples_per_seed": sum(
            target in tail_items for target in targets
        ),
        "blair_top50_tail_hits_recomputed": sum(
            target in row for target, row in zip(targets, retrieved_items, strict=True)
            if target in tail_items
        ),
        "indicator_alignment_passed": True,
        "blair_indicator_recomputation_passed": True,
    }
    analysis = {
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "category": data["category"],
        "synthetic_data_used": False,
        "split": "test",
        "seeds": list(SEEDS),
        "unit": "seed-example pair",
        "locked_mild_parameters": mild_final["locked_parameters"],
        "popularity_strata": {name: len(items) for name, items in strata.items()},
        "candidate_protocol": {
            "generator": "Beam-20 candidates",
            "blair": "history-only exact retrieval over Observed Tail, top-50",
            "final": "Top-10 fused reranking",
        },
        "candidate_funnel": {
            "observed_tail_seed_example_pairs": tail_pairs,
            "generator_top20_contains_target": source_counts["generator_only"]
            + source_counts["generator_and_blair"],
            "blair_top50_contains_target": source_counts["blair_only"]
            + source_counts["generator_and_blair"],
            "union_contains_target": union_hits,
            "final_top10_hits": top10_hits,
            "union_candidate_recall": union_hits / tail_pairs,
            "final_top10_recall": top10_hits / tail_pairs,
            "candidate_to_top10_conversion": top10_hits / union_hits,
            "candidate_miss_share": error_counts["candidate_miss"] / tail_pairs,
            "candidate_present_but_top10_miss_share": error_counts[
                "candidate_present_top10_miss"
            ]
            / tail_pairs,
        },
        "candidate_sources": dict(source_counts),
        "candidate_source_by_error_group": {
            source: dict(counts) for source, counts in source_error_counts.items()
        },
        "error_groups": dict(error_counts),
        "paired_top10_outcomes": dict(pair_counts),
        "by_blair_retrieval_rank": retrieval_table,
        "by_target_train_frequency": frequency_table,
        "group_descriptives": feature_table,
        "per_seed": per_seed,
        "statistical_result": mild_final["acceptance"],
        "mechanism": {
            "fused_score": "0.75 * generator_relevance + 0.25 * BLaIR_relevance - 0.10 * normalized_log_frequency + soft_quota_bonus",
            "blair_only_generator_relevance": 0.0,
            "soft_quota_profile": "tail_20",
            "soft_quota_strength": 0.01,
            "interpretation": (
                "BLaIR-only tail candidates enter the union but start with zero generator "
                "relevance; a 0.25 semantic weight and 0.01 soft bonus usually cannot "
                "displace high-scoring generator candidates from ten available slots."
            ),
        },
        "conclusions": [
            "Candidate generation remains the first bottleneck because most tail targets are absent from the 70-source union.",
            "Candidate-to-Top10 conversion is a second, larger conditional bottleneck for targets that BLaIR does retrieve.",
            "The mild fusion deliberately preserves overall Recall@10, so it cannot aggressively replace head-biased generator candidates.",
            "Only a handful of tail hits occur per seed, making the three-seed Student-t interval wide even when all paired tail changes are non-negative.",
        ],
        "consistency_checks": consistency,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "error_analysis.json"
    cases_path = output_dir / "error_cases.jsonl"
    provenance_path = output_dir / "provenance.json"
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in selected_cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    provenance = {
        "dataset": analysis["dataset"],
        "synthetic_data_used": False,
        "inputs": {
            str(path.resolve()): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in input_files
        },
        "outputs": {
            str(analysis_path.resolve()): {"sha256": sha256(analysis_path)},
            str(cases_path.resolve()): {"sha256": sha256(cases_path)},
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"analysis": analysis, "cases": selected_cases}, ensure_ascii=False, indent=2))
    print(f"wrote {analysis_path}")
    print(f"wrote {cases_path}")
    print(f"wrote {provenance_path}")


if __name__ == "__main__":
    main()
