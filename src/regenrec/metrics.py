from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence


_T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def mean_confidence_interval(values: Sequence[float]) -> dict[str, float | int]:
    """Return a two-sided 95% Student-t interval across independent runs."""
    normalized = [float(value) for value in values]
    if len(normalized) < 2:
        raise ValueError("at least two independent values are required for a confidence interval")
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("confidence interval values must be finite")
    count = len(normalized)
    mean = statistics.mean(normalized)
    standard_deviation = statistics.stdev(normalized)
    critical = _T_975.get(count - 1, 1.96)
    margin = critical * standard_deviation / math.sqrt(count)
    return {
        "n": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "ci95_margin": margin,
    }


def ranking_metrics(
    predictions: Sequence[Sequence[str]],
    targets: Sequence[str],
    *,
    ks: Sequence[int] = (5, 10),
) -> dict[str, float]:
    """Compute single-relevant-item Recall@K and NDCG@K."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must contain the same number of examples")
    if not predictions:
        raise ValueError("predictions cannot be empty")
    normalized_ks = sorted(set(int(k) for k in ks))
    if not normalized_ks or normalized_ks[0] <= 0:
        raise ValueError("ks must contain positive integers")

    recalls = {k: 0.0 for k in normalized_ks}
    ndcgs = {k: 0.0 for k in normalized_ks}
    for ranked_items, target in zip(predictions, targets, strict=True):
        try:
            rank = list(ranked_items).index(target) + 1
        except ValueError:
            rank = None
        for k in normalized_ks:
            if rank is not None and rank <= k:
                recalls[k] += 1.0
                ndcgs[k] += 1.0 / math.log2(rank + 1)

    count = len(targets)
    metrics: dict[str, float] = {}
    for k in normalized_ks:
        metrics[f"recall_at_{k}"] = recalls[k] / count
        metrics[f"ndcg_at_{k}"] = ndcgs[k] / count
    return metrics


def popularity_partition(
    item_ids: Sequence[str],
    item_frequencies: Mapping[str, int],
    *,
    head_fraction: float = 0.2,
) -> tuple[frozenset[str], frozenset[str]]:
    """Split a catalog into popular head and long tail using training frequency."""
    normalized_items = [str(item_id) for item_id in item_ids]
    if not normalized_items:
        raise ValueError("item_ids cannot be empty")
    if len(set(normalized_items)) != len(normalized_items):
        raise ValueError("item_ids must be unique")
    if not 0.0 < head_fraction < 1.0:
        raise ValueError("head_fraction must be in (0, 1)")
    if any(int(count) < 0 for count in item_frequencies.values()):
        raise ValueError("item frequencies must be non-negative")

    ranked = sorted(
        normalized_items,
        key=lambda item_id: (-int(item_frequencies.get(item_id, 0)), item_id),
    )
    head_count = max(1, math.ceil(len(ranked) * head_fraction))
    head = frozenset(ranked[:head_count])
    return head, frozenset(ranked[head_count:])


def popularity_strata(
    item_ids: Sequence[str],
    item_frequencies: Mapping[str, int],
    *,
    head_fraction: float = 0.2,
    torso_fraction: float = 0.3,
) -> dict[str, frozenset[str]]:
    """Split observed items by popularity and isolate zero-frequency cold-start items.

    Fractions are applied only to items observed as training targets. The remaining
    observed items form ``observed_tail``; zero-frequency catalog items form
    ``cold_start`` and never affect observed-item quantile boundaries.
    """
    normalized_items = [str(item_id) for item_id in item_ids]
    if not normalized_items:
        raise ValueError("item_ids cannot be empty")
    if len(set(normalized_items)) != len(normalized_items):
        raise ValueError("item_ids must be unique")
    if not 0.0 < head_fraction < 1.0:
        raise ValueError("head_fraction must be in (0, 1)")
    if not 0.0 <= torso_fraction < 1.0:
        raise ValueError("torso_fraction must be in [0, 1)")
    if head_fraction + torso_fraction >= 1.0:
        raise ValueError("head_fraction + torso_fraction must be below 1")
    if any(int(count) < 0 for count in item_frequencies.values()):
        raise ValueError("item frequencies must be non-negative")

    observed = sorted(
        (
            item_id
            for item_id in normalized_items
            if int(item_frequencies.get(item_id, 0)) > 0
        ),
        key=lambda item_id: (-int(item_frequencies.get(item_id, 0)), item_id),
    )
    cold_start = frozenset(
        item_id
        for item_id in normalized_items
        if int(item_frequencies.get(item_id, 0)) == 0
    )
    if not observed:
        return {
            "head": frozenset(),
            "torso": frozenset(),
            "observed_tail": frozenset(),
            "cold_start": cold_start,
        }
    head_end = min(len(observed), max(1, math.ceil(len(observed) * head_fraction)))
    torso_end = min(
        len(observed),
        max(head_end, math.ceil(len(observed) * (head_fraction + torso_fraction))),
    )
    return {
        "head": frozenset(observed[:head_end]),
        "torso": frozenset(observed[head_end:torso_end]),
        "observed_tail": frozenset(observed[torso_end:]),
        "cold_start": cold_start,
    }


def recommendation_strata_coverage_metrics(
    predictions: Sequence[Sequence[str]],
    *,
    catalog_items: Sequence[str],
    strata: Mapping[str, frozenset[str]],
) -> dict[str, object]:
    """Measure recommendation coverage and slot share for disjoint catalog strata."""
    if not predictions:
        raise ValueError("predictions cannot be empty")
    catalog = frozenset(str(item_id) for item_id in catalog_items)
    if not catalog:
        raise ValueError("catalog_items cannot be empty")
    normalized_strata = {
        str(name): frozenset(str(item_id) for item_id in items)
        for name, items in strata.items()
    }
    assigned: set[str] = set()
    for name, items in normalized_strata.items():
        if not items.issubset(catalog):
            raise ValueError(f"stratum {name} contains an item outside the catalog")
        overlap = assigned.intersection(items)
        if overlap:
            raise ValueError(f"strata must be disjoint; overlap found in {name}")
        assigned.update(items)
    if assigned != set(catalog):
        raise ValueError("strata must cover the complete catalog")

    total_slots = 0
    valid_recommendations: list[str] = []
    for ranked_items in predictions:
        normalized = [str(item_id) for item_id in ranked_items]
        total_slots += len(normalized)
        valid_recommendations.extend(item_id for item_id in normalized if item_id in catalog)
    if total_slots == 0:
        raise ValueError("predictions must contain at least one recommendation")
    unique_recommended = frozenset(valid_recommendations)
    return {
        "examples": len(predictions),
        "recommendation_slots": total_slots,
        "valid_recommendation_slots": len(valid_recommendations),
        "strata": {
            name: {
                "catalog_items": len(items),
                "unique_recommended_items": len(unique_recommended & items),
                "catalog_coverage": len(unique_recommended & items) / max(1, len(items)),
                "recommendation_slots": sum(
                    item_id in items for item_id in valid_recommendations
                ),
                "recommendation_share": sum(
                    item_id in items for item_id in valid_recommendations
                )
                / max(1, len(valid_recommendations)),
            }
            for name, items in normalized_strata.items()
        },
    }


def recommendation_coverage_metrics(
    predictions: Sequence[Sequence[str]],
    *,
    catalog_items: Sequence[str],
    tail_items: frozenset[str],
    item_frequencies: Mapping[str, int] | None = None,
) -> dict[str, float | int]:
    """Measure catalog and long-tail coverage for ranked recommendation lists."""
    if not predictions:
        raise ValueError("predictions cannot be empty")
    catalog = frozenset(str(item_id) for item_id in catalog_items)
    if not catalog:
        raise ValueError("catalog_items cannot be empty")
    if not tail_items.issubset(catalog):
        raise ValueError("tail_items must be a subset of catalog_items")
    head_items = catalog - tail_items

    total_slots = 0
    duplicate_slots = 0
    valid_recommendations: list[str] = []
    for ranked_items in predictions:
        normalized = [str(item_id) for item_id in ranked_items]
        total_slots += len(normalized)
        duplicate_slots += len(normalized) - len(set(normalized))
        valid_recommendations.extend(item_id for item_id in normalized if item_id in catalog)
    if total_slots == 0:
        raise ValueError("predictions must contain at least one recommendation")

    unique_recommended = frozenset(valid_recommendations)
    unique_head = unique_recommended & head_items
    unique_tail = unique_recommended & tail_items
    tail_slots = sum(item_id in tail_items for item_id in valid_recommendations)
    frequencies = item_frequencies or {}
    recommended_frequencies = [
        float(frequencies.get(item_id, 0)) for item_id in valid_recommendations
    ]
    return {
        "examples": len(predictions),
        "recommendation_slots": total_slots,
        "mean_list_length": total_slots / len(predictions),
        "valid_item_rate": len(valid_recommendations) / total_slots,
        "duplicate_slot_rate": duplicate_slots / total_slots,
        "unique_recommended_items": len(unique_recommended),
        "catalog_coverage": len(unique_recommended) / len(catalog),
        "head_unique_recommended_items": len(unique_head),
        "head_catalog_coverage": len(unique_head) / max(1, len(head_items)),
        "tail_unique_recommended_items": len(unique_tail),
        "tail_catalog_coverage": len(unique_tail) / max(1, len(tail_items)),
        "tail_recommendation_share": tail_slots / max(1, len(valid_recommendations)),
        "mean_recommended_train_frequency": statistics.mean(recommended_frequencies)
        if recommended_frequencies
        else 0.0,
        "median_recommended_train_frequency": statistics.median(recommended_frequencies)
        if recommended_frequencies
        else 0.0,
    }


def popularity_calibrated_rerank(
    candidate_items: Sequence[Sequence[str]],
    candidate_scores: Sequence[Sequence[float]],
    item_frequencies: Mapping[str, int],
    *,
    alpha: float,
    top_k: int,
) -> list[list[str]]:
    """Rerank beam candidates with a training-frequency popularity penalty.

    Frequencies are log-scaled and normalized by the largest training frequency,
    so ``alpha`` is the maximum score penalty applied to the most popular item.
    Candidate generation scores remain untouched and no evaluation labels are used.
    """
    if len(candidate_items) != len(candidate_scores):
        raise ValueError("candidate items and scores must contain the same number of rows")
    if alpha < 0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if any(int(count) < 0 for count in item_frequencies.values()):
        raise ValueError("item frequencies must be non-negative")

    maximum_frequency = max((int(count) for count in item_frequencies.values()), default=0)
    denominator = math.log1p(maximum_frequency)
    reranked: list[list[str]] = []
    for items, scores in zip(candidate_items, candidate_scores, strict=True):
        if len(items) != len(scores):
            raise ValueError("each candidate row must align with its score row")
        seen: set[str] = set()
        ranked: list[tuple[float, int, str]] = []
        for original_rank, (raw_item, raw_score) in enumerate(
            zip(items, scores, strict=True)
        ):
            item_id = str(raw_item)
            score = float(raw_score)
            if item_id in seen:
                continue
            if not math.isfinite(score):
                raise ValueError("candidate scores must be finite")
            seen.add(item_id)
            frequency = int(item_frequencies.get(item_id, 0))
            popularity = math.log1p(frequency) / denominator if denominator > 0 else 0.0
            ranked.append((score - alpha * popularity, original_rank, item_id))
        ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
        reranked.append([item_id for _, _, item_id in ranked[:top_k]])
    return reranked


def uplift_ranking_metrics(
    uplift_scores: Sequence[float],
    treatment: Sequence[int],
    outcome: Sequence[int],
    *,
    top_fraction: float = 0.1,
) -> dict[str, float]:
    """Compute AUUC, Qini coefficient, and top-fraction uplift on logged RCT data."""
    if not (len(uplift_scores) == len(treatment) == len(outcome)):
        raise ValueError("uplift scores, treatment, and outcome must align")
    if not uplift_scores:
        raise ValueError("uplift inputs cannot be empty")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if any(value not in (0, 1) for value in treatment):
        raise ValueError("treatment must be binary")
    if any(value not in (0, 1) for value in outcome):
        raise ValueError("outcome must be binary")
    if len(set(treatment)) != 2:
        raise ValueError("treated and control observations are both required")

    order = sorted(range(len(uplift_scores)), key=lambda index: (-uplift_scores[index], index))
    treated_count = control_count = 0
    treated_outcomes = control_outcomes = 0.0
    uplift_curve: list[float] = []
    qini_curve: list[float] = []
    top_count = max(1, math.ceil(len(order) * top_fraction))
    top_uplift = 0.0
    for rank, index in enumerate(order, start=1):
        if treatment[index] == 1:
            treated_count += 1
            treated_outcomes += outcome[index]
        else:
            control_count += 1
            control_outcomes += outcome[index]
        if treated_count and control_count:
            uplift = treated_outcomes / treated_count - control_outcomes / control_count
            qini_gain = treated_outcomes - treated_count * control_outcomes / control_count
        else:
            uplift = 0.0
            qini_gain = 0.0
        uplift_curve.append(uplift)
        qini_curve.append(qini_gain)
        if rank == top_count:
            top_uplift = uplift

    final_qini = qini_curve[-1]
    random_qini = [final_qini * rank / len(order) for rank in range(1, len(order) + 1)]
    return {
        "auuc": sum(uplift_curve) / len(uplift_curve),
        "qini_coefficient": sum(
            observed - random for observed, random in zip(qini_curve, random_qini, strict=True)
        )
        / len(order),
        f"uplift_at_{int(top_fraction * 100)}_percent": top_uplift,
        "overall_ate": uplift_curve[-1],
    }
