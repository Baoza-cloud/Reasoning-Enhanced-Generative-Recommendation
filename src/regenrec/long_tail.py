from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from regenrec.catalog import SemanticIdCatalog


def inverse_frequency_item_weights(
    item_frequencies: Mapping[str, int],
    *,
    alpha: float,
    max_weight: float,
) -> dict[str, float]:
    """Build mean-one inverse-frequency weights over observed training targets."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if max_weight < 1.0 or not math.isfinite(max_weight):
        raise ValueError("max_weight must be finite and at least 1")
    normalized = {str(item): int(count) for item, count in item_frequencies.items()}
    if not normalized or any(count <= 0 for count in normalized.values()):
        raise ValueError("item frequencies must contain positive observed counts")
    if alpha == 0.0:
        return {item: 1.0 for item in normalized}

    largest = max(normalized.values())
    raw = {
        item: min(max_weight, (largest / count) ** alpha)
        for item, count in normalized.items()
    }
    denominator = sum(normalized.values())
    weighted_mean = sum(normalized[item] * raw[item] for item in normalized) / denominator
    return {item: weight / weighted_mean for item, weight in raw.items()}


def semantic_token_loss_weights(
    catalog: SemanticIdCatalog,
    item_frequencies: Mapping[str, int],
    *,
    alpha: float,
    max_weight: float,
) -> torch.Tensor:
    """Return position-aware mean-one inverse-frequency Semantic-ID token weights."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if max_weight < 1.0 or not math.isfinite(max_weight):
        raise ValueError("max_weight must be finite and at least 1")
    positions = catalog.sequence_length + 1
    counts = torch.zeros((positions, catalog.vocab_size), dtype=torch.float64)
    total_examples = 0
    for item_id, raw_count in item_frequencies.items():
        count = int(raw_count)
        if count <= 0:
            raise ValueError("item frequencies must be positive for observed targets")
        tokens = catalog.tokens_for_item(str(item_id)).tolist()
        for position, token in enumerate(tokens):
            counts[position, token] += count
        counts[-1, catalog.special_tokens.eos] += count
        total_examples += count
    if total_examples == 0:
        raise ValueError("item frequencies cannot be empty")

    weights = torch.ones_like(counts)
    if alpha == 0.0:
        return weights.to(torch.float32)
    for position in range(positions):
        observed = counts[position] > 0
        position_counts = counts[position, observed]
        largest = float(position_counts.max())
        raw = (largest / position_counts).pow(alpha).clamp(max=max_weight)
        mean = float((raw * position_counts).sum() / position_counts.sum())
        weights[position, observed] = raw / mean
    return weights.to(torch.float32)


def semantic_mmr_rerank(
    candidate_items: Sequence[Sequence[str]],
    candidate_scores: Sequence[Sequence[float]],
    catalog: SemanticIdCatalog,
    *,
    diversity_weight: float,
    top_k: int,
) -> list[list[str]]:
    """Rerank beam candidates with Semantic-ID maximal marginal relevance.

    Scores are min-max normalized within each row. Similarity is the fraction of
    matching RQ-code levels; the collision-rank token is intentionally excluded.
    """
    if len(candidate_items) != len(candidate_scores):
        raise ValueError("candidate items and scores must contain the same number of rows")
    if not 0.0 <= diversity_weight < 1.0 or not math.isfinite(diversity_weight):
        raise ValueError("diversity_weight must be finite and in [0, 1)")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    output: list[list[str]] = []
    for raw_items, raw_scores in zip(candidate_items, candidate_scores, strict=True):
        if len(raw_items) != len(raw_scores):
            raise ValueError("each candidate row must align with its score row")
        items: list[str] = []
        scores: list[float] = []
        for item, score in zip(raw_items, raw_scores, strict=True):
            normalized_item = str(item)
            normalized_score = float(score)
            if normalized_item not in catalog.item_to_index:
                raise ValueError(f"candidate is not in the catalog: {normalized_item}")
            if not math.isfinite(normalized_score):
                raise ValueError("candidate scores must be finite")
            if normalized_item not in items:
                items.append(normalized_item)
                scores.append(normalized_score)
        if diversity_weight == 0.0 or len(items) <= 1:
            output.append(items[:top_k])
            continue

        minimum = min(scores)
        maximum = max(scores)
        if maximum > minimum:
            relevance = [(score - minimum) / (maximum - minimum) for score in scores]
        else:
            denominator = max(1, len(scores) - 1)
            relevance = [1.0 - index / denominator for index in range(len(scores))]
        codes = {
            item: catalog.tokens_for_item(item)[: catalog.num_levels].tolist() for item in items
        }
        remaining = list(range(len(items)))
        selected: list[int] = []
        while remaining and len(selected) < top_k:
            def objective(index: int) -> tuple[float, float, int]:
                if not selected:
                    maximum_similarity = 0.0
                else:
                    maximum_similarity = max(
                        sum(
                            left == right
                            for left, right in zip(
                                codes[items[index]], codes[items[chosen]], strict=True
                            )
                        )
                        / catalog.num_levels
                        for chosen in selected
                    )
                value = (
                    (1.0 - diversity_weight) * relevance[index]
                    - diversity_weight * maximum_similarity
                )
                return value, relevance[index], -index

            chosen = max(remaining, key=objective)
            selected.append(chosen)
            remaining.remove(chosen)
        output.append([items[index] for index in selected])
    return output
