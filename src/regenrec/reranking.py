from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from regenrec.data import SequenceExample


@dataclass(frozen=True)
class CandidateFeatureRow:
    item_ids: tuple[str, ...]
    generator_relevance: tuple[float, ...]
    semantic_relevance: tuple[float, ...]
    popularity: tuple[float, ...]
    groups: tuple[str, ...]


HYBRID_FEATURE_NAMES = (
    "generator_relevance",
    "semantic_relevance",
    "generator_semantic_interaction",
    "popularity",
    "is_torso",
    "is_observed_tail",
    "is_cold_start",
    "semantic_observed_tail_interaction",
    "semantic_torso_interaction",
)


@dataclass(frozen=True)
class LearnedHybridScorer:
    """Validation-trained linear pairwise scorer over leakage-free candidate features."""

    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    scales: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "pairwise_linear_hybrid_v1",
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
            "scales": list(self.scales),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> LearnedHybridScorer:
        if payload.get("type") != "pairwise_linear_hybrid_v1":
            raise ValueError("unsupported hybrid scorer type")
        feature_names = tuple(str(value) for value in payload["feature_names"])
        weights = tuple(float(value) for value in payload["weights"])
        scales = tuple(float(value) for value in payload["scales"])
        if feature_names != HYBRID_FEATURE_NAMES:
            raise ValueError("hybrid scorer feature schema does not match this code")
        if len(weights) != len(feature_names) or len(scales) != len(feature_names):
            raise ValueError("hybrid scorer parameter dimensions are inconsistent")
        if any(not math.isfinite(value) for value in (*weights, *scales)):
            raise ValueError("hybrid scorer parameters must be finite")
        if any(value <= 0 for value in scales):
            raise ValueError("hybrid scorer scales must be positive")
        return cls(feature_names=feature_names, weights=weights, scales=scales)


def _hybrid_feature_vector(row: CandidateFeatureRow, index: int) -> tuple[float, ...]:
    generator = float(row.generator_relevance[index])
    semantic = float(row.semantic_relevance[index])
    popularity = float(row.popularity[index])
    group = row.groups[index]
    is_torso = float(group == "torso")
    is_tail = float(group == "observed_tail")
    is_cold = float(group == "cold_start")
    return (
        generator,
        semantic,
        generator * semantic,
        popularity,
        is_torso,
        is_tail,
        is_cold,
        semantic * is_tail,
        semantic * is_torso,
    )


def fit_pairwise_hybrid_scorer(
    features: Sequence[CandidateFeatureRow],
    targets: Sequence[str],
    *,
    seed: int,
    epochs: int = 300,
    learning_rate: float = 0.05,
    l2_weight: float = 0.01,
    max_negatives_per_positive: int = 20,
    tail_positive_weight: float = 1.5,
) -> tuple[LearnedHybridScorer, dict[str, float | int]]:
    """Fit RankNet-style linear fusion using candidate-contained validation positives.

    Rows whose target is absent from the generator/retrieval union cannot supervise
    a reranker and are reported rather than silently treated as negatives.
    """
    if len(features) != len(targets):
        raise ValueError("features and targets must align")
    if epochs <= 0 or learning_rate <= 0 or l2_weight < 0:
        raise ValueError("invalid hybrid scorer optimization parameters")
    if max_negatives_per_positive <= 0 or tail_positive_weight <= 0:
        raise ValueError("negative count and tail positive weight must be positive")

    differences: list[torch.Tensor] = []
    pair_weights: list[float] = []
    supervised_rows = 0
    supervised_tail_rows = 0
    for row, target in zip(features, targets, strict=True):
        if target not in row.item_ids:
            continue
        positive_index = row.item_ids.index(target)
        negative_indices = [
            index for index in range(len(row.item_ids)) if index != positive_index
        ]
        negative_indices.sort(
            key=lambda index: (
                max(row.generator_relevance[index], row.semantic_relevance[index]),
                row.generator_relevance[index] + row.semantic_relevance[index],
                -index,
            ),
            reverse=True,
        )
        negative_indices = negative_indices[:max_negatives_per_positive]
        if not negative_indices:
            continue
        positive = torch.tensor(
            _hybrid_feature_vector(row, positive_index), dtype=torch.float32
        )
        is_tail = row.groups[positive_index] == "observed_tail"
        example_weight = tail_positive_weight if is_tail else 1.0
        for negative_index in negative_indices:
            negative = torch.tensor(
                _hybrid_feature_vector(row, negative_index), dtype=torch.float32
            )
            differences.append(positive - negative)
            pair_weights.append(example_weight)
        supervised_rows += 1
        supervised_tail_rows += int(is_tail)
    if not differences:
        raise ValueError("no target-positive candidate pairs are available for scorer fitting")

    matrix = torch.stack(differences)
    scales = matrix.std(dim=0, unbiased=False).clamp_min(1e-3)
    normalized = matrix / scales
    initial_weights = torch.zeros(normalized.size(1), dtype=torch.float32)
    initial_weights[0] = scales[0]
    weights = initial_weights.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([weights], lr=learning_rate)
    pair_weight_tensor = torch.tensor(pair_weights, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    final_loss = 0.0
    for _ in range(epochs):
        order = torch.randperm(normalized.size(0), generator=generator)
        logits = normalized[order] @ weights
        ordered_weights = pair_weight_tensor[order]
        ranking_loss = (
            F.softplus(-logits) * ordered_weights
        ).sum() / ordered_weights.sum().clamp_min(1e-12)
        loss = ranking_loss + l2_weight * (weights - initial_weights).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            weights[0].clamp_(min=0.0)
        final_loss = float(loss.detach())

    scorer = LearnedHybridScorer(
        feature_names=HYBRID_FEATURE_NAMES,
        weights=tuple(float(value) for value in weights.detach().tolist()),
        scales=tuple(float(value) for value in scales.tolist()),
    )
    return scorer, {
        "rows": len(features),
        "supervised_rows": supervised_rows,
        "supervised_tail_rows": supervised_tail_rows,
        "pairs": len(differences),
        "candidate_positive_rate": supervised_rows / len(features),
        "epochs": epochs,
        "final_loss": final_loss,
        "tail_positive_weight": tail_positive_weight,
    }


def learned_hybrid_rerank(
    features: Sequence[CandidateFeatureRow],
    scorer: LearnedHybridScorer,
    *,
    top_k: int,
) -> list[list[str]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    weights = torch.tensor(scorer.weights, dtype=torch.float32)
    scales = torch.tensor(scorer.scales, dtype=torch.float32)
    output: list[list[str]] = []
    for row in features:
        item_features = torch.tensor(
            [_hybrid_feature_vector(row, index) for index in range(len(row.item_ids))],
            dtype=torch.float32,
        )
        scores = (item_features / scales) @ weights
        ranked = sorted(
            range(len(row.item_ids)),
            key=lambda index: (float(scores[index]), -index),
            reverse=True,
        )[:top_k]
        output.append([row.item_ids[index] for index in ranked])
    return output


def _minmax(values: Sequence[float], *, rank_fallback: bool = False) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum > minimum:
        return [(float(value) - minimum) / (maximum - minimum) for value in values]
    if rank_fallback and len(values) > 1:
        denominator = len(values) - 1
        return [1.0 - index / denominator for index in range(len(values))]
    return [1.0 for _ in values]


def build_blair_user_profiles(
    examples: Sequence[SequenceExample],
    item_embeddings: torch.Tensor,
    item_to_index: Mapping[str, int],
    *,
    recency_decay: float = 0.12,
    behavior_weights: Mapping[int, float] | None = None,
) -> torch.Tensor:
    """Build leakage-free history profiles from normalized BLaIR item vectors.

    Negative reviews receive zero positive-interest weight by default. Neutral and
    positive reviews receive 0.25 and 1.0 respectively, multiplied by observed
    rating intensity and an exponential time-gap decay. Histories without a
    positive-weight event fall back to their latest observed item.
    """
    if item_embeddings.ndim != 2:
        raise ValueError("item_embeddings must be a matrix")
    if recency_decay < 0 or not math.isfinite(recency_decay):
        raise ValueError("recency_decay must be finite and non-negative")
    weights_by_behavior = dict(behavior_weights or {1: 0.0, 2: 0.25, 3: 1.0})
    profiles: list[torch.Tensor] = []
    for example in examples:
        if not example.history:
            raise ValueError("each example must contain history")
        indices: list[int] = []
        weights: list[float] = []
        for event in example.history:
            if event.item_id not in item_to_index:
                raise ValueError(f"history item missing from embeddings: {event.item_id}")
            indices.append(int(item_to_index[event.item_id]))
            behavior_weight = float(weights_by_behavior.get(event.behavior, 0.0))
            weight = (
                behavior_weight
                * float(event.behavior_intensity)
                * math.exp(-recency_decay * float(event.time_gap_bucket))
            )
            weights.append(max(0.0, weight))
        if sum(weights) <= 0:
            profile = item_embeddings[indices[-1]].to(torch.float32)
        else:
            vectors = item_embeddings[torch.tensor(indices, dtype=torch.long)].to(
                torch.float32
            )
            weight_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
            profile = (vectors * weight_tensor).sum(dim=0) / weight_tensor.sum()
        profiles.append(F.normalize(profile, p=2, dim=0))
    if not profiles:
        raise ValueError("examples cannot be empty")
    return torch.stack(profiles)


def exact_blair_retrieve(
    profiles: torch.Tensor,
    candidate_item_ids: Sequence[str],
    item_embeddings: torch.Tensor,
    item_to_index: Mapping[str, int],
    histories: Sequence[Sequence[str]],
    *,
    top_k: int,
    batch_size: int = 256,
) -> tuple[list[list[str]], list[list[float]]]:
    """Exact inner-product retrieval over a fixed candidate stratum."""
    if profiles.ndim != 2 or profiles.size(1) != item_embeddings.size(1):
        raise ValueError("profile and item embedding dimensions must align")
    if len(profiles) != len(histories):
        raise ValueError("profiles and histories must align")
    if top_k <= 0 or batch_size <= 0:
        raise ValueError("top_k and batch_size must be positive")
    normalized_ids = [str(item_id) for item_id in candidate_item_ids]
    if not normalized_ids:
        raise ValueError("candidate_item_ids cannot be empty")
    if top_k > len(normalized_ids):
        raise ValueError("top_k cannot exceed candidate item count")
    indices = torch.tensor(
        [int(item_to_index[item_id]) for item_id in normalized_ids], dtype=torch.long
    )
    candidate_embeddings = item_embeddings[indices].to(torch.float32)
    candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
    candidate_position = {item_id: index for index, item_id in enumerate(normalized_ids)}
    retrieved_items: list[list[str]] = []
    retrieved_scores: list[list[float]] = []
    for start in range(0, len(profiles), batch_size):
        stop = min(start + batch_size, len(profiles))
        similarities = profiles[start:stop].to(torch.float32) @ candidate_embeddings.T
        for local_index, history in enumerate(histories[start:stop]):
            seen_positions = {
                candidate_position[str(item_id)]
                for item_id in history
                if str(item_id) in candidate_position
            }
            if seen_positions:
                similarities[
                    local_index,
                    torch.tensor(sorted(seen_positions), dtype=torch.long),
                ] = float("-inf")
        scores, positions = similarities.topk(top_k, dim=1)
        for row_positions, row_scores in zip(
            positions.tolist(), scores.tolist(), strict=True
        ):
            retrieved_items.append([normalized_ids[index] for index in row_positions])
            retrieved_scores.append([float(score) for score in row_scores])
    return retrieved_items, retrieved_scores


def score_item_rows(
    profiles: torch.Tensor,
    item_rows: Sequence[Sequence[str]],
    item_embeddings: torch.Tensor,
    item_to_index: Mapping[str, int],
    *,
    batch_size: int = 512,
) -> list[list[float]]:
    """Score aligned variable-length item rows against user BLaIR profiles."""
    if len(profiles) != len(item_rows):
        raise ValueError("profiles and item_rows must align")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output: list[list[float]] = []
    for start in range(0, len(item_rows), batch_size):
        rows = item_rows[start : start + batch_size]
        maximum = max((len(row) for row in rows), default=0)
        if maximum == 0:
            output.extend([[] for _ in rows])
            continue
        indices = torch.zeros((len(rows), maximum), dtype=torch.long)
        mask = torch.zeros((len(rows), maximum), dtype=torch.bool)
        for row_index, row in enumerate(rows):
            for column, item_id in enumerate(row):
                indices[row_index, column] = int(item_to_index[str(item_id)])
                mask[row_index, column] = True
        vectors = item_embeddings[indices].to(torch.float32)
        scores = torch.einsum(
            "bd,bkd->bk",
            profiles[start : start + len(rows)].to(torch.float32),
            vectors,
        )
        for row_index, row in enumerate(rows):
            output.append(scores[row_index, : len(row)].tolist())
    return output


def build_candidate_features(
    generator_items: Sequence[Sequence[str]],
    generator_scores: Sequence[Sequence[float]],
    generator_semantic_scores: Sequence[Sequence[float]],
    retrieval_items: Sequence[Sequence[str]],
    retrieval_scores: Sequence[Sequence[float]],
    item_frequencies: Mapping[str, int],
    item_groups: Mapping[str, str],
) -> list[CandidateFeatureRow]:
    """Create normalized, reusable candidate features for fusion sweeps."""
    row_count = len(generator_items)
    if not all(
        len(rows) == row_count
        for rows in (
            generator_scores,
            generator_semantic_scores,
            retrieval_items,
            retrieval_scores,
        )
    ):
        raise ValueError("all candidate feature inputs must align")
    maximum_frequency = max((int(value) for value in item_frequencies.values()), default=0)
    popularity_denominator = math.log1p(maximum_frequency)
    output: list[CandidateFeatureRow] = []
    for gen_items, gen_scores, gen_semantic, ret_items, ret_scores in zip(
        generator_items,
        generator_scores,
        generator_semantic_scores,
        retrieval_items,
        retrieval_scores,
        strict=True,
    ):
        if len(gen_items) != len(gen_scores) or len(gen_items) != len(gen_semantic):
            raise ValueError("generator items and features must align")
        if len(ret_items) != len(ret_scores):
            raise ValueError("retrieval items and scores must align")
        items: list[str] = []
        generator_raw: dict[str, float] = {}
        semantic_raw: dict[str, float] = {}
        for item_id, score, semantic in zip(
            gen_items, gen_scores, gen_semantic, strict=True
        ):
            item = str(item_id)
            if item not in items:
                items.append(item)
                generator_raw[item] = float(score)
                semantic_raw[item] = float(semantic)
        for item_id, semantic in zip(ret_items, ret_scores, strict=True):
            item = str(item_id)
            if item not in items:
                items.append(item)
            semantic_raw[item] = max(float(semantic), semantic_raw.get(item, -math.inf))
        normalized_generator_values = _minmax(
            [generator_raw[item] for item in items if item in generator_raw],
            rank_fallback=True,
        )
        normalized_generator = {
            item: score
            for item, score in zip(
                [item for item in items if item in generator_raw],
                normalized_generator_values,
                strict=True,
            )
        }
        normalized_semantic_values = _minmax([semantic_raw[item] for item in items])
        groups = []
        popularity = []
        for item in items:
            if item not in item_groups:
                raise ValueError(f"item missing from popularity strata: {item}")
            groups.append(str(item_groups[item]))
            frequency = int(item_frequencies.get(item, 0))
            popularity.append(
                math.log1p(frequency) / popularity_denominator
                if popularity_denominator > 0
                else 0.0
            )
        output.append(
            CandidateFeatureRow(
                item_ids=tuple(items),
                generator_relevance=tuple(
                    normalized_generator.get(item, 0.0) for item in items
                ),
                semantic_relevance=tuple(normalized_semantic_values),
                popularity=tuple(popularity),
                groups=tuple(groups),
            )
        )
    return output


def fused_soft_quota_rerank(
    features: Sequence[CandidateFeatureRow],
    *,
    retrieval_weight: float,
    frequency_alpha: float,
    quota_shares: Mapping[str, float] | None,
    quota_strength: float,
    top_k: int,
) -> list[list[str]]:
    """Fuse generator/BLaIR scores and apply a non-binding proportional quota bonus."""
    ranked = fused_soft_quota_rerank_with_scores(
        features,
        retrieval_weight=retrieval_weight,
        frequency_alpha=frequency_alpha,
        quota_shares=quota_shares,
        quota_strength=quota_strength,
        top_k=top_k,
    )
    return [[item_id for item_id, _ in row] for row in ranked]


def fused_soft_quota_rerank_with_scores(
    features: Sequence[CandidateFeatureRow],
    *,
    retrieval_weight: float,
    frequency_alpha: float,
    quota_shares: Mapping[str, float] | None,
    quota_strength: float,
    top_k: int,
) -> list[list[tuple[str, float]]]:
    """Return reranked items with the objective value used at selection time."""
    if not 0.0 <= retrieval_weight <= 1.0 or not math.isfinite(retrieval_weight):
        raise ValueError("retrieval_weight must be finite and in [0, 1]")
    if frequency_alpha < 0 or not math.isfinite(frequency_alpha):
        raise ValueError("frequency_alpha must be finite and non-negative")
    if quota_strength < 0 or not math.isfinite(quota_strength):
        raise ValueError("quota_strength must be finite and non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    normalized_shares = {
        str(group): float(share) for group, share in (quota_shares or {}).items()
    }
    if any(share < 0 or not math.isfinite(share) for share in normalized_shares.values()):
        raise ValueError("quota shares must be finite and non-negative")
    if normalized_shares and not math.isclose(sum(normalized_shares.values()), 1.0):
        raise ValueError("quota shares must sum to 1")

    output: list[list[tuple[str, float]]] = []
    for row in features:
        if not (
            len(row.item_ids)
            == len(row.generator_relevance)
            == len(row.semantic_relevance)
            == len(row.popularity)
            == len(row.groups)
        ):
            raise ValueError("candidate feature row is misaligned")
        base_scores = [
            (1.0 - retrieval_weight) * generator
            + retrieval_weight * semantic
            - frequency_alpha * popularity
            for generator, semantic, popularity in zip(
                row.generator_relevance,
                row.semantic_relevance,
                row.popularity,
                strict=True,
            )
        ]
        remaining = list(range(len(row.item_ids)))
        selected: list[int] = []
        selected_scores: list[float] = []
        group_counts: dict[str, int] = {}
        while remaining and len(selected) < top_k:
            position = len(selected) + 1

            def objective(index: int) -> tuple[float, float, int]:
                group = row.groups[index]
                desired = normalized_shares.get(group, 0.0) * position
                deficit = max(0.0, desired - group_counts.get(group, 0))
                score = base_scores[index] + quota_strength * deficit
                return score, base_scores[index], -index

            chosen = max(remaining, key=objective)
            selected.append(chosen)
            selected_scores.append(objective(chosen)[0])
            remaining.remove(chosen)
            group = row.groups[chosen]
            group_counts[group] = group_counts.get(group, 0) + 1
        output.append(
            [
                (row.item_ids[index], float(score))
                for index, score in zip(selected, selected_scores, strict=True)
            ]
        )
    return output
