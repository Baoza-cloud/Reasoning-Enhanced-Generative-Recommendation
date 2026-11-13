from __future__ import annotations

import torch

from regenrec.data import Interaction, SequenceExample
from regenrec.reranking import (
    CandidateFeatureRow,
    LearnedHybridScorer,
    build_blair_user_profiles,
    exact_blair_retrieve,
    fit_pairwise_hybrid_scorer,
    fused_soft_quota_rerank,
    learned_hybrid_rerank,
)


def test_blair_profile_uses_positive_history_and_retrieval_excludes_seen_items() -> None:
    item_ids = ["negative", "positive", "tail_a", "tail_b"]
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
    embeddings = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
        ]
    )
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    example = SequenceExample(
        user_id="u1",
        history=(
            Interaction("negative", behavior=1, time_gap_bucket=0, behavior_intensity=0.2),
            Interaction("positive", behavior=3, time_gap_bucket=0, behavior_intensity=1.0),
            Interaction("tail_a", behavior=1, time_gap_bucket=0, behavior_intensity=0.2),
        ),
        target_item_id="tail_b",
    )
    profiles = build_blair_user_profiles([example], embeddings, item_to_index)
    assert profiles[0, 0] > profiles[0, 1]
    items, _ = exact_blair_retrieve(
        profiles,
        ["tail_a", "tail_b"],
        embeddings,
        item_to_index,
        [[event.item_id for event in example.history]],
        top_k=1,
    )
    assert items == [["tail_b"]]


def test_frequency_adjustment_and_soft_quota_can_promote_tail_without_hard_insertion() -> None:
    features = [
        CandidateFeatureRow(
            item_ids=("head_a", "head_b", "torso", "tail"),
            generator_relevance=(1.0, 0.9, 0.7, 0.6),
            semantic_relevance=(0.1, 0.2, 0.8, 1.0),
            popularity=(1.0, 0.9, 0.4, 0.0),
            groups=("head", "head", "torso", "observed_tail"),
        )
    ]
    baseline = fused_soft_quota_rerank(
        features,
        retrieval_weight=0.0,
        frequency_alpha=0.0,
        quota_shares=None,
        quota_strength=0.0,
        top_k=2,
    )
    corrected = fused_soft_quota_rerank(
        features,
        retrieval_weight=0.4,
        frequency_alpha=0.2,
        quota_shares={"head": 0.6, "torso": 0.2, "observed_tail": 0.2},
        quota_strength=0.5,
        top_k=2,
    )
    assert baseline == [["head_a", "head_b"]]
    assert "tail" in corrected[0]


def test_soft_quota_never_invents_missing_group_candidates() -> None:
    features = [
        CandidateFeatureRow(
            item_ids=("head_a", "head_b"),
            generator_relevance=(1.0, 0.9),
            semantic_relevance=(0.2, 0.1),
            popularity=(1.0, 0.8),
            groups=("head", "head"),
        )
    ]
    predictions = fused_soft_quota_rerank(
        features,
        retrieval_weight=0.5,
        frequency_alpha=0.1,
        quota_shares={"head": 0.6, "torso": 0.2, "observed_tail": 0.2},
        quota_strength=10.0,
        top_k=2,
    )
    assert predictions == [["head_a", "head_b"]]


def test_pairwise_hybrid_scorer_learns_semantic_tail_signal_and_round_trips() -> None:
    features = [
        CandidateFeatureRow(
            item_ids=("head_a", "tail_a", "head_b"),
            generator_relevance=(1.0, 0.4, 0.8),
            semantic_relevance=(0.1, 1.0, 0.2),
            popularity=(1.0, 0.0, 0.8),
            groups=("head", "observed_tail", "head"),
        ),
        CandidateFeatureRow(
            item_ids=("head_c", "tail_b", "torso"),
            generator_relevance=(0.9, 0.3, 0.7),
            semantic_relevance=(0.2, 0.95, 0.4),
            popularity=(0.9, 0.0, 0.5),
            groups=("head", "observed_tail", "torso"),
        ),
    ]
    scorer, diagnostics = fit_pairwise_hybrid_scorer(
        features,
        ["tail_a", "tail_b"],
        seed=2026,
        epochs=100,
        max_negatives_per_positive=2,
    )
    restored = LearnedHybridScorer.from_dict(scorer.to_dict())
    predictions = learned_hybrid_rerank(features, restored, top_k=1)
    assert predictions == [["tail_a"], ["tail_b"]]
    assert diagnostics["supervised_rows"] == 2
    assert diagnostics["supervised_tail_rows"] == 2


def test_pairwise_hybrid_scorer_reports_missing_candidate_positives() -> None:
    features = [
        CandidateFeatureRow(
            item_ids=("head", "tail"),
            generator_relevance=(1.0, 0.2),
            semantic_relevance=(0.1, 0.9),
            popularity=(1.0, 0.0),
            groups=("head", "observed_tail"),
        ),
        CandidateFeatureRow(
            item_ids=("head", "tail"),
            generator_relevance=(1.0, 0.2),
            semantic_relevance=(0.1, 0.9),
            popularity=(1.0, 0.0),
            groups=("head", "observed_tail"),
        ),
    ]
    _, diagnostics = fit_pairwise_hybrid_scorer(
        features,
        ["tail", "absent"],
        seed=2026,
        epochs=2,
    )
    assert diagnostics["rows"] == 2
    assert diagnostics["supervised_rows"] == 1
    assert diagnostics["candidate_positive_rate"] == 0.5
