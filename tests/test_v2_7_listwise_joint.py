from __future__ import annotations

import torch

from scripts.train_v2_5_bounded_residual import exact_mcnemar_probability
from scripts.train_v2_7_listwise_joint import (
    JointScorer,
    joint_features,
    pair_delta,
    weighted_pair_loss,
)


def test_joint_scorer_is_zero_initialized() -> None:
    model = JointScorer(11, 6, 8)
    result = model(torch.randn(5, 11), torch.randn(5, 6))
    assert torch.count_nonzero(result) == 0


def test_pair_delta_preserves_baseline_at_initialization() -> None:
    model = JointScorer(11, 6, 8)
    baseline = torch.tensor([0.5, -0.2])
    delta, positive, negative = pair_delta(
        model,
        torch.randn(2, 11),
        torch.randn(2, 11),
        torch.randn(2, 6),
        baseline,
    )
    assert torch.equal(delta, baseline)
    assert torch.count_nonzero(positive) == 0
    assert torch.count_nonzero(negative) == 0


def test_tail_pair_weight_changes_loss() -> None:
    delta = torch.tensor([-0.5, 0.5])
    head_loss = weighted_pair_loss(
        delta, torch.tensor([0, 0]), margin=0.0, temperature=0.1, tail_weight=8.0
    )
    tail_loss = weighted_pair_loss(
        delta, torch.tensor([2, 0]), margin=0.0, temperature=0.1, tail_weight=8.0
    )
    assert tail_loss > head_loss


def test_joint_features_standardizes_each_row() -> None:
    payload = {
        "candidate_mask": torch.tensor([[True] * 10 + [False], [True] * 11]),
        "semantic_scores": torch.tensor(
            [[float(value) for value in range(11)], [float(value + 2) for value in range(11)]]
        ),
        "beam_scores": torch.tensor(
            [[float(10 - value) for value in range(10)] + [float("-inf")]] * 2
        ),
        "groups": torch.zeros((2, 11), dtype=torch.long),
        "popularity": torch.zeros((2, 11)),
        "history_lengths": torch.tensor([3, 7]),
    }
    candidate, row, base = joint_features(payload)
    assert candidate.shape == (2, 11, 11)
    assert row.shape == (2, 6)
    assert base.shape == (2, 11)
    assert torch.isfinite(candidate).all()


def test_exact_mcnemar_probability_handles_large_discordance() -> None:
    value = exact_mcnemar_probability(30_000, 20_000)
    assert 0.0 <= value < 1e-10
    assert exact_mcnemar_probability(25_000, 25_000) == 1.0
    assert exact_mcnemar_probability(0, 0) == 1.0
