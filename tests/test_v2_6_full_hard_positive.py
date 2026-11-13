from __future__ import annotations

import torch

from scripts.train_v2_6_full_hard_positive import (
    residual_pair_difference,
    tail_discordance,
)


def test_residual_pair_difference_is_baseline_at_zero_initialization() -> None:
    positive = torch.randn(4, 9)
    negative = torch.randn(4, 9)
    row = torch.randn(4, 6)
    base = torch.tensor([0.1, -0.2, 0.0, 0.4])
    difference, residual = residual_pair_difference(
        positive,
        negative,
        row,
        base,
        torch.zeros(9),
        torch.zeros(6),
        torch.tensor(-2.0),
        bound=0.5,
    )
    assert torch.equal(difference, base)
    assert torch.count_nonzero(residual) == 0


def test_residual_pair_difference_respects_two_sided_bound() -> None:
    difference, residual = residual_pair_difference(
        torch.full((2, 9), 100.0),
        torch.full((2, 9), -100.0),
        torch.full((2, 6), 100.0),
        torch.zeros(2),
        torch.ones(9),
        torch.ones(6),
        torch.tensor(100.0),
        bound=0.25,
    )
    assert float(residual.abs().max()) <= 0.25
    assert float(difference.abs().max()) <= 0.5


def test_tail_discordance_counts_only_observed_tail() -> None:
    result = tail_discordance(
        torch.tensor([True, False, True, False]),
        torch.tensor([False, True, True, True]),
        torch.tensor([2, 2, 0, 2]),
    )
    assert result == {"wins": 2, "losses": 1, "net_wins": 1}
