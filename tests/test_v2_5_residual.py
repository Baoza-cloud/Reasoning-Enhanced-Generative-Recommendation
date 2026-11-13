from __future__ import annotations

import torch

from scripts.train_v2_5_bounded_residual import (
    FEATURE_NAMES,
    GATE_FEATURE_NAMES,
    paired_tests,
    rank_scores,
    score_candidates,
)
from scripts.summarize_v2_5_multiseed import student_t


def test_rank_scores_respects_mask_and_stable_order() -> None:
    values = torch.tensor([[2.0, 3.0, 3.0, -9.0]])
    mask = torch.tensor([[True, True, True, False]])
    ranked = rank_scores(values, mask)
    assert torch.equal(ranked[0, :3], torch.tensor([0.0, 1.0, 0.5]))
    assert torch.isneginf(ranked[0, 3])


def test_bounded_residual_is_exactly_baseline_at_zero_initialization() -> None:
    candidate = torch.randn(2, 4, len(FEATURE_NAMES))
    row = torch.randn(2, len(GATE_FEATURE_NAMES))
    base = torch.randn(2, 4)
    scores, _, residual = score_candidates(
        candidate,
        row,
        base,
        torch.zeros(len(FEATURE_NAMES)),
        torch.randn(len(GATE_FEATURE_NAMES)),
        torch.tensor(3.0),
        bound=0.5,
    )
    assert torch.equal(scores, base)
    assert torch.count_nonzero(residual) == 0


def test_bounded_residual_cannot_exceed_registered_bound() -> None:
    candidate = torch.full((2, 4, len(FEATURE_NAMES)), 100.0)
    row = torch.full((2, len(GATE_FEATURE_NAMES)), 100.0)
    base = torch.zeros(2, 4)
    _, _, residual = score_candidates(
        candidate,
        row,
        base,
        torch.ones(len(FEATURE_NAMES)),
        torch.ones(len(GATE_FEATURE_NAMES)),
        torch.tensor(100.0),
        bound=0.75,
    )
    assert float(residual.abs().max()) <= 0.75


def test_paired_tests_report_exact_discordance_and_tail_delta() -> None:
    result = paired_tests(
        torch.tensor([True, False, True, False]),
        torch.tensor([True, True, False, False]),
        torch.tensor([2, 2, 0, 0]),
        samples=200,
        seed=7,
    )
    assert result["mcnemar_exact_top10"] == {
        "wins": 1,
        "losses": 1,
        "discordant": 2,
        "two_sided_p": 1.0,
    }
    assert result["paired_user_bootstrap_top10"]["mean_delta"] == 0.0
    assert (
        result["observed_tail_paired_user_bootstrap_top10"]["mean_delta"]
        == 0.5
    )


def test_student_t_uses_three_seed_df2_interval() -> None:
    result = student_t([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert result["sample_standard_deviation"] == 1.0
    expected_margin = 4.302652729911275 / (3.0**0.5)
    assert abs(result["ci95_low"] - (2.0 - expected_margin)) < 1e-12
    assert abs(result["ci95_high"] - (2.0 + expected_margin)) < 1e-12
