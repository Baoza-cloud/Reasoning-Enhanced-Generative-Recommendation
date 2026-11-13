from __future__ import annotations

import torch

from scripts.analyze_v2_7_top10_boundary import (
    changed_topk_rows,
    rank_positions,
    topk_margin,
)


def test_rank_positions_respects_mask() -> None:
    scores = torch.tensor([[0.2, 0.9, 0.5, 1.0]])
    mask = torch.tensor([[True, True, True, False]])
    assert rank_positions(scores, mask).tolist() == [[3, 1, 2, -1]]


def test_topk_margin_is_target_minus_boundary() -> None:
    scores = torch.tensor([[0.9, 0.8, 0.2], [0.9, 0.4, 0.1]])
    mask = torch.ones_like(scores, dtype=torch.bool)
    positions = torch.tensor([2, -1])
    margin = topk_margin(scores, mask, positions, k=2)
    assert torch.isclose(margin[0], torch.tensor(-0.6))
    assert torch.isnan(margin[1])


def test_changed_topk_rows_counts_swaps() -> None:
    left = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    right = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 1.0, 2.0, 3.0]])
    mask = torch.ones_like(left, dtype=torch.bool)
    changed, swaps = changed_topk_rows(left, right, mask, k=2)
    assert changed.tolist() == [False, True]
    assert swaps.tolist() == [0, 1]
