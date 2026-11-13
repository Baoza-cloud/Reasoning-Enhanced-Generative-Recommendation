from __future__ import annotations

from collections import Counter

import torch

from regenrec.data import Interaction, SequenceExample
from scripts.analyze_v2_5_teacher_rescoring import hard_positive_census
from scripts.run_v2_6_hard_positive_census import (
    empty_oracle,
    finish_oracle,
    histogram,
    merge_oracle,
)


class TinyCatalog:
    item_to_index = {"a": 0, "b": 1, "c": 2}


def example(user: str, target: str, history_length: int) -> SequenceExample:
    event = Interaction(item_id="a", behavior=1, time_gap_bucket=1)
    return SequenceExample(
        user_id=user,
        history=tuple(event for _ in range(history_length)),
        target_item_id=target,
    )


def test_hard_positive_census_requires_retrieval_without_beam() -> None:
    examples = [example("u1", "b", 3), example("u2", "c", 12)]
    summary, payload = hard_positive_census(
        examples,
        generator_items=[["a"], ["c"]],
        retrieval_items=[["a", "b"], ["a", "c"]],
        catalog=TinyCatalog(),
        train_frequencies=Counter({"a": 10, "b": 2, "c": 1}),
        item_groups={"a": "head", "b": "observed_tail", "c": "observed_tail"},
    )
    assert summary["rows"] == 1
    assert summary["target_groups"]["observed_tail"] == 1
    assert payload["target_catalog_indices"].tolist() == [1]
    assert payload["retrieval_ranks"].tolist() == [2]
    assert payload["history_lengths"].tolist() == [3]
    assert payload["target_train_frequencies"].tolist() == [2]


def test_histogram_and_oracle_aggregation() -> None:
    assert histogram(
        torch.tensor([1, 2, 5, 11]),
        (("one", 1, 1), ("two_to_five", 2, 5), ("six_plus", 6, None)),
    ) == {"one": 1, "two_to_five": 2, "six_plus": 1}
    left = empty_oracle()
    right = empty_oracle()
    right["generator"]["overall"].update({"examples": 4, "hits": 1})
    right["generator"]["target_groups"]["head"].update(
        {"examples": 3, "hits": 1}
    )
    right["generator"]["target_groups"]["observed_tail"].update(
        {"examples": 1, "hits": 0}
    )
    merge_oracle(left, right)
    finish_oracle(left)
    assert left["generator"]["overall"]["recall"] == 0.25
    assert left["generator"]["target_groups"]["head"]["recall"] == 1 / 3
