import json

import torch

from regenrec.data import (
    Interaction,
    SequenceExample,
    load_sequence_examples_jsonl,
    write_sequence_examples_jsonl,
)
from regenrec.datasets.recif import (
    RecIFPreparationStats,
    build_catalog_from_sid_rows,
    iter_product_examples,
)


def test_recif_product_conversion_filters_unknown_items_and_builds_recency() -> None:
    catalog = build_catalog_from_sid_rows(
        [
            {"pid": 10, "sid": [0, 1, 2]},
            {"pid": 11, "sid": [1, 2, 0]},
            {"pid": 12, "sid": [2, 0, 1]},
        ]
    )
    rows = [
        {
            "uid": 7,
            "split": 0,
            "hist_goods_pid": [999, 10, 11],
            "target_goods_pid": [12],
        },
        {
            "uid": 8,
            "split": 1,
            "hist_goods_pid": [10],
            "target_goods_pid": [11],
        },
    ]
    stats = RecIFPreparationStats()
    examples = list(
        iter_product_examples(
            rows,
            catalog,
            max_history=2,
            num_time_buckets=4,
            allowed_splits={0},
            stats=stats,
        )
    )

    assert len(examples) == 1
    assert [event.item_id for event in examples[0].history] == ["10", "11"]
    assert [event.time_gap_bucket for event in examples[0].history] == [1, 0]
    assert examples[0].target_item_id == "12"
    assert stats.unknown_history_items == 1
    assert stats.rows_filtered_by_split == 1
    assert stats.examples_emitted == 1


def test_recif_catalog_handles_sid_collisions() -> None:
    catalog = build_catalog_from_sid_rows(
        [
            {"pid": "a", "sid": [1, 2, 3]},
            {"pid": "b", "sid": torch.tensor([1, 2, 3])},
        ],
        codebook_size=4,
    )
    assert catalog.resolve(catalog.tokens_for_item("a").tolist()) == "a"
    assert catalog.resolve(catalog.tokens_for_item("b").tolist()) == "b"
    assert not torch.equal(catalog.tokens_for_item("a"), catalog.tokens_for_item("b"))


def test_sequence_jsonl_round_trip(tmp_path) -> None:
    catalog = build_catalog_from_sid_rows([{"pid": 1, "sid": [0, 0]}, {"pid": 2, "sid": [1, 1]}])
    examples = list(
        iter_product_examples(
            [
                {
                    "uid": "u1",
                    "hist_goods_pid": [1],
                    "target_goods_pid": [2],
                }
            ],
            catalog,
        )
    )
    path = tmp_path / "examples.jsonl"
    assert write_sequence_examples_jsonl(examples, path) == 1
    restored = load_sequence_examples_jsonl(path)
    assert restored == examples
    assert json.loads(path.read_text(encoding="utf-8"))["user_id"] == "u1"


def test_sequence_jsonl_reservoir_subset_is_seeded_and_not_a_prefix(tmp_path) -> None:
    examples = [
        SequenceExample(
            user_id=f"u{index}",
            history=(Interaction(item_id="history", behavior=1, time_gap_bucket=0),),
            target_item_id=f"target-{index}",
        )
        for index in range(20)
    ]
    path = tmp_path / "many.jsonl"
    write_sequence_examples_jsonl(examples, path)
    first = load_sequence_examples_jsonl(
        path,
        limit=5,
        selection_strategy="reservoir",
        seed=2026,
    )
    second = load_sequence_examples_jsonl(
        path,
        limit=5,
        selection_strategy="reservoir",
        seed=2026,
    )
    assert first == second
    assert first != examples[:5]
    assert [int(example.user_id[1:]) for example in first] == sorted(
        int(example.user_id[1:]) for example in first
    )
