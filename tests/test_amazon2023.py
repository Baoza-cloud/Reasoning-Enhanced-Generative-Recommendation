import csv
import gzip
import json

import torch

from regenrec.datasets.amazon2023 import (
    AmazonObservedEvent,
    AmazonPreparationStats,
    benchmark_url,
    compose_item_text,
    iter_amazon_intensity_examples,
    iter_amazon_examples,
    prepare_amazon2023,
    rating_behavior,
)
from regenrec.embeddings import hashing_text_embeddings, load_item_texts


def test_amazon_url_and_example_conversion() -> None:
    assert benchmark_url("Musical_Instruments", "last_out", "train").endswith(
        "/5core/last_out_w_his/Musical_Instruments.train.csv.gz"
    )
    stats = AmazonPreparationStats()
    item_ids: set[str] = set()
    examples = list(
        iter_amazon_examples(
            [
                {
                    "user_id": "u1",
                    "parent_asin": "d",
                    "history": "a b c",
                    "rating": "5.0",
                    "timestamp": "10",
                },
                {
                    "user_id": "u2",
                    "parent_asin": "e",
                    "history": "",
                    "rating": "4.0",
                    "timestamp": "11",
                },
            ],
            max_history=2,
            num_time_buckets=4,
            stats=stats,
            item_ids=item_ids,
        )
    )
    assert len(examples) == 1
    assert [event.item_id for event in examples[0].history] == ["b", "c"]
    assert [event.time_gap_bucket for event in examples[0].history] == [1, 0]
    assert item_ids == {"b", "c", "d"}
    assert stats.rows_without_history == 1


def test_compose_item_text_deduplicates_metadata() -> None:
    text = compose_item_text(
        {
            "title": "Guitar Cable",
            "store": "Music Shop",
            "categories": ["Music", "Cables"],
            "features": ["Durable", "Durable"],
            "description": ["Ten feet"],
        }
    )
    assert text.count("Durable") == 1
    assert "Guitar Cable" in text
    assert "Ten feet" in text


def test_amazon_v2_uses_observed_rating_and_timestamp() -> None:
    rows = [
        {
            "user_id": "u1",
            "parent_asin": "target",
            "history": "negative positive",
            "rating": "5.0",
            "timestamp": str(10 * 86_400_000),
        }
    ]
    lookup = {
        ("u1", "negative"): AmazonObservedEvent(rating=1.0, timestamp_ms=0),
        ("u1", "positive"): AmazonObservedEvent(
            rating=5.0,
            timestamp_ms=9 * 86_400_000,
        ),
    }
    example = next(iter_amazon_intensity_examples(rows, lookup, num_time_buckets=8))
    assert [event.behavior for event in example.history] == [1, 3]
    assert [event.behavior_intensity for event in example.history] == [0.2, 1.0]
    assert [event.time_gap_bucket for event in example.history] == [3, 1]
    assert rating_behavior(3.0) == 2


def _write_csv(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["user_id", "parent_asin", "rating", "timestamp", "history"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_amazon_and_hash_embeddings(tmp_path) -> None:
    raw = tmp_path / "raw"
    category = "Tiny_Category"
    rows = [
        {
            "user_id": "u1",
            "parent_asin": "c",
            "rating": "5.0",
            "timestamp": "3",
            "history": "a b",
        }
    ]
    for split in ("train", "valid", "test"):
        _write_csv(raw / "5core" / "last_out" / f"{category}.{split}.csv.gz", rows)
    metadata_path = raw / "raw" / f"meta_{category}.jsonl.gz"
    metadata_path.parent.mkdir(parents=True)
    with gzip.open(metadata_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"parent_asin": "a", "title": "Alpha item"}) + "\n")
        handle.write(json.dumps({"parent_asin": "b", "title": "Beta item"}) + "\n")

    output = tmp_path / "processed"
    manifest = prepare_amazon2023(
        raw,
        output,
        category=category,
        split_strategy="last_out",
    )
    assert manifest["items"] == {
        "total": 3,
        "metadata_found": 2,
        "metadata_missing": 1,
    }
    item_ids, texts = load_item_texts(output / "items.jsonl")
    embeddings = hashing_text_embeddings(texts, dimension=16)
    assert set(item_ids) == {"a", "b", "c"}
    assert embeddings.shape == (3, 16)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(3))
