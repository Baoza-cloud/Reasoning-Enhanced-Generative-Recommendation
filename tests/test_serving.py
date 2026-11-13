from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from regenrec.api import create_app
from regenrec.catalog import SemanticIdCatalog
from regenrec.data import Interaction
from regenrec.models.generative import GenerativeRecommender
from regenrec.serving import RecommendationEngine, ServingSettings


def make_engine() -> RecommendationEngine:
    torch.manual_seed(2026)
    item_ids = [f"item-{index:02d}" for index in range(16)]
    codes = torch.tensor(
        [[index % 4, (index // 4) % 4, index % 3] for index in range(len(item_ids))]
    )
    catalog = SemanticIdCatalog.from_codes(item_ids, codes, codebook_size=4)
    config = {
        "data": {
            "max_history": 8,
            "num_behaviors": 4,
            "num_time_buckets": 8,
            "test_path": "unused.jsonl",
        },
        "generator": {
            "model_dim": 16,
            "num_heads": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "feedforward_dim": 32,
            "dropout": 0.0,
            "short_term_window": 3,
            "use_behavior_intensity": True,
        },
    }
    model = GenerativeRecommender(
        vocab_size=catalog.vocab_size,
        pad_token_id=catalog.special_tokens.pad,
        num_behaviors=4,
        num_time_buckets=8,
        model_dim=16,
        num_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        feedforward_dim=32,
        dropout=0.0,
        short_term_window=3,
        max_history_length=8,
        max_target_length=catalog.sequence_length + 2,
        use_behavior_intensity=True,
    )
    settings = ServingSettings(
        config_path=Path("unused"),
        artifact_dir=Path("unused"),
        items_path=Path("unused"),
        embedding_path=Path("unused"),
        max_beam_size=20,
        max_batch_size=4,
        retrieval_count=4,
    )
    embeddings = torch.nn.functional.normalize(torch.randn(16, 12), p=2, dim=1)
    return RecommendationEngine(
        settings=settings,
        config=config,
        catalog=catalog,
        model=model,
        item_metadata={
            item_id: {"title": f"Product {index}", "text": f"Product {index}"}
            for index, item_id in enumerate(item_ids)
        },
        item_embeddings=embeddings,
        embedding_ids=item_ids,
        item_frequencies=Counter(
            {item_id: len(item_ids) - index for index, item_id in enumerate(item_ids)}
        ),
    )


def sample_history() -> tuple[Interaction, ...]:
    return (
        Interaction("item-00", behavior=3, behavior_intensity=1.0, time_gap_bucket=3),
        Interaction("item-01", behavior=2, behavior_intensity=0.6, time_gap_bucket=2),
        Interaction("item-02", behavior=3, behavior_intensity=0.8, time_gap_bucket=0),
    )


def test_engine_returns_catalog_items_and_semantic_ids() -> None:
    engine = make_engine()
    result = engine.recommend([sample_history()], beam_size=10)[0]
    assert result["returned_count"] == 10
    assert len(result["recommendations"]) == 10
    assert result["preference_basis"]["history_length"] == 3
    for row in result["recommendations"]:
        assert row["item_id"] in engine.catalog.item_to_index
        assert row["semantic_id"] == engine.catalog.tokens_for_item(row["item_id"]).tolist()
        assert row["candidate_source"] == "generator"
    assert result["beam_candidates"][0]["semantic_trace"][-1]["component"] == (
        "collision_rank"
    )


def test_long_tail_mode_uses_real_catalog_candidates() -> None:
    engine = make_engine()
    result = engine.recommend(
        [sample_history()], beam_size=10, long_tail_mode="mild"
    )[0]
    assert result["long_tail_mode"] == "mild"
    assert result["returned_count"] == 10
    assert all(
        row["item_id"] in engine.catalog.item_to_index
        for row in result["recommendations"]
    )


def test_engine_rejects_unknown_items() -> None:
    engine = make_engine()
    history = (Interaction("missing", behavior=3, time_gap_bucket=0),)
    try:
        engine.recommend([history], beam_size=10)
    except ValueError as exc:
        assert "unknown catalog item" in str(exc)
    else:
        raise AssertionError("unknown products must be rejected")


def test_fastapi_single_and_batch_contract() -> None:
    engine = make_engine()
    event_rows = [
        {
            "item_id": event.item_id,
            "behavior": event.behavior,
            "behavior_intensity": event.behavior_intensity,
            "time_gap_bucket": event.time_gap_bucket,
        }
        for event in sample_history()
    ]
    with TestClient(create_app(engine=engine)) as client:
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        single = client.post(
            "/v1/recommend", json={"history": event_rows, "beam_size": 10}
        )
        assert single.status_code == 200
        assert single.json()["returned_count"] == 10
        batch = client.post(
            "/v1/recommend/batch",
            json={
                "requests": [
                    {"request_id": "a", "history": event_rows},
                    {"request_id": "b", "history": event_rows},
                ],
                "beam_size": 10,
            },
        )
        assert batch.status_code == 200
        assert [row["request_id"] for row in batch.json()["results"]] == ["a", "b"]
