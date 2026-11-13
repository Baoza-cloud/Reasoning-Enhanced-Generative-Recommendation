import math

import pytest
import torch

from regenrec.catalog import SemanticIdCatalog
from regenrec.data import Interaction, SequenceCollator, SequenceExample
from regenrec.diagnostics import semantic_id_diagnostics
from regenrec.long_tail import (
    inverse_frequency_item_weights,
    semantic_mmr_rerank,
    semantic_token_loss_weights,
)
from regenrec.metrics import (
    mean_confidence_interval,
    popularity_partition,
    popularity_strata,
    popularity_calibrated_rerank,
    ranking_metrics,
    recommendation_coverage_metrics,
    recommendation_strata_coverage_metrics,
)
from regenrec.models.generative import GenerativeRecommender
from regenrec.models.rqvae import RQVAE
from scripts.train_amazon2023 import enforce_codebook_quality


def test_rqvae_forward_backward_and_statistics() -> None:
    torch.manual_seed(7)
    model = RQVAE(
        input_dim=12,
        hidden_dim=16,
        code_dim=8,
        num_levels=3,
        codebook_size=4,
    )
    inputs = torch.randn(10, 12)
    output = model(inputs)
    assert output.reconstruction.shape == inputs.shape
    assert output.codes.shape == (10, 3)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert all(codebook.grad is not None for codebook in model.quantizer.codebooks)
    statistics = model.codebook_statistics(inputs)
    assert len(statistics) == 3
    assert all(0 < level["usage_ratio"] <= 1 for level in statistics)


def test_rqvae_kmeans_initialization_and_ema_updates() -> None:
    torch.manual_seed(19)
    model = RQVAE(
        input_dim=12,
        hidden_dim=16,
        code_dim=4,
        num_levels=2,
        codebook_size=4,
        use_ema=True,
        ema_decay=0.9,
        dead_code_threshold=1.0,
        normalize_reconstruction=True,
    )
    inputs = torch.randn(32, 12)
    initialization = model.initialize_codebooks(
        inputs,
        sample_size=24,
        batch_size=8,
        num_iters=4,
        seed=3,
    )
    assert len(initialization) == 2
    assert all(level["used_codes"] == 4 for level in initialization)
    assert model.quantizer.ema_cluster_size_0.sum().item() == pytest.approx(8.0)
    before = [codebook.detach().clone() for codebook in model.quantizer.codebooks]
    output = model(inputs)
    output.loss.backward()
    assert all(codebook.grad is None for codebook in model.quantizer.codebooks)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.quantizer.codebooks, strict=True)
    )


def test_rqvae_uses_vector_squared_norm_losses() -> None:
    model = RQVAE(
        input_dim=3,
        hidden_dim=4,
        code_dim=2,
        num_levels=1,
        codebook_size=2,
    )
    inputs = torch.randn(5, 3)
    output = model(inputs)
    expected = (output.reconstruction - inputs).pow(2).sum(dim=-1).mean()
    assert output.reconstruction_loss.item() == pytest.approx(expected.item())


def test_semantic_id_diagnostics_reports_dead_codes_and_collisions() -> None:
    diagnostics = semantic_id_diagnostics(
        torch.tensor([[0, 0], [0, 0], [1, 0], [1, 2]]),
        codebook_size=4,
    )
    assert diagnostics["levels"][0]["used_codes"] == 2
    assert diagnostics["levels"][1]["dead_codes"] == 2
    assert diagnostics["code_tuples"]["unique"] == 3
    assert diagnostics["code_tuples"]["collision_groups"] == 1
    assert diagnostics["code_tuples"]["max_collision_size"] == 2


def test_single_target_ranking_metrics() -> None:
    metrics = ranking_metrics(
        [["a", "b", "c"], ["x", "y", "z"]],
        ["b", "missing"],
        ks=(1, 2, 3),
    )
    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_2"] == 0.5
    assert metrics["ndcg_at_2"] == pytest.approx(0.5 / math.log2(3.0))


def test_mean_confidence_interval_uses_student_t_for_small_samples() -> None:
    summary = mean_confidence_interval([1.0, 2.0, 3.0])
    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["standard_deviation"] == pytest.approx(1.0)
    assert summary["ci95_margin"] == pytest.approx(4.303 / math.sqrt(3.0))
    with pytest.raises(ValueError, match="at least two"):
        mean_confidence_interval([1.0])


def test_popularity_partition_and_long_tail_coverage() -> None:
    items = ["a", "b", "c", "d", "e"]
    frequencies = {"a": 10, "b": 5, "c": 1, "d": 0, "e": 0}
    head, tail = popularity_partition(items, frequencies, head_fraction=0.4)
    assert head == frozenset({"a", "b"})
    assert tail == frozenset({"c", "d", "e"})

    metrics = recommendation_coverage_metrics(
        [["a", "c", "c"], ["d", "missing"]],
        catalog_items=items,
        tail_items=tail,
        item_frequencies=frequencies,
    )
    assert metrics["valid_item_rate"] == pytest.approx(0.8)
    assert metrics["duplicate_slot_rate"] == pytest.approx(0.2)
    assert metrics["catalog_coverage"] == pytest.approx(0.6)
    assert metrics["tail_catalog_coverage"] == pytest.approx(2 / 3)
    assert metrics["tail_recommendation_share"] == pytest.approx(0.75)


def test_popularity_strata_isolate_cold_start_and_measure_each_group() -> None:
    items = ["a", "b", "c", "d", "e"]
    frequencies = {"a": 10, "b": 5, "c": 1, "d": 0, "e": 0}
    strata = popularity_strata(
        items,
        frequencies,
        head_fraction=0.2,
        torso_fraction=0.3,
    )
    assert strata == {
        "head": frozenset({"a"}),
        "torso": frozenset({"b"}),
        "observed_tail": frozenset({"c"}),
        "cold_start": frozenset({"d", "e"}),
    }
    coverage = recommendation_strata_coverage_metrics(
        [["a", "b", "c", "d"], ["e", "a"]],
        catalog_items=items,
        strata=strata,
    )
    assert coverage["strata"]["head"]["recommendation_share"] == pytest.approx(2 / 6)
    assert coverage["strata"]["observed_tail"]["catalog_coverage"] == 1.0
    assert coverage["strata"]["cold_start"]["catalog_coverage"] == 1.0


def test_popularity_calibrated_rerank_uses_only_training_frequency() -> None:
    candidates = [["head", "tail", "middle"], ["tail", "head"]]
    scores = [[10.0, 9.6, 9.5], [3.0, 2.9]]
    frequencies = {"head": 99, "middle": 9, "tail": 0}

    unchanged = popularity_calibrated_rerank(
        candidates,
        scores,
        frequencies,
        alpha=0.0,
        top_k=2,
    )
    calibrated = popularity_calibrated_rerank(
        candidates,
        scores,
        frequencies,
        alpha=2.0,
        top_k=2,
    )

    assert unchanged == [["head", "tail"], ["tail", "head"]]
    assert calibrated == [["tail", "middle"], ["tail", "head"]]
    with pytest.raises(ValueError, match="same number"):
        popularity_calibrated_rerank(candidates, scores[:1], frequencies, alpha=1.0, top_k=2)


def test_inverse_frequency_item_weights_are_example_mean_one() -> None:
    frequencies = {"head": 100, "middle": 10, "tail": 1}
    weights = inverse_frequency_item_weights(
        frequencies,
        alpha=0.5,
        max_weight=5.0,
    )
    weighted_mean = sum(frequencies[item] * weights[item] for item in frequencies) / sum(
        frequencies.values()
    )
    assert weighted_mean == pytest.approx(1.0)
    assert weights["tail"] > weights["middle"] > weights["head"]
    assert inverse_frequency_item_weights(
        frequencies, alpha=0.0, max_weight=5.0
    ) == {"head": 1.0, "middle": 1.0, "tail": 1.0}


def test_semantic_token_weights_are_position_aware() -> None:
    catalog = SemanticIdCatalog.from_codes(
        ["a", "b", "c", "d"],
        torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]),
        codebook_size=2,
    )
    frequencies = {"a": 100, "b": 10, "c": 2, "d": 1}
    weights = semantic_token_loss_weights(
        catalog,
        frequencies,
        alpha=0.5,
        max_weight=5.0,
    )
    assert weights.shape == (catalog.sequence_length + 1, catalog.vocab_size)
    common_level_zero = catalog.tokens_for_item("a")[0]
    rare_level_zero = catalog.tokens_for_item("c")[0]
    assert weights[0, rare_level_zero] > weights[0, common_level_zero]
    assert weights[-1, catalog.special_tokens.eos] == pytest.approx(1.0)


def test_semantic_mmr_preserves_relevance_at_zero_and_diversifies() -> None:
    catalog = SemanticIdCatalog.from_codes(
        ["a", "b", "c"],
        torch.tensor([[0, 0], [0, 1], [1, 1]]),
        codebook_size=2,
    )
    candidates = [["a", "b", "c"]]
    scores = [[1.0, 0.99, 0.98]]
    assert semantic_mmr_rerank(
        candidates,
        scores,
        catalog,
        diversity_weight=0.0,
        top_k=2,
    ) == [["a", "b"]]
    assert semantic_mmr_rerank(
        candidates,
        scores,
        catalog,
        diversity_weight=0.6,
        top_k=2,
    ) == [["a", "c"]]


def test_codebook_quality_gate_rejects_collapsed_level() -> None:
    diagnostics = {
        "levels": [
            {"level": 0, "usage_ratio": 0.9},
            {"level": 1, "usage_ratio": 0.1},
        ],
        "code_tuples": {"unique_ratio": 0.8},
    }
    with pytest.raises(RuntimeError, match="levels_below_usage_threshold=\\[1\\]"):
        enforce_codebook_quality(
            diagnostics,
            min_level_usage_ratio=0.75,
            min_unique_tuple_ratio=0.7,
        )


def test_generator_loss_and_constrained_generation() -> None:
    torch.manual_seed(11)
    catalog = SemanticIdCatalog.from_codes(
        ["a", "b", "c"],
        torch.tensor([[0, 1], [1, 0], [1, 1]]),
        codebook_size=2,
    )
    examples = [
        SequenceExample(
            user_id="u1",
            history=(
                Interaction("a", behavior=1, time_gap_bucket=2),
                Interaction("b", behavior=2, time_gap_bucket=1),
            ),
            target_item_id="c",
        ),
        SequenceExample(
            user_id="u2",
            history=(Interaction("c", behavior=1, time_gap_bucket=1),),
            target_item_id="a",
        ),
    ]
    batch = SequenceCollator(catalog, item_loss_weights={"c": 3.0})(examples)
    assert batch["sample_weights"].tolist() == pytest.approx([3.0, 1.0])
    model = GenerativeRecommender(
        vocab_size=catalog.vocab_size,
        pad_token_id=catalog.special_tokens.pad,
        num_behaviors=4,
        num_time_buckets=4,
        model_dim=16,
        num_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        feedforward_dim=32,
        dropout=0.0,
        short_term_window=2,
    )
    logits, preference = model(
        batch["history_tokens"],
        batch["behavior_types"],
        batch["time_gap_buckets"],
        batch["history_mask"],
        batch["target_input"],
    )
    assert logits.shape == (*batch["target_input"].shape, catalog.vocab_size)
    assert preference.short_term_weight.shape == (2, 16)
    loss = model.loss(logits, batch["target_labels"])
    constrained_loss = model.trie_constrained_loss(
        logits,
        batch["target_input"],
        batch["target_labels"],
        catalog,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(constrained_loss)
    token_weights = torch.ones(
        batch["target_labels"].size(1), catalog.vocab_size
    )
    token_weights[0, batch["target_labels"][0, 0]] = 2.0
    weighted_loss = model.loss(
        logits,
        batch["target_labels"],
        sample_weights=batch["sample_weights"],
        token_weights=token_weights,
    )
    weighted_constrained_loss = model.trie_constrained_loss(
        logits,
        batch["target_input"],
        batch["target_labels"],
        catalog,
        sample_weights=batch["sample_weights"],
        token_weights=token_weights,
    )
    assert torch.isfinite(weighted_loss)
    assert torch.isfinite(weighted_constrained_loss)
    (loss + constrained_loss + weighted_loss + weighted_constrained_loss).backward()

    shifted_logits = logits.detach().clone()
    shifted_logits[..., catalog.special_tokens.pad] += 100.0
    shifted_constrained_loss = model.trie_constrained_loss(
        shifted_logits,
        batch["target_input"],
        batch["target_labels"],
        catalog,
    )
    assert shifted_constrained_loss.item() == pytest.approx(constrained_loss.item())
    assert model.loss(shifted_logits, batch["target_labels"]) > loss.detach()

    item_ids, generated = model.generate(
        history_tokens=batch["history_tokens"],
        behavior_types=batch["behavior_types"],
        time_gap_buckets=batch["time_gap_buckets"],
        history_mask=batch["history_mask"],
        catalog=catalog,
    )
    assert all(item_id in catalog.item_ids for item_id in item_ids)
    assert generated.shape == (2, catalog.sequence_length + 2)

    ranked_items, beam_scores = model.generate_beam(
        history_tokens=batch["history_tokens"],
        behavior_types=batch["behavior_types"],
        time_gap_buckets=batch["time_gap_buckets"],
        history_mask=batch["history_mask"],
        catalog=catalog,
        beam_size=3,
    )
    assert all(len(items) == 3 for items in ranked_items)
    assert all(len(set(items)) == 3 for items in ranked_items)
    assert all(set(items) == set(catalog.item_ids) for items in ranked_items)
    assert all(scores == sorted(scores, reverse=True) for scores in beam_scores)

    candidate_tokens = torch.stack(
        [
            torch.stack([catalog.tokens_for_item(item) for item in items])
            for items in ranked_items
        ]
    )
    rescored = model.score_semantic_candidates(
        history_tokens=batch["history_tokens"],
        behavior_types=batch["behavior_types"],
        time_gap_buckets=batch["time_gap_buckets"],
        history_mask=batch["history_mask"],
        candidate_tokens=candidate_tokens,
        candidate_mask=torch.ones(candidate_tokens.shape[:2], dtype=torch.bool),
        pair_batch_size=2,
    )
    assert torch.allclose(rescored, torch.tensor(beam_scores), atol=1e-5)

    top_two_items, top_two_scores = model.generate_beam(
        history_tokens=batch["history_tokens"],
        behavior_types=batch["behavior_types"],
        time_gap_buckets=batch["time_gap_buckets"],
        history_mask=batch["history_mask"],
        catalog=catalog,
        beam_size=3,
        num_return_sequences=2,
        renormalize_allowed_tokens=True,
    )
    assert all(len(items) == 2 for items in top_two_items)
    assert all(len(scores) == 2 for scores in top_two_scores)
    with pytest.raises(ValueError, match="num_return_sequences"):
        model.generate_beam(
            history_tokens=batch["history_tokens"],
            behavior_types=batch["behavior_types"],
            time_gap_buckets=batch["time_gap_buckets"],
            history_mask=batch["history_mask"],
            catalog=catalog,
            beam_size=2,
            num_return_sequences=3,
        )


def test_intensity_aware_generator_exposes_decay_and_event_weights() -> None:
    torch.manual_seed(29)
    catalog = SemanticIdCatalog.from_codes(
        ["a", "b", "c"],
        torch.tensor([[0, 1], [1, 0], [1, 1]]),
        codebook_size=2,
    )
    examples = [
        SequenceExample(
            user_id="u1",
            history=(
                Interaction("a", behavior=1, time_gap_bucket=3, behavior_intensity=0.2),
                Interaction("b", behavior=3, time_gap_bucket=0, behavior_intensity=1.0),
            ),
            target_item_id="c",
        )
    ]
    batch = SequenceCollator(catalog)(examples)
    model = GenerativeRecommender(
        vocab_size=catalog.vocab_size,
        pad_token_id=catalog.special_tokens.pad,
        num_behaviors=4,
        num_time_buckets=4,
        model_dim=16,
        num_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        feedforward_dim=32,
        dropout=0.0,
        short_term_window=2,
        use_behavior_intensity=True,
    )
    model.eval()
    logits, preference = model(
        batch["history_tokens"],
        batch["behavior_types"],
        batch["time_gap_buckets"],
        batch["history_mask"],
        batch["target_input"],
        batch["behavior_intensities"],
    )
    zero_logits, _ = model(
        batch["history_tokens"],
        batch["behavior_types"],
        batch["time_gap_buckets"],
        batch["history_mask"],
        batch["target_input"],
        torch.zeros_like(batch["behavior_intensities"]),
    )
    assert preference.event_weights is not None
    assert preference.decay_rate is not None
    assert torch.all(preference.decay_rate > 0)
    assert torch.all(preference.event_weights >= 0)
    assert not torch.allclose(logits, zero_logits)


def test_learned_history_forgetting_retains_recent_events_and_is_differentiable() -> None:
    torch.manual_seed(30)
    catalog = SemanticIdCatalog.from_codes(
        ["a", "b", "c", "d"],
        torch.tensor([[0, 1], [1, 0], [1, 1], [0, 0]]),
        codebook_size=2,
    )
    examples = [
        SequenceExample(
            user_id="u1",
            history=(
                Interaction("a", behavior=3, time_gap_bucket=3, behavior_intensity=1.0),
                Interaction("b", behavior=3, time_gap_bucket=1, behavior_intensity=1.0),
                Interaction("c", behavior=3, time_gap_bucket=0, behavior_intensity=1.0),
            ),
            target_item_id="d",
        ),
        SequenceExample(
            user_id="u2",
            history=(
                Interaction("a", behavior=3, time_gap_bucket=1, behavior_intensity=1.0),
                Interaction("b", behavior=3, time_gap_bucket=0, behavior_intensity=1.0),
            ),
            target_item_id="c",
        ),
    ]
    batch = SequenceCollator(catalog)(examples)
    model = GenerativeRecommender(
        vocab_size=catalog.vocab_size,
        pad_token_id=catalog.special_tokens.pad,
        num_behaviors=4,
        num_time_buckets=4,
        model_dim=16,
        num_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        feedforward_dim=32,
        dropout=0.0,
        short_term_window=2,
        use_behavior_intensity=True,
        history_forgetting_mode="learned_exponential",
    )
    logits, preference = model(
        batch["history_tokens"],
        batch["behavior_types"],
        batch["time_gap_buckets"],
        batch["history_mask"],
        batch["target_input"],
        batch["behavior_intensities"],
    )
    assert preference.forgetting_rate is not None
    assert preference.history_retention is not None
    assert torch.all(preference.forgetting_rate > 0)
    for row, mask in zip(preference.history_retention, batch["history_mask"], strict=True):
        valid = row[mask]
        assert valid[-1].item() == pytest.approx(1.0)
        assert valid[0] < valid[-1]
        assert torch.count_nonzero(row[~mask]) == 0
    model.loss(logits, batch["target_labels"]).backward()
    assert model.history_forgetting.weight.grad is not None
    assert torch.isfinite(model.history_forgetting.weight.grad).all()


def test_generator_rejects_unknown_history_forgetting_mode() -> None:
    with pytest.raises(ValueError, match="history_forgetting_mode"):
        GenerativeRecommender(
            vocab_size=8,
            pad_token_id=0,
            num_behaviors=4,
            num_time_buckets=4,
            model_dim=16,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            feedforward_dim=32,
            dropout=0.0,
            short_term_window=2,
            history_forgetting_mode="unknown",
        )


def test_mean_pool_preference_ablation_removes_explicit_gate() -> None:
    torch.manual_seed(31)
    catalog = SemanticIdCatalog.from_codes(
        ["a", "b", "c"],
        torch.tensor([[0, 1], [1, 0], [1, 1]]),
        codebook_size=2,
    )
    examples = [
        SequenceExample(
            user_id="u1",
            history=(
                Interaction("a", behavior=1, time_gap_bucket=3, behavior_intensity=0.25),
                Interaction("b", behavior=3, time_gap_bucket=0, behavior_intensity=1.0),
            ),
            target_item_id="c",
        )
    ]
    batch = SequenceCollator(catalog)(examples)
    model = GenerativeRecommender(
        vocab_size=catalog.vocab_size,
        pad_token_id=catalog.special_tokens.pad,
        num_behaviors=4,
        num_time_buckets=4,
        model_dim=16,
        num_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        feedforward_dim=32,
        dropout=0.0,
        short_term_window=2,
        use_behavior_intensity=True,
        preference_reasoning_mode="mean_pool",
    )
    logits, preference = model(
        batch["history_tokens"],
        batch["behavior_types"],
        batch["time_gap_buckets"],
        batch["history_mask"],
        batch["target_input"],
        batch["behavior_intensities"],
    )
    assert torch.allclose(preference.vector, preference.long_term)
    assert torch.count_nonzero(preference.short_term_weight) == 0
    assert preference.event_weights is not None
    assert torch.isfinite(logits).all()


def test_generator_rejects_unknown_preference_reasoning_mode() -> None:
    with pytest.raises(ValueError, match="reasoning_mode"):
        GenerativeRecommender(
            vocab_size=8,
            pad_token_id=0,
            num_behaviors=4,
            num_time_buckets=4,
            model_dim=16,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            feedforward_dim=32,
            dropout=0.0,
            short_term_window=2,
            preference_reasoning_mode="unknown",
        )


def test_fine_grained_preference_reasoning_modes_isolate_components() -> None:
    torch.manual_seed(41)
    catalog = SemanticIdCatalog.from_codes(
        ["a", "b", "c"],
        torch.tensor([[0, 1], [1, 0], [1, 1]]),
        codebook_size=2,
    )
    examples = [
        SequenceExample(
            user_id="u1",
            history=(
                Interaction("a", behavior=1, time_gap_bucket=3, behavior_intensity=0.25),
                Interaction("b", behavior=3, time_gap_bucket=0, behavior_intensity=1.0),
            ),
            target_item_id="c",
        )
    ]
    batch = SequenceCollator(catalog)(examples)

    def build(mode: str) -> GenerativeRecommender:
        return GenerativeRecommender(
            vocab_size=catalog.vocab_size,
            pad_token_id=catalog.special_tokens.pad,
            num_behaviors=4,
            num_time_buckets=4,
            model_dim=16,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            feedforward_dim=32,
            dropout=0.0,
            short_term_window=1,
            use_behavior_intensity=True,
            preference_reasoning_mode=mode,
        ).eval()

    full = build("full")
    state_dict = full.state_dict()
    states = {}
    for mode in (
        "full",
        "long_term_only",
        "short_term_only",
        "fixed_0p5_gate",
        "no_event_attention",
        "no_fusion_mlp",
    ):
        model = build(mode)
        model.load_state_dict(state_dict)
        _, states[mode] = model.encode_history(
            batch["history_tokens"],
            batch["behavior_types"],
            batch["time_gap_buckets"],
            batch["history_mask"],
            batch["behavior_intensities"],
        )

    assert torch.allclose(
        states["long_term_only"].vector,
        states["long_term_only"].long_term,
    )
    assert torch.allclose(
        states["short_term_only"].vector,
        states["short_term_only"].short_term,
    )
    assert torch.all(states["fixed_0p5_gate"].short_term_weight == 0.5)
    learned = states["no_fusion_mlp"]
    expected_mix = (
        learned.short_term_weight * learned.short_term
        + (1.0 - learned.short_term_weight) * learned.long_term
    )
    assert torch.allclose(learned.vector, expected_mix)
    assert not torch.allclose(
        states["full"].event_weights,
        states["no_event_attention"].event_weights,
    )
