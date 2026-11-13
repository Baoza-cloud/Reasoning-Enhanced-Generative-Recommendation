from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from regenrec.data import RecommendationDataset, SequenceCollator, load_sequence_examples_jsonl
from regenrec.diagnostics import semantic_id_diagnostics
from regenrec.embeddings import load_embedding_bundle
from regenrec.long_tail import (
    inverse_frequency_item_weights,
    semantic_mmr_rerank,
    semantic_token_loss_weights,
)
from regenrec.metrics import (
    popularity_partition,
    ranking_metrics,
    recommendation_coverage_metrics,
)
from regenrec.models.generative import GenerativeRecommender
from regenrec.models.rqvae import RQVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V1 on Amazon Reviews 2023")
    parser.add_argument("--config", default="configs/v1_amazon2023.yaml")
    parser.add_argument("--seed", type=int, help="Override the configured random seed")
    parser.add_argument("--output-dir", help="Override output_dir for an independent run")
    parser.add_argument("--resume-from", help="Load a generator checkpoint from another run")
    parser.add_argument(
        "--preference-reasoning-mode",
        choices=(
            "full",
            "mean_pool",
            "long_term_only",
            "short_term_only",
            "fixed_0p5_gate",
            "no_event_attention",
            "no_fusion_mlp",
        ),
        help="Override generator.preference_reasoning_mode for a controlled retraining run",
    )
    parser.add_argument(
        "--history-forgetting-mode",
        choices=("none", "learned_exponential"),
        help="Override generator.history_forgetting_mode for a controlled retraining run",
    )
    return parser.parse_args()


def generator_loss(
    model: GenerativeRecommender,
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    catalog: SemanticIdCatalog,
    *,
    loss_type: str,
    position_weights: torch.Tensor | None,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if loss_type == "global_cross_entropy":
        return model.loss(
            logits,
            batch["target_labels"],
            sample_weights=batch.get("sample_weights"),
            token_weights=token_weights,
        )
    if loss_type == "trie_constrained":
        return model.trie_constrained_loss(
            logits,
            batch["target_input"],
            batch["target_labels"],
            catalog,
            position_weights=position_weights,
            sample_weights=batch.get("sample_weights"),
            token_weights=token_weights,
        )
    raise ValueError(f"unknown generator loss_type: {loss_type}")


def train_rqvae(
    model: RQVAE,
    embeddings: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    kmeans_init: bool,
    kmeans_iters: int,
    kmeans_sample_size: int,
    seed: int,
) -> list[dict[str, float]]:
    initialization: list[dict[str, float]] = []
    if kmeans_init:
        initialization = model.initialize_codebooks(
            embeddings,
            sample_size=kmeans_sample_size,
            batch_size=batch_size,
            num_iters=kmeans_iters,
            seed=seed,
        )
        print(json.dumps({"kmeans_initialization": initialization}, indent=2))
    loader = DataLoader(TensorDataset(embeddings), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_reconstruction = 0.0
        total_codebook = 0.0
        total_commitment = 0.0
        batches = 0
        for (batch,) in loader:
            output = model(batch.to(device))
            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            optimizer.step()
            total_loss += output.loss.item()
            total_reconstruction += output.reconstruction_loss.item()
            total_codebook += output.codebook_loss.item()
            total_commitment += output.commitment_loss.item()
            batches += 1
        denominator = max(1, batches)
        print(
            f"rqvae epoch={epoch + 1} loss={total_loss / denominator:.4f} "
            f"reconstruction={total_reconstruction / denominator:.4f} "
            f"codebook={total_codebook / denominator:.4f} "
            f"commitment={total_commitment / denominator:.4f}"
        )
    return initialization


@torch.no_grad()
def encode_catalog(
    model: RQVAE,
    item_ids: list[str],
    embeddings: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    codebook_size: int,
) -> tuple[SemanticIdCatalog, torch.Tensor]:
    codes = []
    model.eval()
    for start in range(0, embeddings.size(0), batch_size):
        batch = embeddings[start : start + batch_size].to(device)
        codes.append(model.encode_codes(batch).cpu())
    all_codes = torch.cat(codes)
    catalog = SemanticIdCatalog.from_codes(
        item_ids,
        all_codes,
        codebook_size=codebook_size,
    )
    return catalog, all_codes


def make_loader(
    path: str,
    catalog: SemanticIdCatalog,
    *,
    batch_size: int,
    limit: int | None,
    shuffle: bool,
    item_loss_weights: Mapping[str, float] | None = None,
    sampling_item_weights: Mapping[str, float] | None = None,
    selection_strategy: str = "prefix",
    selection_seed: int = 0,
) -> tuple[DataLoader, int]:
    examples = load_sequence_examples_jsonl(
        path,
        limit=limit,
        selection_strategy=selection_strategy,
        seed=selection_seed,
    )
    sampler = None
    if sampling_item_weights is not None:
        row_weights = torch.tensor(
            [
                float(sampling_item_weights.get(example.target_item_id, 1.0))
                for example in examples
            ],
            dtype=torch.double,
        )
        if torch.any(row_weights <= 0) or not torch.all(torch.isfinite(row_weights)):
            raise ValueError("sampling weights must be finite and positive")
        sampler = WeightedRandomSampler(
            row_weights,
            num_samples=len(examples),
            replacement=True,
        )
    loader = DataLoader(
        RecommendationDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=SequenceCollator(catalog, item_loss_weights=item_loss_weights),
    )
    return loader, len(examples)


def group_ranking_metrics(
    predictions: list[list[str]],
    targets: list[str],
    group: frozenset[str],
    *,
    ks: tuple[int, ...],
) -> dict[str, float | int]:
    selected = [
        (prediction, target)
        for prediction, target in zip(predictions, targets, strict=True)
        if target in group
    ]
    if not selected:
        return {"examples": 0}
    group_predictions, group_targets = zip(*selected, strict=True)
    return {
        "examples": len(selected),
        **ranking_metrics(group_predictions, group_targets, ks=ks),
    }


@torch.no_grad()
def evaluate(
    model: GenerativeRecommender,
    loader: DataLoader,
    *,
    device: torch.device,
    catalog: SemanticIdCatalog,
    beam_size: int,
    num_return_sequences: int,
    candidate_count: int,
    renormalize_allowed_tokens: bool,
    diversity_weight: float,
    loss_type: str,
    position_weights: torch.Tensor | None,
    ks: tuple[int, ...],
    split_name: str,
    item_frequencies: Mapping[str, int],
    head_items: frozenset[str],
    tail_items: frozenset[str],
    log_every_batches: int = 100,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_predictions: list[list[str]] = []
    all_targets: list[str] = []
    decode_seconds = 0.0
    for batch_index, batch in enumerate(loader, start=1):
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        logits, _ = model(
            batch["history_tokens"],
            batch["behavior_types"],
            batch["time_gap_buckets"],
            batch["history_mask"],
            batch["target_input"],
            batch["behavior_intensities"],
        )
        batch_size = batch["history_tokens"].size(0)
        total_loss += generator_loss(
            model,
            logits,
            batch,
            catalog,
            loss_type=loss_type,
            position_weights=position_weights,
        ).item() * batch_size
        decode_started = time.perf_counter()
        candidates, candidate_scores = model.generate_beam(
            history_tokens=batch["history_tokens"],
            behavior_types=batch["behavior_types"],
            time_gap_buckets=batch["time_gap_buckets"],
            history_mask=batch["history_mask"],
            catalog=catalog,
            beam_size=beam_size,
            num_return_sequences=candidate_count,
            renormalize_allowed_tokens=renormalize_allowed_tokens,
            behavior_intensities=batch["behavior_intensities"],
        )
        predicted = semantic_mmr_rerank(
            candidates,
            candidate_scores,
            catalog,
            diversity_weight=diversity_weight,
            top_k=num_return_sequences,
        )
        decode_seconds += time.perf_counter() - decode_started
        targets = [catalog.resolve(row.tolist()) for row in batch["target_labels"][:, :-1]]
        if any(target is None for target in targets):
            raise RuntimeError("evaluation target could not be resolved in the semantic catalog")
        all_predictions.extend(predicted)
        all_targets.extend(str(target) for target in targets)
        total += batch_size
        if batch_index % log_every_batches == 0:
            print(f"evaluate split={split_name} batches={batch_index} examples={total}")
    metrics = ranking_metrics(all_predictions, all_targets, ks=ks)
    return {
        "loss": total_loss / max(1, total),
        **metrics,
        "target_groups": {
            "head": group_ranking_metrics(
                all_predictions, all_targets, head_items, ks=ks
            ),
            "tail": group_ranking_metrics(
                all_predictions, all_targets, tail_items, ks=ks
            ),
        },
        "coverage": recommendation_coverage_metrics(
            all_predictions,
            catalog_items=catalog.item_ids,
            tail_items=tail_items,
            item_frequencies=item_frequencies,
        ),
        "efficiency": {
            "decode_wall_seconds": decode_seconds,
            "examples_per_second": total / max(decode_seconds, 1e-12),
            "milliseconds_per_example": 1000.0 * decode_seconds / max(total, 1),
        },
    }


def enforce_codebook_quality(
    diagnostics: dict[str, object],
    *,
    min_level_usage_ratio: float,
    min_unique_tuple_ratio: float,
) -> None:
    levels = diagnostics["levels"]
    if not isinstance(levels, list):
        raise TypeError("rqvae diagnostics levels must be a list")
    failed_levels = [
        int(level["level"])
        for level in levels
        if float(level["usage_ratio"]) < min_level_usage_ratio
    ]
    code_tuples = diagnostics["code_tuples"]
    if not isinstance(code_tuples, dict):
        raise TypeError("rqvae diagnostics code_tuples must be a mapping")
    unique_ratio = float(code_tuples["unique_ratio"])
    if failed_levels or unique_ratio < min_unique_tuple_ratio:
        raise RuntimeError(
            "RQ-VAE codebook quality gate failed: "
            f"levels_below_usage_threshold={failed_levels}, "
            f"unique_tuple_ratio={unique_ratio:.6f}"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.preference_reasoning_mode is not None:
        config["generator"]["preference_reasoning_mode"] = (
            args.preference_reasoning_mode
        )
    if args.history_forgetting_mode is not None:
        config["generator"]["history_forgetting_mode"] = args.history_forgetting_mode
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = config["data"]
    audit_path = Path(
        data_config.get(
            "data_audit_path",
            Path(data_config["item_embeddings_path"]).parent / "data_audit.json",
        )
    )
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"missing data provenance audit: {audit_path}; run scripts/audit_amazon2023.py"
        )
    data_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if data_audit.get("status") != "passed" or data_audit.get("synthetic_data_used") is not False:
        raise ValueError("Amazon training requires a passed real-data provenance audit")
    item_ids, item_embeddings, embedding_metadata = load_embedding_bundle(
        data_config["item_embeddings_path"]
    )
    expected_items = int(data_audit["checks"]["actual_unique_items"])
    if len(item_ids) != expected_items:
        raise ValueError(
            f"embedding item count {len(item_ids)} does not match audited count {expected_items}"
        )
    rq_config = config["rqvae"]
    rqvae = RQVAE(
        input_dim=item_embeddings.size(1),
        hidden_dim=int(rq_config["hidden_dim"]),
        code_dim=int(rq_config["code_dim"]),
        num_levels=int(rq_config["num_levels"]),
        codebook_size=int(rq_config["codebook_size"]),
        commitment_weight=float(rq_config["commitment_weight"]),
        use_ema=rq_config.get("use_ema", False),
        ema_decay=float(rq_config.get("ema_decay", 0.99)),
        ema_epsilon=float(rq_config.get("ema_epsilon", 1e-5)),
        dead_code_threshold=float(rq_config.get("dead_code_threshold", 2.0)),
        normalize_reconstruction=rq_config.get("normalize_reconstruction", True),
    ).to(device)
    pretrained_artifact_dir = rq_config.get("pretrained_artifact_dir")
    if pretrained_artifact_dir:
        pretrained_path = Path(pretrained_artifact_dir)
        rqvae.load_state_dict(
            torch.load(pretrained_path / "rqvae.pt", map_location=device, weights_only=True)
        )
        catalog = SemanticIdCatalog.load(pretrained_path / "catalog.json")
        if catalog.item_ids != item_ids:
            raise ValueError("pretrained Semantic ID catalog does not match embedding item order")
        codebook_diagnostics = json.loads(
            (pretrained_path / "rqvae_diagnostics.json").read_text(encoding="utf-8")
        )
        kmeans_initialization = []
        rqvae_source = str(pretrained_path.resolve())
    else:
        kmeans_initialization = train_rqvae(
            rqvae,
            item_embeddings,
            device=device,
            epochs=rq_config["epochs"],
            batch_size=rq_config["batch_size"],
            learning_rate=rq_config["learning_rate"],
            kmeans_init=rq_config.get("kmeans_init", True),
            kmeans_iters=rq_config.get("kmeans_iters", 20),
            kmeans_sample_size=rq_config.get("kmeans_sample_size", 20000),
            seed=config["seed"],
        )
        catalog, codes = encode_catalog(
            rqvae,
            item_ids,
            item_embeddings,
            device=device,
            batch_size=rq_config["batch_size"],
            codebook_size=rq_config["codebook_size"],
        )
        codebook_diagnostics = semantic_id_diagnostics(
            codes, codebook_size=rq_config["codebook_size"]
        )
        rqvae_source = "trained_in_run"
    print(json.dumps({"rqvae_diagnostics": codebook_diagnostics}, ensure_ascii=False, indent=2))
    torch.save(rqvae.state_dict(), output_dir / "rqvae.pt")
    catalog.save(output_dir / "catalog.json")
    with (output_dir / "rqvae_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(codebook_diagnostics, handle, ensure_ascii=False, indent=2)
    quality_gate = config.get("quality_gate", {})
    enforce_codebook_quality(
        codebook_diagnostics,
        min_level_usage_ratio=float(quality_gate.get("min_level_usage_ratio", 0.5)),
        min_unique_tuple_ratio=float(quality_gate.get("min_unique_tuple_ratio", 0.5)),
    )

    generator_config = config["generator"]
    if args.resume_from is not None:
        generator_config["resume"] = False
        generator_config["resume_from"] = args.resume_from
    train_subset_strategy = str(data_config.get("train_subset_strategy", "prefix"))
    train_subset_seed = int(data_config.get("train_subset_seed", config["seed"]))
    frequency_examples = load_sequence_examples_jsonl(
        data_config["train_path"],
        limit=data_config.get("max_train_examples"),
        selection_strategy=train_subset_strategy,
        seed=train_subset_seed,
    )
    train_frequencies: Counter[str] = Counter(
        example.target_item_id for example in frequency_examples
    )
    del frequency_examples
    long_tail_config = config.get("long_tail_training", {})
    max_long_tail_weight = float(long_tail_config.get("max_weight", 5.0))
    sampling_alpha = float(long_tail_config.get("sampling_alpha", 0.0))
    item_loss_alpha = float(long_tail_config.get("item_loss_alpha", 0.0))
    token_loss_alpha = float(long_tail_config.get("token_loss_alpha", 0.0))
    sampling_item_weights = (
        inverse_frequency_item_weights(
            train_frequencies,
            alpha=sampling_alpha,
            max_weight=max_long_tail_weight,
        )
        if sampling_alpha > 0
        else None
    )
    item_loss_weights = (
        inverse_frequency_item_weights(
            train_frequencies,
            alpha=item_loss_alpha,
            max_weight=max_long_tail_weight,
        )
        if item_loss_alpha > 0
        else None
    )
    token_loss_weights = (
        semantic_token_loss_weights(
            catalog,
            train_frequencies,
            alpha=token_loss_alpha,
            max_weight=max_long_tail_weight,
        ).to(device)
        if token_loss_alpha > 0
        else None
    )
    head_items, tail_items = popularity_partition(
        catalog.item_ids,
        train_frequencies,
        head_fraction=float(long_tail_config.get("head_fraction", 0.2)),
    )
    train_loader, train_size = make_loader(
        data_config["train_path"],
        catalog,
        batch_size=generator_config["batch_size"],
        limit=data_config.get("max_train_examples"),
        shuffle=True,
        item_loss_weights=item_loss_weights,
        sampling_item_weights=sampling_item_weights,
        selection_strategy=train_subset_strategy,
        selection_seed=train_subset_seed,
    )
    requested_evaluation_splits = {
        str(split)
        for split in config.get("evaluation", {}).get(
            "splits", ["validation", "test"]
        )
    }
    if not requested_evaluation_splits or not requested_evaluation_splits.issubset(
        {"validation", "test"}
    ):
        raise ValueError("evaluation.splits must contain validation and/or test")
    valid_loader: DataLoader | None = None
    valid_size: int | None = None
    if "validation" in requested_evaluation_splits:
        valid_loader, valid_size = make_loader(
            data_config["valid_path"],
            catalog,
            batch_size=generator_config["batch_size"],
            limit=data_config.get("max_eval_examples"),
            shuffle=False,
        )
    test_loader: DataLoader | None = None
    test_size: int | None = None
    if "test" in requested_evaluation_splits:
        test_loader, test_size = make_loader(
            data_config["test_path"],
            catalog,
            batch_size=generator_config["batch_size"],
            limit=data_config.get("max_eval_examples"),
            shuffle=False,
        )
    model = GenerativeRecommender(
        vocab_size=catalog.vocab_size,
        pad_token_id=catalog.special_tokens.pad,
        num_behaviors=data_config["num_behaviors"],
        num_time_buckets=data_config["num_time_buckets"],
        model_dim=generator_config["model_dim"],
        num_heads=generator_config["num_heads"],
        num_encoder_layers=generator_config["num_encoder_layers"],
        num_decoder_layers=generator_config["num_decoder_layers"],
        feedforward_dim=generator_config["feedforward_dim"],
        dropout=generator_config["dropout"],
        short_term_window=generator_config["short_term_window"],
        max_history_length=data_config["max_history"],
        max_target_length=catalog.sequence_length + 2,
        use_behavior_intensity=generator_config.get("use_behavior_intensity", False),
        preference_reasoning_mode=generator_config.get(
            "preference_reasoning_mode", "full"
        ),
        history_forgetting_mode=generator_config.get("history_forgetting_mode", "none"),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=generator_config["learning_rate"])
    loss_type = str(generator_config.get("loss_type", "global_cross_entropy"))
    raw_position_weights = generator_config.get("position_loss_weights")
    position_weights = (
        torch.tensor(raw_position_weights, dtype=torch.float32, device=device)
        if raw_position_weights is not None
        else None
    )
    start_epoch = 0
    resume_rng_restored = False
    generator_history: list[dict[str, float | int]] = []
    resume_path: Path | None = None
    if generator_config.get("resume", False):
        resume_path = output_dir / "generator_checkpoint.pt"
    elif generator_config.get("resume_from"):
        resume_path = Path(generator_config["resume_from"])
    if resume_path is not None and resume_path.is_file():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        generator_history = list(checkpoint.get("history", []))
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            resume_rng_restored = True
        print(
            f"resume generator checkpoint={resume_path} completed_epochs={start_epoch} "
            f"rng_restored={resume_rng_restored}"
        )
    for epoch in range(start_epoch, generator_config["epochs"]):
        model.train()
        losses: list[float] = []
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            logits, _ = model(
                batch["history_tokens"],
                batch["behavior_types"],
                batch["time_gap_buckets"],
                batch["history_mask"],
                batch["target_input"],
                batch["behavior_intensities"],
            )
            loss = generator_loss(
                model,
                logits,
                batch,
                catalog,
                loss_type=loss_type,
                position_weights=position_weights,
                token_weights=token_loss_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            log_every = int(generator_config.get("log_every_batches", 250))
            if batch_index % log_every == 0:
                print(
                    f"generator epoch={epoch + 1} batch={batch_index}/{len(train_loader)} "
                    f"loss={sum(losses[-log_every:]) / log_every:.4f}"
                )
        epoch_loss = sum(losses) / len(losses)
        generator_history.append({"epoch": epoch + 1, "loss": epoch_loss})
        print(f"generator epoch={epoch + 1} loss={epoch_loss:.4f}")
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "history": generator_history,
            },
            output_dir / "generator_checkpoint.pt",
        )

    torch.save(rqvae.state_dict(), output_dir / "rqvae.pt")
    torch.save(model.state_dict(), output_dir / "generator.pt")
    catalog.save(output_dir / "catalog.json")
    evaluation_config = config.get("evaluation", {})
    ks = tuple(int(k) for k in evaluation_config.get("ks", [5, 10]))
    beam_size = int(evaluation_config.get("beam_size", max(ks)))
    num_return_sequences = int(evaluation_config.get("num_return_sequences", max(ks)))
    candidate_count = int(evaluation_config.get("candidate_count", num_return_sequences))
    diversity_weight = float(evaluation_config.get("diversity_weight", 0.0))
    renormalize_allowed_tokens = bool(
        evaluation_config.get("renormalize_allowed_tokens", False)
    )
    evaluation_splits = requested_evaluation_splits
    if (
        beam_size < candidate_count
        or candidate_count < num_return_sequences
        or num_return_sequences < max(ks)
    ):
        raise ValueError(
            "evaluation requires beam_size >= candidate_count >= "
            "num_return_sequences >= max(evaluation.ks)"
        )
    if not 0.0 <= diversity_weight < 1.0:
        raise ValueError("evaluation.diversity_weight must be in [0, 1)")
    long_tail_summary = {
        "sampling_alpha": sampling_alpha,
        "item_loss_alpha": item_loss_alpha,
        "token_loss_alpha": token_loss_alpha,
        "max_weight": max_long_tail_weight,
        "head_fraction": float(long_tail_config.get("head_fraction", 0.2)),
        "observed_training_target_items": len(train_frequencies),
        "sampling_weight_range": (
            [
                min(sampling_item_weights.values()),
                max(sampling_item_weights.values()),
            ]
            if sampling_item_weights
            else [1.0, 1.0]
        ),
        "item_loss_weight_range": (
            [min(item_loss_weights.values()), max(item_loss_weights.values())]
            if item_loss_weights
            else [1.0, 1.0]
        ),
        "token_loss_weight_range": (
            [
                float(token_loss_weights.min().item()),
                float(token_loss_weights.max().item()),
            ]
            if token_loss_weights is not None
            else [1.0, 1.0]
        ),
    }
    report = {
        "seed": config["seed"],
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "category": data_config["category"],
        "data_audit": data_audit,
        "embedding": embedding_metadata,
        "items": len(item_ids),
        "examples": {"train": train_size, "valid": valid_size, "test": test_size},
        "training_subset": {
            "strategy": train_subset_strategy,
            "selection_seed": train_subset_seed,
            "limit": data_config.get("max_train_examples"),
        },
        "semantic_id_length": catalog.sequence_length,
        "vocab_size": catalog.vocab_size,
        "rqvae_diagnostics": codebook_diagnostics,
        "rqvae_config": rq_config,
        "rqvae_source": rqvae_source,
        "generator_config": generator_config,
        "long_tail_training": long_tail_summary,
        "generator_resume": {
            "checkpoint": str(resume_path.resolve()) if resume_path is not None else None,
            "completed_epochs_before_run": start_epoch,
            "rng_restored": resume_rng_restored,
        },
        "generator_history": generator_history,
        "kmeans_initialization": kmeans_initialization,
        "evaluation": {
            "decoder": "catalog_trie_beam_search",
            "beam_size": beam_size,
            "num_return_sequences": num_return_sequences,
            "candidate_count": candidate_count,
            "renormalize_allowed_tokens": renormalize_allowed_tokens,
            "diversity_method": "semantic_id_mmr",
            "diversity_weight": diversity_weight,
        },
    }
    if "validation" in evaluation_splits:
        assert valid_loader is not None
        report["validation"] = evaluate(
            model,
            valid_loader,
            device=device,
            catalog=catalog,
            beam_size=beam_size,
            num_return_sequences=num_return_sequences,
            candidate_count=candidate_count,
            renormalize_allowed_tokens=renormalize_allowed_tokens,
            diversity_weight=diversity_weight,
            loss_type=loss_type,
            position_weights=position_weights,
            ks=ks,
            split_name="validation",
            item_frequencies=train_frequencies,
            head_items=head_items,
            tail_items=tail_items,
        )
    if "test" in evaluation_splits:
        assert test_loader is not None
        report["test"] = evaluate(
            model,
            test_loader,
            device=device,
            catalog=catalog,
            beam_size=beam_size,
            num_return_sequences=num_return_sequences,
            candidate_count=candidate_count,
            renormalize_allowed_tokens=renormalize_allowed_tokens,
            diversity_weight=diversity_weight,
            loss_type=loss_type,
            position_weights=position_weights,
            ks=ks,
            split_name="test",
            item_frequencies=train_frequencies,
            head_items=head_items,
            tail_items=tail_items,
        )
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
