from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from regenrec.data import RecommendationDataset, SequenceCollator
from regenrec.models.generative import GenerativeRecommender
from regenrec.models.rqvae import RQVAE
from regenrec.synthetic import make_synthetic_catalog, make_synthetic_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny end-to-end V1 training job")
    parser.add_argument("--config", default="configs/v1_smoke.yaml")
    return parser.parse_args()


def train_rqvae(
    model: RQVAE,
    item_embeddings: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        output = model(item_embeddings)
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        optimizer.step()
        print(
            f"rqvae epoch={epoch + 1} loss={output.loss.item():.4f} "
            f"reconstruction={output.reconstruction_loss.item():.4f}"
        )


def train_generator(
    model: GenerativeRecommender,
    loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        losses: list[float] = []
        for batch in loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            logits, _ = model(
                batch["history_tokens"],
                batch["behavior_types"],
                batch["time_gap_buckets"],
                batch["history_mask"],
                batch["target_input"],
            )
            loss = model.loss(logits, batch["target_labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        mean_loss = sum(losses) / len(losses)
        print(f"generator epoch={epoch + 1} loss={mean_loss:.4f}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])

    data_config = config["data"]
    item_ids, item_embeddings, categories = make_synthetic_catalog(
        num_items=data_config["num_items"],
        embedding_dim=data_config["item_embedding_dim"],
        seed=config["seed"],
    )
    item_embeddings = item_embeddings.to(device)

    rq_config = config["rqvae"]
    rqvae = RQVAE(
        input_dim=data_config["item_embedding_dim"],
        hidden_dim=rq_config["hidden_dim"],
        code_dim=rq_config["code_dim"],
        num_levels=rq_config["num_levels"],
        codebook_size=rq_config["codebook_size"],
        commitment_weight=rq_config["commitment_weight"],
    ).to(device)
    train_rqvae(
        rqvae,
        item_embeddings,
        epochs=rq_config["epochs"],
        learning_rate=rq_config["learning_rate"],
    )

    codes = rqvae.encode_codes(item_embeddings).cpu()
    catalog = SemanticIdCatalog.from_codes(
        item_ids, codes, codebook_size=rq_config["codebook_size"]
    )
    examples = make_synthetic_sequences(
        item_ids=item_ids,
        categories=categories,
        num_users=data_config["num_users"],
        min_history=data_config["min_history"],
        max_history=data_config["max_history"],
        num_behaviors=data_config["num_behaviors"],
        num_time_buckets=data_config["num_time_buckets"],
        seed=config["seed"],
    )
    dataset = RecommendationDataset(examples)
    generator_config = config["generator"]
    loader = DataLoader(
        dataset,
        batch_size=generator_config["batch_size"],
        shuffle=True,
        collate_fn=SequenceCollator(catalog),
    )

    recommender = GenerativeRecommender(
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
    ).to(device)
    train_generator(
        recommender,
        loader,
        device=device,
        epochs=generator_config["epochs"],
        learning_rate=generator_config["learning_rate"],
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(rqvae.state_dict(), output_dir / "rqvae.pt")
    torch.save(recommender.state_dict(), output_dir / "generator.pt")
    catalog.save(output_dir / "catalog.json")

    sample_batch = next(iter(loader))
    sample_batch = {name: tensor.to(device) for name, tensor in sample_batch.items()}
    item_predictions, token_predictions = recommender.generate(
        history_tokens=sample_batch["history_tokens"],
        behavior_types=sample_batch["behavior_types"],
        time_gap_buckets=sample_batch["time_gap_buckets"],
        history_mask=sample_batch["history_mask"],
        catalog=catalog,
    )
    report = {
        "codebook_statistics": rqvae.codebook_statistics(item_embeddings),
        "catalog_items": len(catalog.item_ids),
        "semantic_id_length": catalog.sequence_length,
        "vocab_size": catalog.vocab_size,
        "sample_predictions": item_predictions,
        "sample_token_predictions": token_predictions.cpu().tolist(),
        "valid_prediction_rate": sum(item is not None for item in item_predictions)
        / len(item_predictions),
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
