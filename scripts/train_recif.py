from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from regenrec.data import RecommendationDataset, SequenceCollator, load_sequence_examples_jsonl
from regenrec.models.generative import GenerativeRecommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V1 on prepared RecIF product data")
    parser.add_argument("--config", default="configs/v1_recif.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])

    data_config = config["data"]
    catalog = SemanticIdCatalog.load(data_config["catalog_path"])
    examples = load_sequence_examples_jsonl(
        data_config["examples_path"], limit=data_config.get("max_examples")
    )
    loader = DataLoader(
        RecommendationDataset(examples),
        batch_size=config["generator"]["batch_size"],
        shuffle=True,
        collate_fn=SequenceCollator(catalog),
    )

    generator_config = config["generator"]
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

    model.train()
    for epoch in range(generator_config["epochs"]):
        total_loss = 0.0
        batches = 0
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
            total_loss += loss.item()
            batches += 1
        print(f"epoch={epoch + 1} loss={total_loss / max(1, batches):.4f}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "generator.pt")
    report = {
        "dataset": "OpenOneRec/OpenOneRec-RecIF",
        "domain": "product",
        "examples": len(examples),
        "catalog_items": len(catalog.item_ids),
        "semantic_id_length": catalog.sequence_length,
        "vocab_size": catalog.vocab_size,
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
