from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from regenrec.data import RecommendationDataset, SequenceExample, load_sequence_examples_jsonl
from regenrec.metrics import ranking_metrics
from regenrec.models.sasrec import SASRec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SASRec on the Amazon V1 protocol")
    parser.add_argument("--config", default="configs/baseline_sasrec_amazon2023.yaml")
    parser.add_argument("--seed", type=int, help="Override the configured random seed")
    parser.add_argument("--output-dir", help="Override output_dir for an independent run")
    return parser.parse_args()


class SASRecCollator:
    def __init__(self, item_to_index: dict[str, int]) -> None:
        self.item_to_index = item_to_index

    def __call__(self, examples: list[SequenceExample]) -> dict[str, torch.Tensor]:
        max_history = max(len(example.history) for example in examples)
        history_items = torch.zeros((len(examples), max_history), dtype=torch.long)
        history_mask = torch.zeros_like(history_items, dtype=torch.bool)
        targets = torch.empty(len(examples), dtype=torch.long)
        for batch_index, example in enumerate(examples):
            for history_index, event in enumerate(example.history):
                history_items[batch_index, history_index] = self.item_to_index[event.item_id] + 1
                history_mask[batch_index, history_index] = True
            targets[batch_index] = self.item_to_index[example.target_item_id]
        return {
            "history_items": history_items,
            "history_mask": history_mask,
            "targets": targets,
        }


def make_loader(
    path: str | Path,
    *,
    collator: SASRecCollator,
    batch_size: int,
    limit: int | None,
    shuffle: bool,
) -> tuple[DataLoader, int]:
    examples = load_sequence_examples_jsonl(path, limit=limit)
    return (
        DataLoader(
            RecommendationDataset(examples),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
        ),
        len(examples),
    )


@torch.no_grad()
def evaluate(
    model: SASRec,
    loader: DataLoader,
    item_ids: list[str],
    *,
    ks: tuple[int, ...],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions: list[list[str]] = []
    targets: list[str] = []
    max_k = max(ks)
    for batch in loader:
        history_items = batch["history_items"].to(device)
        history_mask = batch["history_mask"].to(device)
        target_indices = batch["targets"].to(device)
        logits = model(history_items, history_mask)
        total_loss += float(model.loss(logits, target_indices).item()) * target_indices.size(0)
        top_indices = logits.topk(max_k, dim=-1).indices.cpu().tolist()
        predictions.extend([[item_ids[index] for index in row] for row in top_indices])
        targets.extend(item_ids[index] for index in target_indices.cpu().tolist())
        total_examples += target_indices.size(0)
    return {
        "loss": total_loss / total_examples,
        **ranking_metrics(predictions, targets, ks=ks),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    data_config = config["data"]
    audit_path = Path(data_config["data_audit_path"])
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    if audit.get("status") != "passed" or audit.get("synthetic_data_used") is not False:
        raise ValueError("SASRec baseline requires a passed real-data audit")
    catalog = SemanticIdCatalog.load(data_config["catalog_path"])
    item_ids = catalog.item_ids
    if len(item_ids) != int(audit["checks"]["actual_unique_items"]):
        raise ValueError("catalog item count does not match the real-data audit")
    collator = SASRecCollator(catalog.item_to_index)
    model_config = config["model"]
    train_loader, train_size = make_loader(
        data_config["train_path"],
        collator=collator,
        batch_size=model_config["batch_size"],
        limit=data_config.get("max_train_examples"),
        shuffle=True,
    )
    valid_loader, valid_size = make_loader(
        data_config["valid_path"],
        collator=collator,
        batch_size=model_config["batch_size"],
        limit=data_config.get("max_eval_examples"),
        shuffle=False,
    )
    test_loader, test_size = make_loader(
        data_config["test_path"],
        collator=collator,
        batch_size=model_config["batch_size"],
        limit=data_config.get("max_eval_examples"),
        shuffle=False,
    )
    model = SASRec(
        num_items=len(item_ids),
        model_dim=model_config["model_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        feedforward_dim=model_config["feedforward_dim"],
        dropout=model_config["dropout"],
        max_history_length=data_config["max_history"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config["learning_rate"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    start_epoch = 0
    history: list[dict[str, float | int]] = []
    rng_restored = False
    if model_config.get("resume", False) and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint.get("history", []))
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            rng_restored = True
        print(f"resume SASRec completed_epochs={start_epoch} rng_restored={rng_restored}")

    for epoch in range(start_epoch, model_config["epochs"]):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            history_items = batch["history_items"].to(device)
            history_mask = batch["history_mask"].to(device)
            targets = batch["targets"].to(device)
            loss = model.loss(model(history_items, history_mask), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
            if batch_index % int(model_config.get("log_every_batches", 250)) == 0:
                print(
                    f"sasrec epoch={epoch + 1} batch={batch_index}/{len(train_loader)} "
                    f"loss={total_loss / batches:.4f}"
                )
        epoch_loss = total_loss / batches
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "torch_rng_state": torch.get_rng_state(),
            },
            checkpoint_path,
        )
        print(f"sasrec epoch={epoch + 1} loss={epoch_loss:.4f}")

    torch.save(model.state_dict(), output_dir / "sasrec.pt")
    ks = tuple(int(value) for value in config["evaluation"]["ks"])
    report = {
        "seed": config["seed"],
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "category": data_config["category"],
        "synthetic_data_used": False,
        "data_audit": {
            "status": audit["status"],
            "path": str(audit_path.resolve()),
            "sha256": hashlib.sha256(audit_bytes).hexdigest(),
        },
        "protocol": "same leave-last-out splits; history items are not filtered",
        "items": len(item_ids),
        "examples": {"train": train_size, "valid": valid_size, "test": test_size},
        "model_config": model_config,
        "training_history": history,
        "resume": {
            "completed_epochs_before_run": start_epoch,
            "rng_restored": rng_restored,
        },
        "validation": evaluate(model, valid_loader, item_ids, ks=ks, device=device),
        "test": evaluate(model, test_loader, item_ids, ks=ks, device=device),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
