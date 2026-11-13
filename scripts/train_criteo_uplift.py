from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from regenrec.causal import doubly_robust_pseudo_outcome
from regenrec.config import load_config
from regenrec.datasets.criteo_uplift import CRITEO_FEATURES, load_criteo_uplift_tensors
from regenrec.metrics import uplift_ranking_metrics
from regenrec.models.uplift import (
    TwoHeadOutcomeModel,
    UpliftEffectModel,
    inverse_propensity_outcome_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DR uplift learner on Criteo RCT data")
    parser.add_argument("--config", default="configs/v3_criteo_uplift_smoke.yaml")
    return parser.parse_args()


def deterministic_split_indices(num_rows: int, seed: int) -> dict[str, torch.Tensor]:
    row_indices = torch.arange(num_rows, dtype=torch.int64)
    buckets = ((row_indices * 1_103_515_245 + seed) % 2_147_483_647) % 10
    return {
        "train": row_indices[buckets < 8],
        "validation": row_indices[buckets == 8],
        "test": row_indices[buckets == 9],
    }


def normalization_statistics(
    features: torch.Tensor,
    indices: torch.Tensor,
    *,
    chunk_size: int = 200_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(features.size(1), dtype=torch.float64)
    squared_total = torch.zeros_like(total)
    count = 0
    for start in range(0, indices.numel(), chunk_size):
        values = features[indices[start : start + chunk_size]].to(torch.float64)
        total += values.sum(dim=0)
        squared_total += values.square().sum(dim=0)
        count += values.size(0)
    mean = total / count
    variance = (squared_total / count - mean.square()).clamp_min(1e-12)
    return mean.to(torch.float32), variance.sqrt().to(torch.float32)


def make_index_loader(
    indices: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(TensorDataset(indices), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def outcome_validation_loss(
    model: TwoHeadOutcomeModel,
    loader: DataLoader,
    features: torch.Tensor,
    treatment: torch.Tensor,
    outcome: torch.Tensor,
    *,
    propensity_value: float,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for (indices,) in loader:
        batch_features = features[indices].to(device)
        batch_treatment = treatment[indices].to(device)
        batch_outcome = outcome[indices].to(device)
        logits = model.factual_logits(batch_features, batch_treatment)
        loss = F.binary_cross_entropy_with_logits(logits, batch_outcome, reduction="sum")
        total_loss += float(loss.item())
        total_examples += indices.numel()
    return total_loss / total_examples


@torch.no_grad()
def evaluate_uplift(
    model: UpliftEffectModel,
    loader: DataLoader,
    features: torch.Tensor,
    treatment: torch.Tensor,
    outcome: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    scores: list[float] = []
    treatments: list[int] = []
    outcomes: list[int] = []
    for (indices,) in loader:
        scores.extend(model(features[indices].to(device)).cpu().tolist())
        treatments.extend(treatment[indices].to(torch.int64).tolist())
        outcomes.extend(outcome[indices].to(torch.int64).tolist())
    return uplift_ranking_metrics(scores, treatments, outcomes, top_fraction=0.1)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    data_config = config["data"]
    audit = json.loads(Path(data_config["audit_path"]).read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("synthetic_data_used") is not False:
        raise ValueError("Criteo training requires a passed real-data audit")
    max_rows = data_config.get("max_rows")
    features, treatment, outcome = load_criteo_uplift_tensors(
        data_config["path"],
        expected_rows=int(audit["statistics"]["rows"]),
        max_rows=max_rows,
        sample_seed=config["seed"],
    )
    splits = deterministic_split_indices(features.size(0), config["seed"])
    mean, std = normalization_statistics(features, splits["train"])
    features.sub_(mean).div_(std)

    training_config = config["training"]
    loaders = {
        name: make_index_loader(
            indices,
            batch_size=training_config["batch_size"],
            shuffle=name == "train",
        )
        for name, indices in splits.items()
    }
    train_treatment = treatment[splits["train"]]
    propensity_value = float(train_treatment.mean().item())
    if not 0.01 <= propensity_value <= 0.99:
        raise ValueError("observed randomized treatment rate violates overlap bounds")

    hidden_dims = tuple(int(value) for value in training_config["hidden_dims"])
    outcome_model = TwoHeadOutcomeModel(len(CRITEO_FEATURES), hidden_dims).to(device)
    outcome_optimizer = torch.optim.AdamW(
        outcome_model.parameters(), lr=training_config["learning_rate"]
    )
    effect_model = UpliftEffectModel(len(CRITEO_FEATURES), hidden_dims).to(device)
    effect_optimizer = torch.optim.AdamW(
        effect_model.parameters(), lr=training_config["learning_rate"]
    )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "training_checkpoint.pt"
    start_outcome_epoch = 0
    start_effect_epoch = 0
    outcome_history: list[dict[str, float | int]] = []
    effect_history: list[dict[str, float | int]] = []
    resume_rng_restored = False
    if training_config.get("resume", False) and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        outcome_model.load_state_dict(checkpoint["outcome_model"])
        effect_model.load_state_dict(checkpoint["effect_model"])
        outcome_optimizer.load_state_dict(checkpoint["outcome_optimizer"])
        effect_optimizer.load_state_dict(checkpoint["effect_optimizer"])
        start_outcome_epoch = int(checkpoint["outcome_epoch"])
        start_effect_epoch = int(checkpoint["effect_epoch"])
        outcome_history = list(checkpoint.get("outcome_history", []))
        effect_history = list(checkpoint.get("effect_history", []))
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            resume_rng_restored = True
        print(
            f"resume uplift checkpoint outcome_epochs={start_outcome_epoch} "
            f"effect_epochs={start_effect_epoch} rng_restored={resume_rng_restored}"
        )

    def save_checkpoint(outcome_epoch: int, effect_epoch: int) -> None:
        torch.save(
            {
                "outcome_epoch": outcome_epoch,
                "effect_epoch": effect_epoch,
                "outcome_model": outcome_model.state_dict(),
                "effect_model": effect_model.state_dict(),
                "outcome_optimizer": outcome_optimizer.state_dict(),
                "effect_optimizer": effect_optimizer.state_dict(),
                "outcome_history": outcome_history,
                "effect_history": effect_history,
                "torch_rng_state": torch.get_rng_state(),
            },
            checkpoint_path,
        )

    for epoch in range(start_outcome_epoch, training_config["outcome_epochs"]):
        outcome_model.train()
        total_loss = 0.0
        batches = 0
        for (indices,) in loaders["train"]:
            batch_features = features[indices].to(device)
            batch_treatment = treatment[indices].to(device)
            batch_outcome = outcome[indices].to(device)
            propensity = torch.full_like(batch_treatment, propensity_value)
            loss = inverse_propensity_outcome_loss(
                outcome_model,
                batch_features,
                batch_treatment,
                batch_outcome,
                propensity,
            )
            outcome_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            outcome_optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        validation_loss = outcome_validation_loss(
            outcome_model,
            loaders["validation"],
            features,
            treatment,
            outcome,
            propensity_value=propensity_value,
            device=device,
        )
        epoch_report = {
            "epoch": epoch + 1,
            "train_loss": total_loss / batches,
            "validation_factual_log_loss": validation_loss,
        }
        outcome_history.append(epoch_report)
        print(json.dumps({"outcome_model": epoch_report}))
        save_checkpoint(epoch + 1, start_effect_epoch)

    outcome_model.eval()
    for epoch in range(start_effect_epoch, training_config["effect_epochs"]):
        effect_model.train()
        total_loss = 0.0
        batches = 0
        for (indices,) in loaders["train"]:
            batch_features = features[indices].to(device)
            batch_treatment = treatment[indices].to(device)
            batch_outcome = outcome[indices].to(device)
            propensity = torch.full_like(batch_treatment, propensity_value)
            with torch.no_grad():
                mu0, mu1 = outcome_model.potential_outcomes(batch_features)
                pseudo_outcome = doubly_robust_pseudo_outcome(
                    batch_outcome,
                    batch_treatment,
                    propensity,
                    mu0,
                    mu1,
                )
            loss = F.smooth_l1_loss(effect_model(batch_features), pseudo_outcome)
            effect_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            effect_optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        epoch_report = {"epoch": epoch + 1, "train_loss": total_loss / batches}
        effect_history.append(epoch_report)
        print(json.dumps({"effect_model": epoch_report}))
        save_checkpoint(training_config["outcome_epochs"], epoch + 1)

    torch.save(outcome_model.state_dict(), output_dir / "outcome_model.pt")
    torch.save(effect_model.state_dict(), output_dir / "effect_model.pt")
    torch.save({"mean": mean, "std": std}, output_dir / "normalization.pt")
    report = {
        "dataset": audit["source_dataset"],
        "synthetic_data_used": False,
        "source_sha256": audit["source_sha256"],
        "rows_loaded": features.size(0),
        "row_sampling": (
            "full dataset"
            if max_rows is None
            else f"uniform reservoir sample over all audited rows, seed={config['seed']}"
        ),
        "split": {name: indices.numel() for name, indices in splits.items()},
        "split_rule": "deterministic integer hash into 80/10/10 buckets",
        "observed_train_treatment_rate": propensity_value,
        "resume": {
            "checkpoint": str(checkpoint_path.resolve()),
            "outcome_epochs_before_run": start_outcome_epoch,
            "effect_epochs_before_run": start_effect_epoch,
            "rng_restored": resume_rng_restored,
        },
        "outcome_history": outcome_history,
        "effect_history": effect_history,
        "validation": evaluate_uplift(
            effect_model,
            loaders["validation"],
            features,
            treatment,
            outcome,
            device=device,
        ),
        "test": evaluate_uplift(
            effect_model,
            loaders["test"],
            features,
            treatment,
            outcome,
            device=device,
        ),
        "interpretation_guardrail": audit["identifiability_note"],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
