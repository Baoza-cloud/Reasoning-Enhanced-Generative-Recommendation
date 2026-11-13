from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from regenrec.models.generative import GenerativeRecommender
from train_amazon2023 import evaluate, make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare trie beam widths and probability normalization on a saved model"
    )
    parser.add_argument("--config", default="configs/v2_amazon2023.yaml")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/amazon2023_musical_instruments_blair_v2_intensity"),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-examples", type=int, default=2048)
    parser.add_argument("--beam-size", action="append", type=int, required=True)
    parser.add_argument(
        "--normalization",
        action="append",
        choices=("global", "allowed"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_examples <= 0:
        raise ValueError("max-examples must be positive")
    config = load_config(args.config)
    data_config = config["data"]
    audit_path = Path(data_config["data_audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("synthetic_data_used") is not False:
        raise ValueError("beam diagnostics require a passed real-data audit")

    catalog = SemanticIdCatalog.load(args.artifact_dir / "catalog.json")
    generator_config = config["generator"]
    device = torch.device(config["device"])
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
    model_path = args.artifact_dir / "generator.pt"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    split_path = data_config["valid_path"] if args.split == "validation" else data_config["test_path"]
    loader, examples = make_loader(
        split_path,
        catalog,
        batch_size=generator_config["batch_size"],
        limit=args.max_examples,
        shuffle=False,
    )
    ks = tuple(int(value) for value in config["evaluation"]["ks"])
    results: list[dict[str, object]] = []
    for normalization in args.normalization:
        for beam_size in args.beam_size:
            if beam_size < max(ks):
                raise ValueError("beam-size must be at least the maximum requested K")
            metrics = evaluate(
                model,
                loader,
                device=device,
                catalog=catalog,
                beam_size=beam_size,
                num_return_sequences=max(ks),
                renormalize_allowed_tokens=normalization == "allowed",
                loss_type="global_cross_entropy",
                position_weights=None,
                ks=ks,
                split_name=f"{args.split}_{normalization}_beam{beam_size}",
            )
            results.append(
                {"normalization": normalization, "beam_size": beam_size, **metrics}
            )

    report = {
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "synthetic_data_used": False,
        "data_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "split": args.split,
        "examples": examples,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
