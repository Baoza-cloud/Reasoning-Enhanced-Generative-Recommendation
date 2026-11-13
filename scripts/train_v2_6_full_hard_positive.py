from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from scripts.train_v2_5_bounded_residual import (
    FEATURE_NAMES,
    GATE_FEATURE_NAMES,
    build_training_pairs,
    catalog_groups,
    evaluate,
    load_cache,
    make_features,
    paired_tests,
    score_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V2.6 full-Train compact hard-pair bounded residual fit"
    )
    parser.add_argument("--config", default="configs/v2_final_default.yaml")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--census-dir", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-negatives", type=int, default=20)
    parser.add_argument("--max-preserve-pairs-per-shard", type=int, default=6000)
    parser.add_argument("--bound", type=float, default=0.5)
    parser.add_argument("--tail-positive-weight", type=float, default=8.0)
    parser.add_argument("--preserve-weight", type=float, default=4.0)
    parser.add_argument("--distill-weight", type=float, default=0.2)
    parser.add_argument("--gate-supervision-weight", type=float, default=1.0)
    parser.add_argument("--gate-positive-weight", type=float, default=20.0)
    parser.add_argument(
        "--deployment-scales",
        type=float,
        nargs="+",
        default=(0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def compact_training_rows(
    payload: dict[str, object],
    *,
    max_negatives: int,
    max_preserve_pairs: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    candidate, row, base = make_features(payload)
    pairs = build_training_pairs(
        payload, base, candidate[:, :, 0], max_negatives=max_negatives
    )
    rr = pairs["rank_rows"]
    pp = pairs["positives"]
    nn = pairs["negatives"]
    pr = pairs["preserve_rows"]
    pi = pairs["preserve_inside"]
    po = pairs["preserve_outside"]
    if len(pr) > max_preserve_pairs:
        generator = torch.Generator().manual_seed(seed)
        selected = torch.randperm(len(pr), generator=generator)[:max_preserve_pairs]
        pr, pi, po = pr[selected], pi[selected], po[selected]
    positions = payload["target_positions"].long()
    contained = positions >= 0
    safe_positions = positions.clamp_min(0).unsqueeze(1)
    retrieval_only = ~torch.isfinite(
        payload["beam_scores"].gather(1, safe_positions).squeeze(1)
    )
    gate_labels = (contained & retrieval_only).float()
    compact = {
        "rank_positive_features": candidate[rr, pp],
        "rank_negative_features": candidate[rr, nn],
        "rank_row_features": row[rr],
        "rank_base_difference": base[rr, pp] - base[rr, nn],
        "rank_group_ids": pairs["group_ids"],
        "preserve_inside_features": candidate[pr, pi],
        "preserve_outside_features": candidate[pr, po],
        "preserve_row_features": row[pr],
        "preserve_base_difference": base[pr, pi] - base[pr, po],
        "gate_row_features": row,
        "gate_labels": gate_labels,
    }
    stats = {
        "examples": int(base.size(0)),
        "supervised_rows": int(contained.sum()),
        "hard_positive_rows": int(gate_labels.sum()),
        "rank_pairs": len(rr),
        "tail_rank_pairs": int((pairs["group_ids"] == 2).sum()),
        "preserve_pairs": len(pr),
    }
    return compact, stats


def concatenate_compact(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("no compact Train shards were loaded")
    return {name: torch.cat([row[name] for row in rows]) for name in rows[0]}


def residual_pair_difference(
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
    row_features: torch.Tensor,
    base_difference: torch.Tensor,
    residual_weights: torch.Tensor,
    gate_weights: torch.Tensor,
    gate_bias: torch.Tensor,
    *,
    bound: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate = torch.sigmoid(row_features @ gate_weights + gate_bias)
    positive = bound * gate * torch.tanh(positive_features @ residual_weights)
    negative = bound * gate * torch.tanh(negative_features @ residual_weights)
    return base_difference + positive - negative, torch.stack((positive, negative), dim=1)


def train_model(
    train: dict[str, torch.Tensor],
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    bound: float,
    tail_positive_weight: float,
    preserve_weight: float,
    distill_weight: float,
    gate_supervision_weight: float,
    gate_positive_weight: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    torch.manual_seed(seed)
    residual_weights = torch.zeros(len(FEATURE_NAMES), requires_grad=True)
    gate_weights = torch.zeros(len(GATE_FEATURE_NAMES), requires_grad=True)
    gate_bias = torch.tensor(-2.1972246, requires_grad=True)
    optimizer = torch.optim.Adam(
        [residual_weights, gate_weights, gate_bias], lr=learning_rate
    )
    rank_count = len(train["rank_base_difference"])
    preserve_count = len(train["preserve_base_difference"])
    gate_count = len(train["gate_labels"])
    if min(rank_count, preserve_count, gate_count) <= 0:
        raise ValueError("rank, preserve and gate training rows must be non-empty")
    generator = torch.Generator().manual_seed(seed)
    last: dict[str, float] = {}
    for epoch in range(epochs):
        order = torch.randperm(rank_count, generator=generator)
        totals = {"loss": 0.0, "rank": 0.0, "preserve": 0.0, "distill": 0.0, "gate": 0.0}
        steps = 0
        for start in range(0, rank_count, batch_size):
            selected = order[start : start + batch_size]
            pair_delta, residuals = residual_pair_difference(
                train["rank_positive_features"][selected],
                train["rank_negative_features"][selected],
                train["rank_row_features"][selected],
                train["rank_base_difference"][selected],
                residual_weights,
                gate_weights,
                gate_bias,
                bound=bound,
            )
            weights = torch.ones(len(selected))
            groups = train["rank_group_ids"][selected]
            weights[groups == 1] = 2.0
            weights[groups == 2] = tail_positive_weight
            rank_loss = (F.softplus(-pair_delta / 0.10) * weights).sum() / weights.sum()
            preserve_selected = torch.randint(
                preserve_count,
                (min(batch_size, preserve_count),),
                generator=generator,
            )
            preserve_delta, _ = residual_pair_difference(
                train["preserve_inside_features"][preserve_selected],
                train["preserve_outside_features"][preserve_selected],
                train["preserve_row_features"][preserve_selected],
                train["preserve_base_difference"][preserve_selected],
                residual_weights,
                gate_weights,
                gate_bias,
                bound=bound,
            )
            preserve_loss = F.relu(0.01 - preserve_delta).mean()
            distill_loss = residuals.square().mean()
            gate_selected = torch.randint(
                gate_count, (min(batch_size, gate_count),), generator=generator
            )
            gate_logits = (
                train["gate_row_features"][gate_selected] @ gate_weights + gate_bias
            )
            gate_loss = F.binary_cross_entropy_with_logits(
                gate_logits,
                train["gate_labels"][gate_selected],
                pos_weight=torch.tensor(gate_positive_weight),
            )
            parameter_loss = residual_weights.square().mean() + gate_weights.square().mean()
            loss = (
                rank_loss
                + preserve_weight * preserve_loss
                + distill_weight * distill_loss
                + gate_supervision_weight * gate_loss
                + 1e-3 * parameter_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["rank"] += float(rank_loss.detach())
            totals["preserve"] += float(preserve_loss.detach())
            totals["distill"] += float(distill_loss.detach())
            totals["gate"] += float(gate_loss.detach())
            steps += 1
        last = {name: value / steps for name, value in totals.items()}
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(
                f"epoch={epoch + 1}/{epochs} loss={last['loss']:.6f} "
                f"rank={last['rank']:.6f} preserve={last['preserve']:.6f}",
                flush=True,
            )
    return {
        "residual_weights": residual_weights.detach(),
        "gate_weights": gate_weights.detach(),
        "gate_bias": gate_bias.detach(),
    }, last


def serializable_model(model: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        "residual_weights": model["residual_weights"].tolist(),
        "gate_weights": model["gate_weights"].tolist(),
        "gate_bias": float(model["gate_bias"]),
    }


def tail_discordance(
    baseline_hits: torch.Tensor,
    candidate_hits: torch.Tensor,
    target_groups: torch.Tensor,
) -> dict[str, int]:
    selected = target_groups == 2
    baseline = baseline_hits[selected]
    candidate = candidate_hits[selected]
    return {
        "wins": int((candidate & ~baseline).sum()),
        "losses": int((~candidate & baseline).sum()),
        "net_wins": int(candidate.sum() - baseline.sum()),
    }


def main() -> None:
    args = parse_args()
    if min(
        args.epochs,
        args.batch_size,
        args.max_negatives,
        args.max_preserve_pairs_per_shard,
        args.bootstrap_samples,
        args.torch_threads,
    ) <= 0:
        raise ValueError("epoch, batch, pair, bootstrap and thread settings must be positive")
    if (
        args.learning_rate <= 0
        or args.bound <= 0
        or args.tail_positive_weight <= 0
        or args.preserve_weight < 0
        or args.distill_weight < 0
        or args.gate_supervision_weight < 0
        or args.gate_positive_weight <= 0
        or any(not 0 < value <= 1 for value in args.deployment_scales)
    ):
        raise ValueError("invalid V2.6 optimization settings")
    torch.set_num_threads(args.torch_threads)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    try:
        census_report_path = args.census_dir / "report.json"
        census = json.loads(census_report_path.read_text(encoding="utf-8"))
        if census.get("status") != "completed" or census.get("test_evaluated") is not False:
            raise ValueError("V2.6 census is incomplete or has an invalid Test flag")
        if not census.get("go_no_go", {}).get("passed"):
            write_json(
                state_path,
                {
                    "status": "blocked",
                    "reason": "full-Train hard-positive census did not pass",
                    "test_evaluated": False,
                    "census_report": str(census_report_path.resolve()),
                },
            )
            return
        write_json(
            state_path,
            {
                "status": "running",
                "phase": "compact_full_train_pairs",
                "seed": args.seed,
                "test_evaluated": False,
            },
        )
        compact_shards: list[dict[str, torch.Tensor]] = []
        train_stats = {
            "examples": 0,
            "supervised_rows": 0,
            "hard_positive_rows": 0,
            "rank_pairs": 0,
            "tail_rank_pairs": 0,
            "preserve_pairs": 0,
        }
        for shard_index, shard in enumerate(census["shards"]):
            start, stop = int(shard["start_index"]), int(shard["stop_index"])
            cache_path = (
                args.census_dir
                / "shards"
                / f"shard_{start:09d}_{stop:09d}"
                / "candidate_cache.pt"
            )
            payload = load_cache(cache_path, "train")
            compact, stats = compact_training_rows(
                payload,
                max_negatives=args.max_negatives,
                max_preserve_pairs=args.max_preserve_pairs_per_shard,
                seed=args.seed + shard_index,
            )
            compact_shards.append(compact)
            for name in train_stats:
                train_stats[name] += stats[name]
            print(
                f"compacted shard {shard_index + 1}/{len(census['shards'])}: "
                f"examples={stats['examples']} hard_positive={stats['hard_positive_rows']}",
                flush=True,
            )
        train = concatenate_compact(compact_shards)
        del compact_shards
        write_json(
            state_path,
            {
                "status": "running",
                "phase": "fit_full_train_hard_pairs",
                "seed": args.seed,
                "test_evaluated": False,
                "train": train_stats,
            },
        )
        model, losses = train_model(
            train,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            bound=args.bound,
            tail_positive_weight=args.tail_positive_weight,
            preserve_weight=args.preserve_weight,
            distill_weight=args.distill_weight,
            gate_supervision_weight=args.gate_supervision_weight,
            gate_positive_weight=args.gate_positive_weight,
            seed=args.seed,
        )
        del train
        write_json(
            state_path,
            {
                "status": "running",
                "phase": "validation_only_selection",
                "seed": args.seed,
                "test_evaluated": False,
            },
        )
        config = load_config(Path(args.config))
        catalog = SemanticIdCatalog.load(args.artifact_dir / "catalog.json")
        groups = catalog_groups(config, catalog)
        validation = load_cache(args.validation_cache, "validation")
        candidate, row, base = make_features(validation)
        baseline_metrics, baseline_hits = evaluate(
            base, validation, groups, catalog_size=len(catalog.item_ids)
        )
        target_groups = groups[validation["target_catalog_indices"].long()]
        variants: dict[str, object] = {}
        passing: list[str] = []
        for scale in args.deployment_scales:
            started = time.perf_counter()
            scores, gate, residual = score_candidates(
                candidate,
                row,
                base,
                model["residual_weights"],
                model["gate_weights"],
                model["gate_bias"],
                bound=args.bound * scale,
            )
            score_ms = 1000.0 * (time.perf_counter() - started) / len(base)
            metrics, hits = evaluate(
                scores, validation, groups, catalog_size=len(catalog.item_ids)
            )
            tests = paired_tests(
                baseline_hits,
                hits,
                target_groups,
                samples=args.bootstrap_samples,
                seed=args.seed + round(scale * 100),
            )
            discordance = tail_discordance(baseline_hits, hits, target_groups)
            recall = metrics["ranking"]["recall_at_10"]
            baseline_recall = baseline_metrics["ranking"]["recall_at_10"]
            ndcg_delta = (
                metrics["ranking"]["ndcg_at_10"]
                - baseline_metrics["ranking"]["ndcg_at_10"]
            )
            tail_coverage_delta = (
                metrics["catalog"]["observed_tail_coverage_at_10"]
                - baseline_metrics["catalog"]["observed_tail_coverage_at_10"]
            )
            incremental_latency = score_ms + (
                metrics["ranking_milliseconds_per_example"]
                - baseline_metrics["ranking_milliseconds_per_example"]
            )
            checks = {
                "recall_at_10_retention_ge_0_99": recall / baseline_recall >= 0.99,
                "ndcg_at_10_delta_ge_minus_0_0002": ndcg_delta >= -0.0002,
                "tail_net_wins_ge_6": discordance["net_wins"] >= 6,
                "tail_bootstrap_ci_low_gt_0": tests[
                    "observed_tail_paired_user_bootstrap_top10"
                ]["ci95_low"]
                > 0,
                "tail_coverage_delta_ge_0_02": tail_coverage_delta >= 0.02,
                "incremental_latency_le_5ms": incremental_latency <= 5.0,
            }
            name = f"scale_{scale:g}"
            variants[name] = {
                "deployment_scale": scale,
                "metrics": metrics,
                "recall_at_10_retention": recall / baseline_recall,
                "ndcg_at_10_delta": ndcg_delta,
                "observed_tail_coverage_delta": tail_coverage_delta,
                "incremental_latency_ms_per_example": incremental_latency,
                "tail_discordance": discordance,
                "paired_tests": tests,
                "gate": {
                    "mean": float(gate.mean()),
                    "p50": float(gate.quantile(0.5)),
                    "p90": float(gate.quantile(0.9)),
                    "maximum": float(gate.max()),
                },
                "absolute_residual": {
                    "mean": float(residual.abs().mean()),
                    "p99": float(residual.abs().quantile(0.99)),
                    "maximum": float(residual.abs().max()),
                },
                "checks": checks,
                "passed": all(checks.values()),
            }
            if all(checks.values()):
                passing.append(name)
        selected = (
            max(
                passing,
                key=lambda name: (
                    variants[name]["metrics"]["target_groups"]["observed_tail"][
                        "recall_at_10"
                    ],
                    variants[name]["metrics"]["ranking"]["recall_at_10"],
                ),
            )
            if passing
            else None
        )
        report = {
            "status": "completed",
            "protocol": "V2.6 full-Train hard-pair fit / Validation-only selection",
            "test_evaluated": False,
            "synthetic_data_used": False,
            "seed": args.seed,
            "train": train_stats,
            "hyperparameters": {
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "max_negatives": args.max_negatives,
                "max_preserve_pairs_per_shard": args.max_preserve_pairs_per_shard,
                "bound": args.bound,
                "tail_positive_weight": args.tail_positive_weight,
                "preserve_weight": args.preserve_weight,
                "distill_weight": args.distill_weight,
                "gate_supervision_weight": args.gate_supervision_weight,
                "gate_positive_weight": args.gate_positive_weight,
                "deployment_scales": list(args.deployment_scales),
            },
            "losses": losses,
            "baseline": baseline_metrics,
            "variants": variants,
            "passing_variants": passing,
            "selected_validation_variant": selected,
            "model": serializable_model(model),
            "provenance": {
                "census_report_sha256": sha256_file(census_report_path),
                "validation_cache_sha256": sha256_file(args.validation_cache),
                "config_sha256": sha256_file(Path(args.config)),
                "catalog_sha256": sha256_file(args.artifact_dir / "catalog.json"),
            },
        }
        write_json(output_dir / "report.json", report)
        if selected is not None:
            write_json(
                output_dir / "selected_model.json",
                {
                    "type": "v2_6_full_train_bounded_residual_v1",
                    "selected_on": "validation",
                    "test_evaluated": False,
                    "selected_variant": selected,
                    "feature_names": list(FEATURE_NAMES),
                    "gate_feature_names": list(GATE_FEATURE_NAMES),
                    "effective_bound": args.bound
                    * variants[selected]["deployment_scale"],
                    "model": serializable_model(model),
                    "provenance": report["provenance"],
                },
            )
        write_json(
            state_path,
            {
                "status": "completed",
                "phase": None,
                "seed": args.seed,
                "test_evaluated": False,
                "passed_seed_2026_gate": selected is not None,
                "selected_validation_variant": selected,
                "report": str((output_dir / "report.json").resolve()),
            },
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "passed_seed_2026_gate": selected is not None,
                    "selected_validation_variant": selected,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    except Exception as error:
        write_json(
            state_path,
            {
                "status": "failed",
                "phase": "v2_6_full_hard_positive_fit",
                "seed": args.seed,
                "test_evaluated": False,
                "error": repr(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
