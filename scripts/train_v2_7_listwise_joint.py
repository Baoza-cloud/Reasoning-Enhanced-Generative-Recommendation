from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from scripts.train_v2_5_bounded_residual import (
    build_training_pairs,
    catalog_groups,
    evaluate,
    load_cache,
    make_features,
    paired_tests,
)
from scripts.train_v2_6_full_hard_positive import tail_discordance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V2.7 full-Train unbounded candidate-level listwise joint scorer"
    )
    parser.add_argument("--config", default="configs/v2_final_default.yaml")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--census-dir", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-negatives", type=int, default=20)
    parser.add_argument("--max-preserve-pairs-per-shard", type=int, default=6000)
    parser.add_argument("--pair-temperature", type=float, default=0.10)
    parser.add_argument("--boundary-margin", type=float, default=0.02)
    parser.add_argument("--tail-positive-weight", type=float, default=8.0)
    parser.add_argument("--boundary-weight", type=float, default=4.0)
    parser.add_argument("--preserve-weight", type=float, default=4.0)
    parser.add_argument("--distill-weight", type=float, default=0.01)
    parser.add_argument(
        "--deployment-scales", type=float, nargs="+", default=(0.25, 0.5, 0.75, 1.0)
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


def joint_features(
    payload: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate, row, base = make_features(payload)
    mask = payload["candidate_mask"].bool()
    semantic = payload["semantic_scores"].float().masked_fill(~mask, 0.0)
    count = mask.sum(dim=1).clamp_min(1).float()
    mean = semantic.sum(dim=1) / count
    centered = (semantic - mean.unsqueeze(1)).masked_fill(~mask, 0.0)
    variance = centered.square().sum(dim=1) / count
    semantic_z = (
        centered / variance.sqrt().clamp_min(1e-6).unsqueeze(1)
    ).clamp(-5.0, 5.0) / 5.0
    semantic_raw = semantic.clamp(-1.0, 1.0)
    candidate = torch.cat(
        (candidate, semantic_raw.unsqueeze(2), semantic_z.unsqueeze(2)), dim=2
    )
    return candidate, row, base


class JointScorer(nn.Module):
    def __init__(self, candidate_dim: int, row_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(candidate_dim + row_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, candidate: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
        if candidate.shape[:-1] != row.shape[:-1]:
            raise ValueError("candidate and row leading dimensions must match")
        return self.network(torch.cat((candidate, row), dim=-1)).squeeze(-1)


def pair_delta(
    model: JointScorer,
    positive: torch.Tensor,
    negative: torch.Tensor,
    row: torch.Tensor,
    base_difference: torch.Tensor,
    *,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_residual = model(positive, row)
    negative_residual = model(negative, row)
    delta = base_difference + scale * (positive_residual - negative_residual)
    return delta, positive_residual, negative_residual


def compact_joint_rows(
    payload: dict[str, object],
    *,
    max_negatives: int,
    max_preserve_pairs: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    candidate, row, base = joint_features(payload)
    pairs = build_training_pairs(
        payload, base, candidate[:, :, 0], max_negatives=max_negatives
    )
    rr, pp, nn_index = pairs["rank_rows"], pairs["positives"], pairs["negatives"]
    pr = pairs["preserve_rows"]
    pi, po = pairs["preserve_inside"], pairs["preserve_outside"]
    if len(pr) > max_preserve_pairs:
        generator = torch.Generator().manual_seed(seed)
        selected = torch.randperm(len(pr), generator=generator)[:max_preserve_pairs]
        pr, pi, po = pr[selected], pi[selected], po[selected]

    positions = payload["target_positions"].long()
    contained_rows = torch.nonzero(positions >= 0, as_tuple=False).flatten()
    cutoff = base.masked_fill(~payload["candidate_mask"].bool(), float("-inf")).topk(10, dim=1).indices[:, -1]
    boundary_positive = positions[contained_rows]
    boundary_negative = cutoff[contained_rows]
    boundary_groups = payload["groups"].long()[contained_rows, boundary_positive]
    compact = {
        "rank_positive": candidate[rr, pp],
        "rank_negative": candidate[rr, nn_index],
        "rank_row": row[rr],
        "rank_base_difference": base[rr, pp] - base[rr, nn_index],
        "rank_groups": pairs["group_ids"],
        "boundary_positive": candidate[contained_rows, boundary_positive],
        "boundary_negative": candidate[contained_rows, boundary_negative],
        "boundary_row": row[contained_rows],
        "boundary_base_difference": (
            base[contained_rows, boundary_positive]
            - base[contained_rows, boundary_negative]
        ),
        "boundary_groups": boundary_groups,
        "preserve_inside": candidate[pr, pi],
        "preserve_outside": candidate[pr, po],
        "preserve_row": row[pr],
        "preserve_base_difference": base[pr, pi] - base[pr, po],
    }
    hard_positive = (
        (positions >= 0)
        & ~torch.isfinite(
            payload["beam_scores"].gather(1, positions.clamp_min(0).unsqueeze(1)).squeeze(1)
        )
    )
    stats = {
        "examples": len(positions),
        "supervised_rows": int((positions >= 0).sum()),
        "hard_positive_rows": int(hard_positive.sum()),
        "rank_pairs": len(rr),
        "tail_rank_pairs": int((pairs["group_ids"] == 2).sum()),
        "boundary_pairs": len(contained_rows),
        "tail_boundary_pairs": int((boundary_groups == 2).sum()),
        "preserve_pairs": len(pr),
    }
    return compact, stats


def concatenate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("no Train shards were compacted")
    return {name: torch.cat([row[name] for row in rows]) for name in rows[0]}


def weighted_pair_loss(
    delta: torch.Tensor,
    groups: torch.Tensor,
    *,
    margin: float,
    temperature: float,
    tail_weight: float,
) -> torch.Tensor:
    weights = torch.ones_like(delta)
    weights[groups == 1] = 2.0
    weights[groups == 2] = tail_weight
    losses = F.softplus((margin - delta) / temperature)
    return (losses * weights).sum() / weights.sum()


def train_model(
    train: dict[str, torch.Tensor],
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    pair_temperature: float,
    boundary_margin: float,
    tail_positive_weight: float,
    boundary_weight: float,
    preserve_weight: float,
    distill_weight: float,
    seed: int,
) -> tuple[JointScorer, dict[str, float]]:
    torch.manual_seed(seed)
    model = JointScorer(
        train["rank_positive"].size(1), train["rank_row"].size(1), hidden_dim
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rank_count = len(train["rank_base_difference"])
    boundary_count = len(train["boundary_base_difference"])
    preserve_count = len(train["preserve_base_difference"])
    if min(rank_count, boundary_count, preserve_count) <= 0:
        raise ValueError("rank, boundary and preservation pairs must be non-empty")
    generator = torch.Generator().manual_seed(seed)
    last: dict[str, float] = {}
    for epoch in range(epochs):
        order = torch.randperm(rank_count, generator=generator)
        totals = {"loss": 0.0, "rank": 0.0, "boundary": 0.0, "preserve": 0.0, "distill": 0.0}
        steps = 0
        for start in range(0, rank_count, batch_size):
            selected = order[start : start + batch_size]
            rank_delta, rank_positive, rank_negative = pair_delta(
                model,
                train["rank_positive"][selected],
                train["rank_negative"][selected],
                train["rank_row"][selected],
                train["rank_base_difference"][selected],
            )
            rank_loss = weighted_pair_loss(
                rank_delta,
                train["rank_groups"][selected],
                margin=0.0,
                temperature=pair_temperature,
                tail_weight=tail_positive_weight,
            )
            boundary_selected = torch.randint(
                boundary_count,
                (min(batch_size, boundary_count),),
                generator=generator,
            )
            boundary_delta, boundary_positive, boundary_negative = pair_delta(
                model,
                train["boundary_positive"][boundary_selected],
                train["boundary_negative"][boundary_selected],
                train["boundary_row"][boundary_selected],
                train["boundary_base_difference"][boundary_selected],
            )
            boundary_loss = weighted_pair_loss(
                boundary_delta,
                train["boundary_groups"][boundary_selected],
                margin=boundary_margin,
                temperature=pair_temperature,
                tail_weight=tail_positive_weight,
            )
            preserve_selected = torch.randint(
                preserve_count,
                (min(batch_size, preserve_count),),
                generator=generator,
            )
            preserve_delta, preserve_inside, preserve_outside = pair_delta(
                model,
                train["preserve_inside"][preserve_selected],
                train["preserve_outside"][preserve_selected],
                train["preserve_row"][preserve_selected],
                train["preserve_base_difference"][preserve_selected],
            )
            preserve_loss = F.relu(0.01 - preserve_delta).mean()
            residuals = torch.cat(
                (
                    rank_positive,
                    rank_negative,
                    boundary_positive,
                    boundary_negative,
                    preserve_inside,
                    preserve_outside,
                )
            )
            distill_loss = residuals.square().mean()
            loss = (
                rank_loss
                + boundary_weight * boundary_loss
                + preserve_weight * preserve_loss
                + distill_weight * distill_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for name, value in (
                ("loss", loss),
                ("rank", rank_loss),
                ("boundary", boundary_loss),
                ("preserve", preserve_loss),
                ("distill", distill_loss),
            ):
                totals[name] += float(value.detach())
            steps += 1
        last = {name: value / steps for name, value in totals.items()}
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(
                f"epoch={epoch + 1}/{epochs} loss={last['loss']:.6f} "
                f"rank={last['rank']:.6f} boundary={last['boundary']:.6f} "
                f"preserve={last['preserve']:.6f}",
                flush=True,
            )
    return model.eval(), last


@torch.inference_mode()
def score_all(
    model: JointScorer,
    candidate: torch.Tensor,
    row: torch.Tensor,
    *,
    batch_rows: int = 2048,
) -> torch.Tensor:
    output = torch.empty(candidate.shape[:2], dtype=torch.float32)
    for start in range(0, len(candidate), batch_rows):
        stop = min(start + batch_rows, len(candidate))
        current = candidate[start:stop]
        row_expanded = row[start:stop].unsqueeze(1).expand(-1, current.size(1), -1)
        output[start:stop] = model(
            current.reshape(-1, current.size(2)),
            row_expanded.reshape(-1, row.size(1)),
        ).reshape(stop - start, current.size(1))
    return output


def model_payload(model: JointScorer) -> dict[str, object]:
    return {name: tensor.detach().tolist() for name, tensor in model.state_dict().items()}


def main() -> None:
    args = parse_args()
    positive = (
        args.hidden_dim,
        args.epochs,
        args.batch_size,
        args.max_negatives,
        args.max_preserve_pairs_per_shard,
        args.bootstrap_samples,
        args.torch_threads,
    )
    if min(positive) <= 0:
        raise ValueError("dimensions, counts and threads must be positive")
    if (
        args.learning_rate <= 0
        or args.pair_temperature <= 0
        or args.boundary_margin < 0
        or args.tail_positive_weight <= 0
        or args.boundary_weight < 0
        or args.preserve_weight < 0
        or args.distill_weight < 0
        or any(not 0 < scale <= 1 for scale in args.deployment_scales)
    ):
        raise ValueError("invalid V2.7 hyperparameters")
    torch.set_num_threads(args.torch_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "state.json"
    write_json(
        state_path,
        {"status": "running", "phase": "compact_full_train_pairs", "seed": args.seed, "test_evaluated": False},
    )
    try:
        census_path = args.census_dir / "report.json"
        census = json.loads(census_path.read_text(encoding="utf-8"))
        if census.get("status") != "completed" or not census.get("go_no_go", {}).get("passed"):
            raise ValueError("V2.7 requires the completed and passing V2.6 Train census")
        if census.get("test_evaluated") is not False:
            raise ValueError("V2.7 Train census must not evaluate Test")
        compact_shards: list[dict[str, torch.Tensor]] = []
        stats = {
            "examples": 0,
            "supervised_rows": 0,
            "hard_positive_rows": 0,
            "rank_pairs": 0,
            "tail_rank_pairs": 0,
            "boundary_pairs": 0,
            "tail_boundary_pairs": 0,
            "preserve_pairs": 0,
        }
        for shard_index, shard in enumerate(census["shards"]):
            start, stop = int(shard["start_index"]), int(shard["stop_index"])
            cache = args.census_dir / "shards" / f"shard_{start:09d}_{stop:09d}" / "candidate_cache.pt"
            payload = load_cache(cache, "train")
            compact, current = compact_joint_rows(
                payload,
                max_negatives=args.max_negatives,
                max_preserve_pairs=args.max_preserve_pairs_per_shard,
                seed=args.seed + shard_index,
            )
            compact_shards.append(compact)
            for name in stats:
                stats[name] += current[name]
            print(
                f"compacted shard {shard_index + 1}/{len(census['shards'])}: "
                f"hard_positive={current['hard_positive_rows']}",
                flush=True,
            )
        train = concatenate(compact_shards)
        del compact_shards
        write_json(
            state_path,
            {"status": "running", "phase": "fit_listwise_joint_scorer", "seed": args.seed, "test_evaluated": False, "train": stats},
        )
        model, losses = train_model(
            train,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            pair_temperature=args.pair_temperature,
            boundary_margin=args.boundary_margin,
            tail_positive_weight=args.tail_positive_weight,
            boundary_weight=args.boundary_weight,
            preserve_weight=args.preserve_weight,
            distill_weight=args.distill_weight,
            seed=args.seed,
        )
        del train
        write_json(
            state_path,
            {"status": "running", "phase": "validation_only_gate", "seed": args.seed, "test_evaluated": False},
        )
        config = load_config(Path(args.config))
        catalog = SemanticIdCatalog.load(args.artifact_dir / "catalog.json")
        group_lookup = catalog_groups(config, catalog)
        validation = load_cache(args.validation_cache, "validation")
        candidate, row, base = joint_features(validation)
        baseline_metrics, baseline_hits = evaluate(
            base, validation, group_lookup, catalog_size=len(catalog.item_ids)
        )
        target_groups = group_lookup[validation["target_catalog_indices"].long()]
        started = time.perf_counter()
        raw_residual = score_all(model, candidate, row)
        scoring_ms = 1000.0 * (time.perf_counter() - started) / len(base)
        variants: dict[str, object] = {}
        passing: list[str] = []
        for scale in args.deployment_scales:
            scores = base + scale * raw_residual
            metrics, hits = evaluate(
                scores, validation, group_lookup, catalog_size=len(catalog.item_ids)
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
            ndcg_delta = metrics["ranking"]["ndcg_at_10"] - baseline_metrics["ranking"]["ndcg_at_10"]
            coverage_delta = metrics["catalog"]["observed_tail_coverage_at_10"] - baseline_metrics["catalog"]["observed_tail_coverage_at_10"]
            incremental_latency = scoring_ms + metrics["ranking_milliseconds_per_example"] - baseline_metrics["ranking_milliseconds_per_example"]
            checks = {
                "recall_at_10_retention_ge_0_99": recall / baseline_recall >= 0.99,
                "ndcg_at_10_delta_ge_minus_0_0002": ndcg_delta >= -0.0002,
                "tail_net_wins_ge_6": discordance["net_wins"] >= 6,
                "tail_bootstrap_ci_low_gt_0": tests["observed_tail_paired_user_bootstrap_top10"]["ci95_low"] > 0,
                "tail_coverage_delta_ge_0_02": coverage_delta >= 0.02,
                "incremental_latency_le_5ms": incremental_latency <= 5.0,
            }
            name = f"scale_{scale:g}"
            variants[name] = {
                "deployment_scale": scale,
                "metrics": metrics,
                "recall_at_10_retention": recall / baseline_recall,
                "ndcg_at_10_delta": ndcg_delta,
                "observed_tail_coverage_delta": coverage_delta,
                "incremental_latency_ms_per_example": incremental_latency,
                "tail_discordance": discordance,
                "paired_tests": tests,
                "absolute_residual": {
                    "mean": float((scale * raw_residual).abs().mean()),
                    "p99": float((scale * raw_residual).abs().quantile(0.99)),
                    "maximum": float((scale * raw_residual).abs().max()),
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
                    variants[name]["metrics"]["target_groups"]["observed_tail"]["recall_at_10"],
                    variants[name]["metrics"]["ranking"]["recall_at_10"],
                ),
            )
            if passing
            else None
        )
        report = {
            "status": "completed",
            "protocol": "V2.7 full-Train candidate-level listwise joint scorer / Validation-only gate",
            "test_evaluated": False,
            "synthetic_data_used": False,
            "seed": args.seed,
            "train": stats,
            "hyperparameters": {
                "hidden_dim": args.hidden_dim,
                "activation": "tanh",
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "max_negatives": args.max_negatives,
                "max_preserve_pairs_per_shard": args.max_preserve_pairs_per_shard,
                "pair_temperature": args.pair_temperature,
                "boundary_margin": args.boundary_margin,
                "tail_positive_weight": args.tail_positive_weight,
                "boundary_weight": args.boundary_weight,
                "preserve_weight": args.preserve_weight,
                "distill_weight": args.distill_weight,
                "deployment_scales": list(args.deployment_scales),
            },
            "losses": losses,
            "baseline": baseline_metrics,
            "raw_scoring_latency_ms_per_example": scoring_ms,
            "variants": variants,
            "passing_variants": passing,
            "selected_validation_variant": selected,
            "model": model_payload(model),
            "provenance": {
                "census_report_sha256": sha256_file(census_path),
                "validation_cache_sha256": sha256_file(args.validation_cache),
                "protocol_sha256": sha256_file(args.protocol),
                "config_sha256": sha256_file(Path(args.config)),
                "catalog_sha256": sha256_file(args.artifact_dir / "catalog.json"),
            },
        }
        write_json(args.output_dir / "report.json", report)
        torch.save(
            {
                "type": "v2_7_candidate_listwise_joint_v1",
                "seed": args.seed,
                "selected_validation_variant": selected,
                "state_dict": model.state_dict(),
                "hyperparameters": report["hyperparameters"],
                "provenance": report["provenance"],
                "test_evaluated": False,
            },
            args.output_dir / "model.pt",
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
                "report": str((args.output_dir / "report.json").resolve()),
            },
        )
        print(json.dumps({"status": "completed", "passed_seed_2026_gate": selected is not None, "selected_validation_variant": selected}, indent=2))
    except Exception as error:
        write_json(
            state_path,
            {"status": "failed", "phase": "v2_7_listwise_joint", "seed": args.seed, "test_evaluated": False, "error": repr(error)},
        )
        raise


if __name__ == "__main__":
    main()
