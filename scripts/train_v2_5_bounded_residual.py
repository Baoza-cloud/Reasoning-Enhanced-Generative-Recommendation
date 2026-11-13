from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from regenrec.data import iter_sequence_examples_jsonl
from regenrec.metrics import popularity_strata


FEATURE_NAMES = (
    "semantic_rank",
    "generator_rank",
    "retrieval_only",
    "observed_tail",
    "torso",
    "inverse_popularity",
    "semantic_x_retrieval",
    "semantic_x_tail",
    "semantic_x_inverse_popularity",
)
GATE_FEATURE_NAMES = (
    "inverse_beam_margin",
    "semantic_retrieval_gap",
    "semantic_top_is_retrieval",
    "retrieval_fraction",
    "normalized_history_length",
    "beam_entropy",
)
GROUP_NAMES = ("head", "torso", "observed_tail", "cold_start")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train-only bounded residual scorer; Validation-only selection"
    )
    parser.add_argument("--config", default="configs/v2_final_default.yaml")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-negatives", type=int, default=20)
    parser.add_argument("--distill-weight", type=float, default=0.2)
    parser.add_argument("--gate-supervision-weight", type=float, default=1.0)
    parser.add_argument("--gate-positive-weight", type=float, default=20.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bounds", type=float, nargs="+", default=(0.50, 0.75))
    parser.add_argument(
        "--tail-positive-weights", type=float, nargs="+", default=(4.0, 8.0)
    )
    parser.add_argument(
        "--preserve-weights", type=float, nargs="+", default=(1.0, 4.0)
    )
    parser.add_argument(
        "--deployment-scales",
        type=float,
        nargs="+",
        default=(0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0),
    )
    parser.add_argument("--torch-threads", type=int, default=2)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(path: Path, expected_split: str) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    if metadata.get("split") != expected_split:
        raise ValueError(
            f"cache split mismatch: expected {expected_split}, got {metadata.get('split')}"
        )
    if metadata.get("test_evaluated") is not False:
        raise ValueError("V2.5 scorer development must not use Test")
    required = {
        "candidate_catalog_indices",
        "candidate_mask",
        "teacher_scores",
        "semantic_scores",
        "beam_scores",
        "groups",
        "popularity",
        "target_catalog_indices",
        "target_positions",
        "history_lengths",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"candidate cache is missing fields: {missing}")
    return payload


def rank_scores(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    output = torch.full_like(values, float("-inf"))
    for row in range(values.size(0)):
        valid = torch.nonzero(mask[row], as_tuple=False).flatten()
        ordered = valid[torch.argsort(values[row, valid], descending=True, stable=True)]
        if len(ordered) == 1:
            output[row, ordered] = 1.0
        else:
            output[row, ordered] = torch.linspace(1.0, 0.0, len(ordered))
    return output


def make_features(
    payload: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = payload["candidate_mask"].bool()
    semantic = payload["semantic_scores"].float()
    beam = payload["beam_scores"].float()
    groups = payload["groups"].long()
    popularity = payload["popularity"].float()
    semantic_rank = rank_scores(semantic, mask)
    finite_beam = torch.isfinite(beam)
    generator_rank = rank_scores(beam, finite_beam).masked_fill(~finite_beam, 0.0)
    retrieval = (~finite_beam & mask).float()
    tail = (groups == 2).float()
    torso = (groups == 1).float()
    candidate = torch.stack(
        (
            semantic_rank.masked_fill(~mask, 0.0),
            generator_rank,
            retrieval,
            tail,
            torso,
            (1.0 - popularity).masked_fill(~mask, 0.0),
            semantic_rank.masked_fill(~mask, 0.0) * retrieval,
            semantic_rank.masked_fill(~mask, 0.0) * tail,
            semantic_rank.masked_fill(~mask, 0.0)
            * (1.0 - popularity).masked_fill(~mask, 0.0),
        ),
        dim=-1,
    )

    beam_values = beam.masked_fill(~finite_beam, float("-inf"))
    sorted_beam = beam_values.sort(dim=1, descending=True).values
    margin = (sorted_beam[:, 0] - sorted_beam[:, 9]).clamp_min(0.0)
    inverse_margin = torch.exp(-margin)
    retrieval_semantic = semantic.masked_fill(~(retrieval.bool()), float("-inf"))
    generated_semantic = semantic.masked_fill(~finite_beam, float("-inf"))
    semantic_gap = retrieval_semantic.max(dim=1).values - generated_semantic.max(
        dim=1
    ).values
    semantic_gap = torch.sigmoid(20.0 * semantic_gap)
    semantic_top = semantic.masked_fill(~mask, float("-inf")).argmax(dim=1)
    semantic_top_is_retrieval = retrieval.gather(1, semantic_top.unsqueeze(1)).squeeze(1)
    retrieval_fraction = retrieval.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    history = payload["history_lengths"].float().clamp(max=50.0) / 50.0
    beam_probabilities = torch.softmax(beam_values[:, :20], dim=1)
    beam_entropy = -(
        beam_probabilities * beam_probabilities.clamp_min(1e-12).log()
    ).sum(dim=1) / math.log(20.0)
    row = torch.stack(
        (
            inverse_margin,
            semantic_gap,
            semantic_top_is_retrieval,
            retrieval_fraction,
            history,
            beam_entropy,
        ),
        dim=1,
    )
    return candidate, row, generator_rank


def build_training_pairs(
    payload: dict[str, object],
    base: torch.Tensor,
    semantic_rank: torch.Tensor,
    *,
    max_negatives: int,
) -> dict[str, torch.Tensor]:
    mask = payload["candidate_mask"].bool()
    positions = payload["target_positions"].long()
    groups = payload["groups"].long()
    rank_rows: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    group_ids: list[int] = []
    preserve_rows: list[int] = []
    preserve_inside: list[int] = []
    preserve_outside: list[int] = []
    for row in range(mask.size(0)):
        valid = torch.nonzero(mask[row], as_tuple=False).flatten()
        base_order = valid[torch.argsort(base[row, valid], descending=True, stable=True)]
        semantic_order = valid[
            torch.argsort(semantic_rank[row, valid], descending=True, stable=True)
        ]
        positive = int(positions[row])
        if positive >= 0:
            hard: list[int] = []
            for index in (*base_order[:20].tolist(), *semantic_order[:20].tolist()):
                candidate = int(index)
                if candidate != positive and candidate not in hard:
                    hard.append(candidate)
                if len(hard) >= max_negatives:
                    break
            for negative in hard:
                rank_rows.append(row)
                positives.append(positive)
                negatives.append(negative)
                group_ids.append(int(groups[row, positive]))
        if len(base_order) >= 11:
            cutoff = int(base_order[9])
            outside = [int(index) for index in semantic_order if float(base[row, index]) < float(base[row, cutoff])]
            for candidate in outside[:10]:
                if candidate == positive:
                    continue
                preserve_rows.append(row)
                preserve_inside.append(cutoff)
                preserve_outside.append(candidate)
    if not rank_rows:
        raise ValueError("no target-contained Train rows are available")
    return {
        "rank_rows": torch.tensor(rank_rows, dtype=torch.long),
        "positives": torch.tensor(positives, dtype=torch.long),
        "negatives": torch.tensor(negatives, dtype=torch.long),
        "group_ids": torch.tensor(group_ids, dtype=torch.long),
        "preserve_rows": torch.tensor(preserve_rows, dtype=torch.long),
        "preserve_inside": torch.tensor(preserve_inside, dtype=torch.long),
        "preserve_outside": torch.tensor(preserve_outside, dtype=torch.long),
    }


def score_candidates(
    candidate_features: torch.Tensor,
    row_features: torch.Tensor,
    base: torch.Tensor,
    residual_weights: torch.Tensor,
    gate_weights: torch.Tensor,
    gate_bias: torch.Tensor,
    *,
    bound: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate = torch.sigmoid(row_features @ gate_weights + gate_bias)
    raw = torch.einsum("nkd,d->nk", candidate_features, residual_weights)
    residual = bound * gate.unsqueeze(1) * torch.tanh(raw)
    return base + residual, gate, residual


def fit_model(
    candidate_features: torch.Tensor,
    row_features: torch.Tensor,
    base: torch.Tensor,
    pairs: dict[str, torch.Tensor],
    gate_labels: torch.Tensor,
    *,
    bound: float,
    tail_weight: float,
    preserve_weight: float,
    distill_weight: float,
    gate_supervision_weight: float,
    gate_positive_weight: float,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[dict[str, object], dict[str, float]]:
    torch.manual_seed(seed)
    residual_weights = torch.zeros(len(FEATURE_NAMES), requires_grad=True)
    gate_weights = torch.zeros(len(GATE_FEATURE_NAMES), requires_grad=True)
    gate_bias = torch.tensor(-2.1972246, requires_grad=True)
    optimizer = torch.optim.Adam(
        [residual_weights, gate_weights, gate_bias], lr=learning_rate
    )
    rr = pairs["rank_rows"]
    pp = pairs["positives"]
    nn = pairs["negatives"]
    pair_weights = torch.ones(len(rr))
    pair_weights[pairs["group_ids"] == 1] = 2.0
    pair_weights[pairs["group_ids"] == 2] = tail_weight
    pr = pairs["preserve_rows"]
    pi = pairs["preserve_inside"]
    po = pairs["preserve_outside"]
    final_components: dict[str, float] = {}
    for _ in range(epochs):
        scores, _, residual = score_candidates(
            candidate_features,
            row_features,
            base,
            residual_weights,
            gate_weights,
            gate_bias,
            bound=bound,
        )
        pair_delta = scores[rr, pp] - scores[rr, nn]
        rank_loss = (
            F.softplus(-pair_delta / 0.10) * pair_weights
        ).sum() / pair_weights.sum()
        preserve_delta = scores[pr, pi] - scores[pr, po]
        preserve_loss = F.relu(0.01 - preserve_delta).mean()
        touched_residual = torch.cat((residual[rr, pp], residual[rr, nn]))
        distill_loss = touched_residual.square().mean()
        gate_logits = row_features @ gate_weights + gate_bias
        gate_loss = F.binary_cross_entropy_with_logits(
            gate_logits,
            gate_labels,
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
        final_components = {
            "loss": float(loss.detach()),
            "rank_loss": float(rank_loss.detach()),
            "preserve_loss": float(preserve_loss.detach()),
            "distill_loss": float(distill_loss.detach()),
            "gate_loss": float(gate_loss.detach()),
        }
    return {
        "residual_weights": residual_weights.detach(),
        "gate_weights": gate_weights.detach(),
        "gate_bias": gate_bias.detach(),
    }, final_components


def catalog_groups(config: dict[str, object], catalog: SemanticIdCatalog) -> torch.Tensor:
    frequencies: Counter[str] = Counter()
    for example in iter_sequence_examples_jsonl(config["data"]["train_path"]):
        frequencies[example.target_item_id] += 1
    strata = popularity_strata(catalog.item_ids, frequencies)
    item_group = {item: group for group, items in strata.items() for item in items}
    return torch.tensor(
        [GROUP_NAMES.index(item_group[item]) for item in catalog.item_ids],
        dtype=torch.long,
    )


def evaluate(
    scores: torch.Tensor,
    payload: dict[str, object],
    group_lookup: torch.Tensor,
    *,
    catalog_size: int,
) -> tuple[dict[str, object], torch.Tensor]:
    started = time.perf_counter()
    mask = payload["candidate_mask"].bool()
    candidate_indices = payload["candidate_catalog_indices"].long()
    scores = scores.masked_fill(~mask, float("-inf"))
    top10_positions = scores.topk(10, dim=1).indices
    predictions = candidate_indices.gather(1, top10_positions)
    target = payload["target_catalog_indices"].long()
    matches = predictions.eq(target.unsqueeze(1))
    hits10 = matches.any(dim=1)
    hits5 = matches[:, :5].any(dim=1)
    ranks = torch.where(
        matches,
        torch.arange(1, 11).unsqueeze(0),
        torch.full_like(matches, 11, dtype=torch.long),
    ).min(dim=1).values
    ndcg10 = torch.where(
        hits10, 1.0 / torch.log2(ranks.float() + 1.0), torch.zeros_like(ranks.float())
    )
    ndcg5 = torch.where(hits5, ndcg10, torch.zeros_like(ndcg10))
    target_groups = group_lookup[target]
    group_metrics: dict[str, object] = {}
    for group_id, name in enumerate(GROUP_NAMES):
        selected = target_groups == group_id
        count = int(selected.sum())
        group_metrics[name] = {
            "examples": count,
            "recall_at_5": float(hits5[selected].float().mean()) if count else 0.0,
            "recall_at_10": float(hits10[selected].float().mean()) if count else 0.0,
            "ndcg_at_10": float(ndcg10[selected].mean()) if count else 0.0,
        }
    unique = torch.unique(predictions)
    tail_unique = unique[group_lookup[unique] == 2]
    tail_catalog_size = int((group_lookup == 2).sum())
    prediction_groups = group_lookup[predictions]
    elapsed = time.perf_counter() - started
    return {
        "ranking": {
            "recall_at_5": float(hits5.float().mean()),
            "ndcg_at_5": float(ndcg5.mean()),
            "recall_at_10": float(hits10.float().mean()),
            "ndcg_at_10": float(ndcg10.mean()),
        },
        "target_groups": group_metrics,
        "catalog": {
            "coverage_at_10": len(unique) / catalog_size,
            "observed_tail_coverage_at_10": (
                len(tail_unique) / tail_catalog_size if tail_catalog_size else 0.0
            ),
            "observed_tail_recommendation_share_at_10": float(
                (prediction_groups == 2).float().mean()
            ),
        },
        "ranking_milliseconds_per_example": 1000.0 * elapsed / len(target),
    }, hits10


def serializable_model(model: dict[str, object]) -> dict[str, object]:
    return {
        "residual_weights": model["residual_weights"].tolist(),
        "gate_weights": model["gate_weights"].tolist(),
        "gate_bias": float(model["gate_bias"]),
    }


def paired_tests(
    baseline_hits: torch.Tensor,
    candidate_hits: torch.Tensor,
    target_groups: torch.Tensor,
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    def summarize(differences: np.ndarray, current_seed: int) -> dict[str, object]:
        if len(differences) == 0:
            return {
                "examples": 0,
                "mean_delta": 0.0,
                "ci95_low": 0.0,
                "ci95_high": 0.0,
                "bootstrap_samples": samples,
            }
        values, counts = np.unique(differences, return_counts=True)
        rng = np.random.default_rng(current_seed)
        draws = rng.multinomial(
            len(differences), counts.astype(np.float64) / len(differences), size=samples
        )
        distribution = draws @ values.astype(np.float64) / len(differences)
        return {
            "examples": len(differences),
            "mean_delta": float(differences.mean()),
            "ci95_low": float(np.quantile(distribution, 0.025)),
            "ci95_high": float(np.quantile(distribution, 0.975)),
            "bootstrap_samples": samples,
        }

    baseline = baseline_hits.cpu().numpy().astype(bool)
    candidate = candidate_hits.cpu().numpy().astype(bool)
    differences = candidate.astype(np.int8) - baseline.astype(np.int8)
    wins = int(np.sum((~baseline) & candidate))
    losses = int(np.sum(baseline & (~candidate)))
    discordant = wins + losses
    exact_probability = exact_mcnemar_probability(wins, losses)
    tail_mask = target_groups.cpu().numpy() == 2
    return {
        "mcnemar_exact_top10": {
            "wins": wins,
            "losses": losses,
            "discordant": discordant,
            "two_sided_p": exact_probability,
        },
        "paired_user_bootstrap_top10": summarize(differences, seed),
        "observed_tail_paired_user_bootstrap_top10": summarize(
            differences[tail_mask], seed + 1
        ),
    }


def exact_mcnemar_probability(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value without overflowing on large tables."""
    if wins < 0 or losses < 0:
        raise ValueError("McNemar counts must be non-negative")
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    smaller = min(wins, losses)
    if 2 * smaller >= discordant:
        return 1.0
    # The largest term in the lower tail is C(n, smaller). Sum all earlier
    # terms relative to it, then restore its log probability. This is exactly
    # the same binomial tail as the integer formula but remains finite for
    # tens of thousands of discordant pairs.
    relative_sum = 1.0
    relative_term = 1.0
    for value in range(smaller, 0, -1):
        relative_term *= value / (discordant - value + 1)
        relative_sum += relative_term
    log_largest = (
        math.lgamma(discordant + 1)
        - math.lgamma(smaller + 1)
        - math.lgamma(discordant - smaller + 1)
        - discordant * math.log(2.0)
    )
    log_tail = log_largest + math.log(relative_sum)
    return min(1.0, 2.0 * math.exp(log_tail))


def main() -> None:
    args = parse_args()
    if min(
        args.epochs, args.max_negatives, args.torch_threads, args.bootstrap_samples
    ) <= 0:
        raise ValueError("epoch, negative and thread settings must be positive")
    if (
        args.learning_rate <= 0
        or args.distill_weight < 0
        or args.gate_supervision_weight < 0
        or args.gate_positive_weight <= 0
        or any(value <= 0 for value in args.bounds)
        or any(value <= 0 for value in args.tail_positive_weights)
        or any(value < 0 for value in args.preserve_weights)
        or any(not 0 < value <= 1 for value in args.deployment_scales)
    ):
        raise ValueError("invalid optimization settings")
    torch.set_num_threads(args.torch_threads)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    write_json(
        state_path,
        {
            "status": "running",
            "phase": "load_train_validation_caches",
            "test_evaluated": False,
        },
    )
    try:
        train = load_cache(args.train_cache, "train")
        validation = load_cache(args.validation_cache, "validation")
        train_provenance = train["metadata"].get("provenance", {})
        validation_provenance = validation["metadata"].get("provenance", {})
        for key in ("model_sha256", "catalog_sha256", "embedding_sha256"):
            if train_provenance.get(key) != validation_provenance.get(key):
                raise ValueError(f"Train/Validation provenance mismatch for {key}")
        config = load_config(Path(args.config))
        catalog = SemanticIdCatalog.load(args.artifact_dir / "catalog.json")
        group_lookup = catalog_groups(config, catalog)
        train_candidate, train_row, train_base = make_features(train)
        validation_candidate, validation_row, validation_base = make_features(validation)
        train_semantic_rank = train_candidate[:, :, 0]
        pairs = build_training_pairs(
            train,
            train_base,
            train_semantic_rank,
            max_negatives=args.max_negatives,
        )
        train_positions = train["target_positions"].long()
        contained = train_positions >= 0
        safe_positions = train_positions.clamp_min(0).unsqueeze(1)
        target_is_retrieval_only = ~torch.isfinite(
            train["beam_scores"].gather(1, safe_positions).squeeze(1)
        )
        gate_labels = (contained & target_is_retrieval_only).float()
        baseline_metrics, baseline_hits = evaluate(
            validation_base,
            validation,
            group_lookup,
            catalog_size=len(catalog.item_ids),
        )
        write_json(
            state_path,
            {
                "status": "running",
                "phase": "bounded_residual_grid",
                "test_evaluated": False,
                "train_examples": int(train_base.size(0)),
                "validation_examples": int(validation_base.size(0)),
                "supervised_train_rows": int((train["target_positions"] >= 0).sum()),
                "rank_pairs": len(pairs["rank_rows"]),
                "preservation_pairs": len(pairs["preserve_rows"]),
            },
        )
        results: dict[str, object] = {}
        validation_hits: dict[str, torch.Tensor] = {}
        for index, (bound, tail_weight, preserve_weight) in enumerate(
            itertools.product(
                args.bounds, args.tail_positive_weights, args.preserve_weights
            )
        ):
            fit_name = f"b{bound:g}_tail{tail_weight:g}_preserve{preserve_weight:g}"
            model, losses = fit_model(
                train_candidate,
                train_row,
                train_base,
                pairs,
                gate_labels,
                bound=bound,
                tail_weight=tail_weight,
                preserve_weight=preserve_weight,
                distill_weight=args.distill_weight,
                gate_supervision_weight=args.gate_supervision_weight,
                gate_positive_weight=args.gate_positive_weight,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=args.seed + index,
            )
            for deployment_scale in args.deployment_scales:
                name = f"{fit_name}_scale{deployment_scale:g}"
                validation_scores, gate, residual = score_candidates(
                    validation_candidate,
                    validation_row,
                    validation_base,
                    model["residual_weights"],
                    model["gate_weights"],
                    model["gate_bias"],
                    bound=bound * deployment_scale,
                )
                metrics, hits = evaluate(
                    validation_scores,
                    validation,
                    group_lookup,
                    catalog_size=len(catalog.item_ids),
                )
                recall = metrics["ranking"]["recall_at_10"]
                baseline_recall = baseline_metrics["ranking"]["recall_at_10"]
                results[name] = {
                    "hyperparameters": {
                        "training_bound": bound,
                        "deployment_scale": deployment_scale,
                        "effective_bound": bound * deployment_scale,
                        "tail_positive_weight": tail_weight,
                        "preserve_weight": preserve_weight,
                        "distill_weight": args.distill_weight,
                        "gate_supervision_weight": args.gate_supervision_weight,
                        "gate_positive_weight": args.gate_positive_weight,
                    },
                    "losses": losses,
                    "metrics": metrics,
                    "recall_at_10_retention": recall / baseline_recall,
                    "gate": {
                        "mean": float(gate.mean()),
                        "p50": float(gate.quantile(0.50)),
                        "p90": float(gate.quantile(0.90)),
                        "maximum": float(gate.max()),
                    },
                    "absolute_residual": {
                        "mean": float(residual.abs().mean()),
                        "p99": float(residual.abs().quantile(0.99)),
                        "maximum": float(residual.abs().max()),
                    },
                    "paired_top10": {
                        "wins": int((hits & ~baseline_hits).sum()),
                        "losses": int((~hits & baseline_hits).sum()),
                    },
                    "model": serializable_model(model),
                }
                validation_hits[name] = hits
            print(
                f"trained {fit_name}: evaluated {len(args.deployment_scales)} "
                "deployment scales",
                flush=True,
            )
        feasible = [
            name
            for name, result in results.items()
            if result["recall_at_10_retention"] >= 0.99
            and result["metrics"]["ranking"]["ndcg_at_10"]
            >= baseline_metrics["ranking"]["ndcg_at_10"] - 0.0004
        ]
        selected = (
            max(
                feasible,
                key=lambda name: (
                    results[name]["metrics"]["target_groups"]["observed_tail"][
                        "recall_at_10"
                    ],
                    results[name]["metrics"]["ranking"]["recall_at_10"],
                    results[name]["metrics"]["ranking"]["ndcg_at_10"],
                ),
            )
            if feasible
            else None
        )
        target_groups = group_lookup[validation["target_catalog_indices"].long()]
        selected_paired_tests = (
            paired_tests(
                baseline_hits,
                validation_hits[selected],
                target_groups,
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            if selected is not None
            else None
        )
        report = {
            "status": "completed",
            "protocol": "V2.5 Train-only fit / Validation-only selection",
            "test_evaluated": False,
            "synthetic_data_used": False,
            "seed": args.seed,
            "feature_names": list(FEATURE_NAMES),
            "gate_feature_names": list(GATE_FEATURE_NAMES),
            "search_space": {
                "bounds": list(args.bounds),
                "tail_positive_weights": list(args.tail_positive_weights),
                "preserve_weights": list(args.preserve_weights),
                "deployment_scales": list(args.deployment_scales),
            },
            "train": {
                "examples": int(train_base.size(0)),
                "supervised_rows": int((train["target_positions"] >= 0).sum()),
                "rank_pairs": len(pairs["rank_rows"]),
                "preservation_pairs": len(pairs["preserve_rows"]),
                "positive_gate_rows": int(gate_labels.sum()),
            },
            "validation": {"examples": int(validation_base.size(0))},
            "baseline": baseline_metrics,
            "variants": results,
            "selected_validation_variant": selected,
            "selected_paired_tests": selected_paired_tests,
            "selection_gate": {
                "minimum_recall_at_10_retention": 0.99,
                "maximum_ndcg_at_10_drop": 0.0004,
                "feasible_variants": feasible,
            },
            "provenance": {
                "train_cache_sha256": sha256_file(args.train_cache),
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
                    "type": "bounded_zero_init_residual_v1",
                    "selected_on": "validation",
                    "test_evaluated": False,
                    "feature_names": list(FEATURE_NAMES),
                    "gate_feature_names": list(GATE_FEATURE_NAMES),
                    "hyperparameters": results[selected]["hyperparameters"],
                    "model": results[selected]["model"],
                    "provenance": report["provenance"],
                },
            )
        write_json(
            state_path,
            {
                "status": "completed",
                "phase": None,
                "test_evaluated": False,
                "selected_validation_variant": selected,
            },
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "selected_validation_variant": selected,
                    "report": str((output_dir / "report.json").resolve()),
                },
                indent=2,
            ),
            flush=True,
        )
    except Exception as error:
        write_json(
            state_path,
            {
                "status": "failed",
                "phase": "bounded_residual",
                "test_evaluated": False,
                "error": repr(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
