from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from regenrec.data import RecommendationDataset, SequenceCollator, load_sequence_examples_jsonl
from regenrec.metrics import mean_confidence_interval
from regenrec.models.generative import GenerativeRecommender, PreferenceState


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2026, 2027, 2028)
METRICS = ("recall_at_5", "ndcg_at_5", "recall_at_10", "ndcg_at_10")
BEHAVIOR_LABELS = {
    1: "negative_review_1_2",
    2: "neutral_review_3",
    3: "positive_review_4_5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize explicit Preference Reasoning ablation and real explainability cases"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/preference_reasoning_ablation_v1"),
    )
    parser.add_argument("--case-seed", type=int, default=2026)
    parser.add_argument("--cases-per-outcome", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_row(report: dict[str, object]) -> dict[str, object]:
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("beam report must contain results")
    for result in results:
        if (
            int(result["beam_size"]) == 20
            and int(result["return_count"]) == 10
            and float(result.get("diversity_weight", 0.0)) == 0.0
        ):
            return result
    raise ValueError("missing Beam-20/Top-10 result")


def flatten_metrics(result: dict[str, object]) -> dict[str, float]:
    ranking = result["ranking"]
    tail = result["target_groups"]["observed_tail"]
    coverage = result["coverage"]
    strata = result["strata_coverage"]["strata"]["observed_tail"]
    efficiency = result["efficiency"]
    return {
        **{metric: float(ranking[metric]) for metric in METRICS},
        "observed_tail_recall_at_5": float(tail["recall_at_5"]),
        "observed_tail_ndcg_at_5": float(tail["ndcg_at_5"]),
        "observed_tail_recall_at_10": float(tail["recall_at_10"]),
        "observed_tail_ndcg_at_10": float(tail["ndcg_at_10"]),
        "catalog_coverage": float(coverage["catalog_coverage"]),
        "observed_tail_catalog_coverage": float(strata["catalog_coverage"]),
        "observed_tail_recommendation_share": float(strata["recommendation_share"]),
        "latency_ms_per_example": float(efficiency["milliseconds_per_example"]),
    }


def indicators(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def indicator_rank(row: dict[str, object]) -> object:
    """Read Top-10 rank from either beam or candidate-reranking diagnostics."""
    if "top_k_rank" in row:
        return row["top_k_rank"]
    if "baseline_rank" in row:
        return row["baseline_rank"]
    raise ValueError("hit-indicator row has no Top-10 rank field")


def exact_mcnemar(full: np.ndarray, ablated: np.ndarray) -> dict[str, float | int]:
    full_only = int(np.sum((full == 1) & (ablated == 0)))
    ablated_only = int(np.sum((full == 0) & (ablated == 1)))
    discordant = full_only + ablated_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(full_only, ablated_only)
        tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / 2**discordant
        p_value = min(1.0, 2.0 * tail)
    return {
        "full_only_hits": full_only,
        "mean_pool_only_hits": ablated_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def bootstrap_delta(
    full: np.ndarray,
    ablated: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    delta = full.astype(np.float64) - ablated.astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    batch_size = 100
    for start in range(0, samples, batch_size):
        size = min(batch_size, samples - start)
        draws = rng.integers(0, len(delta), size=(size, len(delta)))
        means[start : start + size] = delta[draws].mean(axis=1)
    return {
        "samples": samples,
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def model_from_config(
    config_path: Path,
    artifact_dir: Path,
) -> tuple[GenerativeRecommender, SemanticIdCatalog, dict[str, object]]:
    config = load_config(config_path)
    data = config["data"]
    generator = config["generator"]
    catalog = SemanticIdCatalog.load(artifact_dir / "catalog.json")
    model = GenerativeRecommender(
        vocab_size=catalog.vocab_size,
        pad_token_id=catalog.special_tokens.pad,
        num_behaviors=data["num_behaviors"],
        num_time_buckets=data["num_time_buckets"],
        model_dim=generator["model_dim"],
        num_heads=generator["num_heads"],
        num_encoder_layers=generator["num_encoder_layers"],
        num_decoder_layers=generator["num_decoder_layers"],
        feedforward_dim=generator["feedforward_dim"],
        dropout=generator["dropout"],
        short_term_window=generator["short_term_window"],
        max_history_length=data["max_history"],
        max_target_length=catalog.sequence_length + 2,
        use_behavior_intensity=generator.get("use_behavior_intensity", False),
        preference_reasoning_mode=generator.get("preference_reasoning_mode", "full"),
        history_forgetting_mode=generator.get("history_forgetting_mode", "none"),
    )
    model.load_state_dict(
        torch.load(artifact_dir / "generator.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    return model, catalog, config


def title_map(path: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            text = str(payload.get("text", ""))
            titles[str(payload["item_id"])] = text.split("[SEP]", 1)[0].strip()
    return titles


def item_title(item_id: str, titles: dict[str, str]) -> str:
    return titles.get(item_id) or item_id


def deterministic_indices(
    full_rows: Sequence[dict[str, object]],
    ablated_rows: Sequence[dict[str, object]],
    count: int,
) -> dict[str, list[int]]:
    if len(full_rows) != len(ablated_rows):
        raise ValueError("paired hit indicators have different lengths")
    groups: dict[str, list[tuple[str, int]]] = {
        "full_only_hit": [],
        "mean_pool_only_hit": [],
        "both_hit": [],
    }
    for left, right in zip(full_rows, ablated_rows, strict=True):
        index = int(left["example_index"])
        if index != int(right["example_index"]) or left["target_item_id"] != right["target_item_id"]:
            raise ValueError("paired hit indicators are not aligned")
        full_hit = indicator_rank(left) is not None
        ablated_hit = indicator_rank(right) is not None
        label = None
        if full_hit and not ablated_hit:
            label = "full_only_hit"
        elif ablated_hit and not full_hit:
            label = "mean_pool_only_hit"
        elif full_hit and ablated_hit:
            label = "both_hit"
        if label is not None:
            digest = hashlib.sha256(
                f"{index}:{left['target_item_id']}:{label}".encode()
            ).hexdigest()
            groups[label].append((digest, index))
    return {
        label: [index for _, index in sorted(rows)[:count]]
        for label, rows in groups.items()
    }


def preference_diagnostics(
    state: PreferenceState,
    row: int,
    example: object,
    titles: dict[str, str],
) -> dict[str, object]:
    history = example.history
    valid_count = len(history)
    raw_weights = (
        state.event_weights[row, :valid_count].detach().cpu()
        if state.event_weights is not None
        else torch.ones(valid_count)
    )
    normalized = raw_weights / raw_weights.sum().clamp_min(1e-12)
    recent_start = max(0, valid_count - 10)
    events = []
    for index, (event, weight) in enumerate(zip(history, normalized.tolist(), strict=True)):
        events.append(
            {
                "history_position": index,
                "item_id": event.item_id,
                "title": item_title(event.item_id, titles),
                "behavior": int(event.behavior),
                "behavior_label": BEHAVIOR_LABELS.get(int(event.behavior), "unknown"),
                "behavior_intensity": float(event.behavior_intensity),
                "time_gap_bucket": int(event.time_gap_bucket),
                "normalized_event_weight": float(weight),
                "inside_short_term_window": index >= recent_start,
            }
        )
    top_events = sorted(
        events,
        key=lambda event: (-float(event["normalized_event_weight"]), -int(event["history_position"])),
    )[:5]
    gate = float(state.short_term_weight[row].mean().item())
    cosine = float(
        F.cosine_similarity(
            state.long_term[row].unsqueeze(0), state.short_term[row].unsqueeze(0)
        ).item()
    )
    decay = (
        float(state.decay_rate[row].item()) if state.decay_rate is not None else None
    )
    return {
        "history_length": valid_count,
        "mean_short_term_gate": gate,
        "long_short_cosine_similarity": cosine,
        "personalized_decay_rate": decay,
        "top_weighted_history_events": top_events,
        "interpretation": (
            ("recent intent receives more gate mass" if gate >= 0.5 else "long-term profile receives more gate mass")
            + "; "
            + ("long- and short-term signals are aligned" if cosine >= 0.5 else "long- and short-term signals conflict")
        ),
    }


def build_cases(
    output_dir: Path,
    seed: int,
    cases_per_outcome: int,
) -> dict[str, object]:
    full_indicator_path = (
        ROOT / f"artifacts/candidate_reranking_mild_v1/test/seed_{seed}/hit_indicators.jsonl"
    )
    mean_indicator_path = output_dir / f"mean_pool/seed_{seed}/hit_indicators/beam_20_diversity_0p0.jsonl"
    full_rows = indicators(full_indicator_path)
    mean_rows = indicators(mean_indicator_path)
    selected = deterministic_indices(full_rows, mean_rows, cases_per_outcome)
    selected_indices = sorted({index for rows in selected.values() for index in rows})
    if not selected_indices:
        raise RuntimeError("no explainability outcomes available")

    full_artifact = ROOT / (
        "artifacts/amazon2023_musical_instruments_long_tail_training_v1/"
        f"formal/token_loss_010/seed_{seed}"
    )
    mean_artifact = output_dir / f"mean_pool/seed_{seed}"
    full_model, full_catalog, full_config = model_from_config(
        ROOT / "configs/v2_final_default.yaml", full_artifact
    )
    mean_model, mean_catalog, _ = model_from_config(
        ROOT / "configs/v2_preference_reasoning_ablation.yaml", mean_artifact
    )
    if full_catalog.item_ids != mean_catalog.item_ids:
        raise ValueError("full and mean-pool catalogs differ")
    data = full_config["data"]
    examples = load_sequence_examples_jsonl(data["test_path"])
    chosen_examples = [examples[index] for index in selected_indices]
    batch = SequenceCollator(full_catalog)(RecommendationDataset(chosen_examples).examples)
    with torch.no_grad():
        full_items, _ = full_model.generate_beam(
            history_tokens=batch["history_tokens"],
            behavior_types=batch["behavior_types"],
            time_gap_buckets=batch["time_gap_buckets"],
            history_mask=batch["history_mask"],
            catalog=full_catalog,
            beam_size=20,
            num_return_sequences=10,
            behavior_intensities=batch["behavior_intensities"],
        )
        mean_items, _ = mean_model.generate_beam(
            history_tokens=batch["history_tokens"],
            behavior_types=batch["behavior_types"],
            time_gap_buckets=batch["time_gap_buckets"],
            history_mask=batch["history_mask"],
            catalog=mean_catalog,
            beam_size=20,
            num_return_sequences=10,
            behavior_intensities=batch["behavior_intensities"],
        )
        _, state = full_model.encode_history(
            batch["history_tokens"],
            batch["behavior_types"],
            batch["time_gap_buckets"],
            batch["history_mask"],
            batch["behavior_intensities"],
        )
    titles = title_map(ROOT / "data/amazon2023/processed/Musical_Instruments/last_out_v2/items.jsonl")
    label_by_index = {
        index: label for label, indices in selected.items() for index in indices
    }
    cases: list[dict[str, object]] = []
    for row, (index, example, full_prediction, mean_prediction) in enumerate(
        zip(selected_indices, chosen_examples, full_items, mean_items, strict=True)
    ):
        target = example.target_item_id
        cases.append(
            {
                "outcome": label_by_index[index],
                "example_index": index,
                "user_hash": hashlib.sha256(example.user_id.encode()).hexdigest()[:16],
                "target": {"item_id": target, "title": item_title(target, titles)},
                "full_reasoning": {
                    "target_rank": full_prediction.index(target) + 1 if target in full_prediction else None,
                    "top_10": [
                        {"item_id": item, "title": item_title(item, titles)}
                        for item in full_prediction
                    ],
                },
                "mean_pool_ablation": {
                    "target_rank": mean_prediction.index(target) + 1 if target in mean_prediction else None,
                    "top_10": [
                        {"item_id": item, "title": item_title(item, titles)}
                        for item in mean_prediction
                    ],
                },
                "preference_reasoning_diagnostics": preference_diagnostics(
                    state, row, example, titles
                ),
                "interpretability_scope": (
                    "Post-hoc inspection of learned gates and event weights; this is a "
                    "model-internal diagnostic, not a causal explanation."
                ),
            }
        )
    payload = {
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "category": "Musical_Instruments",
        "synthetic_data_used": False,
        "seed": seed,
        "case_selection": "deterministic SHA-256 ordering within paired Top-10 outcomes",
        "outcome_counts_selected": {name: len(rows) for name, rows in selected.items()},
        "cases": cases,
    }
    (output_dir / "explainability_cases.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def write_markdown(
    output_dir: Path,
    summary: dict[str, object],
    case_payload: dict[str, object],
) -> None:
    test = summary["test"]
    lines = [
        "# Explicit Preference Reasoning ablation",
        "",
        "- Dataset: real Amazon Reviews 2023 / Musical Instruments.",
        "- Controlled factor: `full` long/short-term reasoning vs `mean_pool`.",
        "- Fixed across arms: BLaIR, RQ-VAE catalog, token-loss 0.10, 10 epochs, Beam-20/Top-10.",
        "- Uncertainty: paired three-seed 95% Student-t CI; Top-10 outcomes also use exact McNemar and user-example bootstrap.",
        "",
        "## Validation results",
        "",
        "| Metric | Full mean | Mean-pool mean | Paired delta (full - ablation), 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        row = summary["validation"]["metrics"][metric]
        delta = row["paired_delta_full_minus_mean_pool"]
        lines.append(
            f"| {metric} | {row['full']['mean']:.6f} | {row['mean_pool']['mean']:.6f} | "
            f"{delta['mean']:+.6f} [{delta['ci95_low']:+.6f}, {delta['ci95_high']:+.6f}] |"
        )
    lines.extend(
        [
            "",
        "## Test results",
        "",
        "| Metric | Full mean | Mean-pool mean | Paired delta (full - ablation), 95% CI |",
        "|---|---:|---:|---:|",
        ]
    )
    for metric in METRICS:
        row = test["metrics"][metric]
        delta = row["paired_delta_full_minus_mean_pool"]
        lines.append(
            f"| {metric} | {row['full']['mean']:.6f} | {row['mean_pool']['mean']:.6f} | "
            f"{delta['mean']:+.6f} [{delta['ci95_low']:+.6f}, {delta['ci95_high']:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "A stable advantage is claimed only when the paired CI excludes zero. "
            "Individual examples below illustrate behavior; they do not replace the aggregate test.",
            "",
            "## Explainability cases",
            "",
        ]
    )
    for case in case_payload["cases"]:
        diagnostic = case["preference_reasoning_diagnostics"]
        lines.extend(
            [
                f"### {case['outcome']} — example {case['example_index']}",
                "",
                f"Target: **{case['target']['title']}** (`{case['target']['item_id']}`)",
                "",
                f"Full rank: {case['full_reasoning']['target_rank']}; mean-pool rank: "
                f"{case['mean_pool_ablation']['target_rank']}. Mean short-term gate: "
                f"{diagnostic['mean_short_term_gate']:.3f}; long/short cosine: "
                f"{diagnostic['long_short_cosine_similarity']:.3f}.",
                "",
                f"Diagnostic: {diagnostic['interpretation']}.",
                "",
                "Highest-weight history events:",
                "",
            ]
        )
        for event in diagnostic["top_weighted_history_events"][:3]:
            lines.append(
                f"- {event['title']} — {event['behavior_label']}, intensity "
                f"{event['behavior_intensity']:.2f}, weight "
                f"{event['normalized_event_weight']:.3f}, recent={event['inside_short_term_window']}"
            )
        lines.append("")
    lines.append(
        "> Gate/event weights are model-internal diagnostics and must not be read as causal attributions."
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.cases_per_outcome <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("case and bootstrap counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed: dict[int, dict[str, object]] = {}
    validation_per_seed: dict[int, dict[str, object]] = {}
    metric_names: tuple[str, ...] | None = None
    paired_tests: dict[int, dict[str, object]] = {}
    for seed in SEEDS:
        full_report_path = ROOT / (
            "artifacts/v2_long_tail_paired_reevaluation/"
            f"token_loss_010/seed_{seed}/test.json"
        )
        mean_report_path = args.output_dir / f"mean_pool/seed_{seed}/beam20_test.json"
        full_metrics = flatten_metrics(result_row(read_json(full_report_path)))
        mean_metrics = flatten_metrics(result_row(read_json(mean_report_path)))
        full_training_path = ROOT / (
            "artifacts/amazon2023_musical_instruments_long_tail_training_v1/"
            f"formal/token_loss_010/seed_{seed}/report.json"
        )
        mean_training_path = args.output_dir / f"mean_pool/seed_{seed}/report.json"
        full_validation = read_json(full_training_path)["validation"]
        mean_validation = read_json(mean_training_path)["validation"]
        validation_per_seed[seed] = {
            "full": {metric: float(full_validation[metric]) for metric in METRICS},
            "mean_pool": {metric: float(mean_validation[metric]) for metric in METRICS},
            "sources": {
                "full": str(full_training_path.resolve()),
                "mean_pool": str(mean_training_path.resolve()),
            },
        }
        metric_names = tuple(full_metrics)
        per_seed[seed] = {
            "full": full_metrics,
            "mean_pool": mean_metrics,
            "delta_full_minus_mean_pool": {
                metric: full_metrics[metric] - mean_metrics[metric]
                for metric in metric_names
            },
            "sources": {
                "full": str(full_report_path.resolve()),
                "mean_pool": str(mean_report_path.resolve()),
            },
        }
        full_indicator_path = (
            ROOT
            / f"artifacts/candidate_reranking_mild_v1/test/seed_{seed}/hit_indicators.jsonl"
        )
        mean_indicator_path = args.output_dir / f"mean_pool/seed_{seed}/hit_indicators/beam_20_diversity_0p0.jsonl"
        full_rows = indicators(full_indicator_path)
        mean_rows = indicators(mean_indicator_path)
        if any(
            int(left["example_index"]) != int(right["example_index"])
            or left["target_item_id"] != right["target_item_id"]
            for left, right in zip(full_rows, mean_rows, strict=True)
        ):
            raise ValueError("paired indicators are not aligned")
        full_hit = np.array(
            [indicator_rank(row) is not None for row in full_rows], dtype=np.int8
        )
        mean_hit = np.array(
            [indicator_rank(row) is not None for row in mean_rows], dtype=np.int8
        )
        paired_tests[seed] = {
            "examples": len(full_hit),
            "mcnemar_top10": exact_mcnemar(full_hit, mean_hit),
            "paired_user_bootstrap_top10": bootstrap_delta(
                full_hit,
                mean_hit,
                samples=args.bootstrap_samples,
                seed=seed,
            ),
        }
    assert metric_names is not None
    test_metrics: dict[str, object] = {}
    validation_metrics: dict[str, object] = {}
    for metric in metric_names:
        full_values = [float(per_seed[seed]["full"][metric]) for seed in SEEDS]
        mean_values = [float(per_seed[seed]["mean_pool"][metric]) for seed in SEEDS]
        deltas = [left - right for left, right in zip(full_values, mean_values, strict=True)]
        interval = mean_confidence_interval(deltas)
        test_metrics[metric] = {
            "full": mean_confidence_interval(full_values),
            "mean_pool": mean_confidence_interval(mean_values),
            "paired_delta_full_minus_mean_pool": interval,
            "stable_full_advantage": interval["ci95_low"] > 0,
            "stable_mean_pool_advantage": interval["ci95_high"] < 0,
        }
    for metric in METRICS:
        full_values = [
            float(validation_per_seed[seed]["full"][metric]) for seed in SEEDS
        ]
        mean_values = [
            float(validation_per_seed[seed]["mean_pool"][metric]) for seed in SEEDS
        ]
        deltas = [left - right for left, right in zip(full_values, mean_values, strict=True)]
        interval = mean_confidence_interval(deltas)
        validation_metrics[metric] = {
            "full": mean_confidence_interval(full_values),
            "mean_pool": mean_confidence_interval(mean_values),
            "paired_delta_full_minus_mean_pool": interval,
            "stable_full_advantage": interval["ci95_low"] > 0,
            "stable_mean_pool_advantage": interval["ci95_high"] < 0,
        }
    summary = {
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "category": "Musical_Instruments",
        "synthetic_data_used": False,
        "seeds": list(SEEDS),
        "controlled_ablation": {
            "full": "learned event attention + long/short gate + fusion MLP",
            "mean_pool": "behavior/recency-weighted mean; no learned attention, gate, or fusion MLP",
            "fixed": [
                "history Transformer",
                "BLaIR embeddings",
                "RQ-VAE Semantic IDs",
                "token-loss alpha=0.10",
                "10 training epochs",
                "Beam-20 / Top-10",
            ],
        },
        "interval": "paired two-sided 95% Student-t CI across three independent seeds",
        "claim_rule": "stable only if the paired CI excludes zero",
        "validation": {"metrics": validation_metrics, "per_seed": validation_per_seed},
        "test": {"metrics": test_metrics, "per_seed": per_seed},
        "paired_top10_tests": paired_tests,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    case_payload = build_cases(args.output_dir, args.case_seed, args.cases_per_outcome)
    write_markdown(args.output_dir, summary, case_payload)
    print(json.dumps({"summary": summary, "case_counts": case_payload["outcome_counts_selected"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
