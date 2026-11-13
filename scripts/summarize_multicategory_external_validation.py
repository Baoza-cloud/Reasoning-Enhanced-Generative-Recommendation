from __future__ import annotations

import argparse
import json
from pathlib import Path

from regenrec.metrics import mean_confidence_interval


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2026, 2027, 2028)
RANKING_METRICS = ("recall_at_5", "ndcg_at_5", "recall_at_10", "ndcg_at_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate frozen cross-category V2 transport experiments"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/multicategory_external_validation_v1"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def beam_result(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    report = read_json(path)
    for result in report["results"]:
        if (
            int(result["beam_size"]) == 20
            and int(result["return_count"]) == 10
            and float(result.get("diversity_weight", 0.0)) == 0.0
        ):
            return result, report
    raise ValueError(f"missing Beam-20/Top-10 result: {path}")


def flatten(
    result: dict[str, object],
    *,
    latency_ms_per_example: float,
) -> dict[str, float]:
    ranking = result["ranking"]
    tail = result["target_groups"]["observed_tail"]
    coverage = result["coverage"]
    tail_coverage = result["strata_coverage"]["strata"]["observed_tail"]
    return {
        **{metric: float(ranking[metric]) for metric in RANKING_METRICS},
        "observed_tail_recall_at_5": float(tail["recall_at_5"]),
        "observed_tail_ndcg_at_5": float(tail["ndcg_at_5"]),
        "observed_tail_recall_at_10": float(tail["recall_at_10"]),
        "observed_tail_ndcg_at_10": float(tail["ndcg_at_10"]),
        "catalog_coverage": float(coverage["catalog_coverage"]),
        "observed_tail_catalog_coverage": float(tail_coverage["catalog_coverage"]),
        "observed_tail_recommendation_share": float(tail_coverage["recommendation_share"]),
        "latency_ms_per_example": latency_ms_per_example,
    }


def beam_metrics(path: Path) -> dict[str, float]:
    result, _ = beam_result(path)
    return flatten(
        result,
        latency_ms_per_example=float(result["efficiency"]["milliseconds_per_example"]),
    )


def candidate_variant(
    path: Path,
    *,
    label: str | None = None,
    selected: bool = False,
) -> dict[str, float]:
    report = read_json(path)
    if selected:
        result = report["selected"]
    else:
        matches = [
            row for row in report["evaluation_ablations"] if row.get("label") == label
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one {label} variant in {path}, got {len(matches)}")
        result = matches[0]
    examples = int(report["examples"])
    rerank_ms = float(result["rerank_efficiency"]["milliseconds_per_example"])
    if label == "generator_only":
        latency = 1000.0 * float(report["timing"]["generator_seconds"]) / examples + rerank_ms
    else:
        latency = float(report["timing"]["fixed_pipeline_milliseconds_per_example"]) + rerank_ms
    return flatten(result["metrics"], latency_ms_per_example=latency)


def musical_rows(seed: int) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    original_path = ROOT / (
        "artifacts/v2_long_tail_paired_reevaluation/"
        f"original_v2/seed_{seed}/test.json"
    )
    token_path = ROOT / (
        "artifacts/v2_long_tail_paired_reevaluation/"
        f"token_loss_010/seed_{seed}/test.json"
    )
    mild_path = ROOT / f"artifacts/candidate_reranking_mild_v1/test/seed_{seed}/report.json"
    aggressive_path = ROOT / f"artifacts/candidate_reranking_v1/formal/test/seed_{seed}/report.json"
    return (
        {
            "original_v2": beam_metrics(original_path),
            "token_loss_010": beam_metrics(token_path),
            "frozen_mild": candidate_variant(mild_path, selected=True),
            "frozen_aggressive": candidate_variant(aggressive_path, selected=True),
        },
        {
            "original_v2": str(original_path.resolve()),
            "token_loss_010": str(token_path.resolve()),
            "frozen_mild": str(mild_path.resolve()),
            "frozen_aggressive": str(aggressive_path.resolve()),
        },
    )


def industrial_rows(
    seed: int, output_dir: Path
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    original_path = output_dir / f"Industrial_and_Scientific/original_v2/seed_{seed}/beam20_test.json"
    candidate_path = output_dir / (
        f"Industrial_and_Scientific/token_loss_010/seed_{seed}/candidate_reranking_test/report.json"
    )
    return (
        {
            "original_v2": beam_metrics(original_path),
            "token_loss_010": candidate_variant(candidate_path, label="generator_only"),
            "frozen_mild": candidate_variant(candidate_path, label="frozen_mild"),
            "frozen_aggressive": candidate_variant(candidate_path, label="frozen_aggressive"),
        },
        {
            "original_v2": str(original_path.resolve()),
            "token_loss_010": str(candidate_path.resolve()),
            "frozen_mild": str(candidate_path.resolve()),
            "frozen_aggressive": str(candidate_path.resolve()),
        },
    )


def aggregate(
    rows: dict[int, dict[str, dict[str, float]]]
) -> dict[str, dict[str, dict[str, float | int]]]:
    variants = tuple(next(iter(rows.values())))
    metrics = tuple(next(iter(next(iter(rows.values())).values())))
    return {
        variant: {
            metric: mean_confidence_interval(
                [rows[seed][variant][metric] for seed in SEEDS]
            )
            for metric in metrics
        }
        for variant in variants
    }


def paired_comparison(
    rows: dict[int, dict[str, dict[str, float]]],
    left: str,
    right: str,
) -> dict[str, object]:
    metrics = tuple(rows[SEEDS[0]][left])
    output: dict[str, object] = {
        "comparison": f"{left} minus {right}",
        "interval": "paired two-sided 95% Student-t CI across three seeds",
        "metrics": {},
    }
    for metric in metrics:
        values = [
            rows[seed][left][metric] - rows[seed][right][metric] for seed in SEEDS
        ]
        interval = mean_confidence_interval(values)
        output["metrics"][metric] = {
            **interval,
            "stable_positive": interval["ci95_low"] > 0.0,
            "stable_negative": interval["ci95_high"] < 0.0,
        }
    return output


def sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def write_report(output_dir: Path, summary: dict[str, object]) -> None:
    lines = [
        "# V2 multi-category external validation",
        "",
        "The Industrial & Scientific test is a transport test: all token-loss and reranking "
        "hyperparameters were frozen from Musical Instruments before Industrial test scoring.",
        "",
        "| Category | Variant | Recall@10 | NDCG@10 | Observed-Tail Recall@10 | Catalog coverage | Tail coverage | Latency ms/example |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, payload in summary["categories"].items():
        for variant, metrics in payload["aggregate"].items():
            lines.append(
                f"| {category} | {variant} | {metrics['recall_at_10']['mean']:.6f} | "
                f"{metrics['ndcg_at_10']['mean']:.6f} | "
                f"{metrics['observed_tail_recall_at_10']['mean']:.6f} | "
                f"{metrics['catalog_coverage']['mean']:.4f} | "
                f"{metrics['observed_tail_catalog_coverage']['mean']:.4f} | "
                f"{metrics['latency_ms_per_example']['mean']:.2f} |"
            )
    conclusion = summary["transport_conclusion"]
    lines.extend(
        [
            "",
            "## Transport conclusion",
            "",
            f"- Verdict: **{conclusion['verdict']}**.",
            f"- {conclusion['statement']}",
            "- A stable cross-category gain is claimed only when the Industrial paired 95% CI excludes zero.",
            "- Cold-start items remain separate from Observed Tail and are not counted as learned long-tail performance.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    categories: dict[str, object] = {}
    row_sets: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    for category, loader in (
        ("Musical_Instruments", lambda seed: musical_rows(seed)),
        ("Industrial_and_Scientific", lambda seed: industrial_rows(seed, output_dir)),
    ):
        rows: dict[int, dict[str, dict[str, float]]] = {}
        sources: dict[int, dict[str, str]] = {}
        for seed in SEEDS:
            rows[seed], sources[seed] = loader(seed)
        row_sets[category] = rows
        categories[category] = {
            "per_seed": rows,
            "aggregate": aggregate(rows),
            "paired_comparisons": {
                "token_vs_original": paired_comparison(rows, "token_loss_010", "original_v2"),
                "mild_vs_token": paired_comparison(rows, "frozen_mild", "token_loss_010"),
                "aggressive_vs_token": paired_comparison(
                    rows, "frozen_aggressive", "token_loss_010"
                ),
            },
            "sources": sources,
        }

    musical = categories["Musical_Instruments"]["paired_comparisons"]
    industrial = categories["Industrial_and_Scientific"]["paired_comparisons"]
    effects = {
        "token_recall_at_10": ("token_vs_original", "recall_at_10"),
        "token_tail_recall_at_10": (
            "token_vs_original",
            "observed_tail_recall_at_10",
        ),
        "mild_recall_at_10": ("mild_vs_token", "recall_at_10"),
        "mild_tail_recall_at_10": (
            "mild_vs_token",
            "observed_tail_recall_at_10",
        ),
    }
    transport_checks: dict[str, object] = {}
    stable_external = 0
    for name, (comparison, metric) in effects.items():
        left = musical[comparison]["metrics"][metric]
        right = industrial[comparison]["metrics"][metric]
        direction_agrees = sign(float(left["mean"])) == sign(float(right["mean"]))
        external_stable = bool(right["stable_positive"] or right["stable_negative"])
        stable_external += int(external_stable)
        transport_checks[name] = {
            "musical_mean_delta": left["mean"],
            "industrial_mean_delta": right["mean"],
            "direction_agrees": direction_agrees,
            "industrial_ci95": [right["ci95_low"], right["ci95_high"]],
            "industrial_stable": external_stable,
        }
    agreements = sum(bool(row["direction_agrees"]) for row in transport_checks.values())
    if agreements == len(transport_checks) and stable_external == len(transport_checks):
        verdict = "stable_cross_category_transport"
        statement = "All pre-specified effects keep direction and exclude zero on Industrial."
    elif agreements == len(transport_checks):
        verdict = "directional_transport_not_yet_stable"
        statement = (
            "The effects are not unique to Musical in direction, but Industrial uncertainty "
            "prevents a stable universal-gain claim."
        )
    else:
        verdict = "category_dependent"
        statement = (
            "At least one pre-specified effect reverses direction on Industrial, so the V2 "
            "long-tail conclusion is category-dependent."
        )
    summary = {
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "synthetic_data_used": False,
        "seeds": list(SEEDS),
        "protocol": {
            "source_category_for_selection": "Musical_Instruments validation",
            "external_category_used_for_selection": False,
            "external_test_used_for_selection": False,
            "decoder": "Beam-20 / Top-10",
            "popularity_groups": "Head 20%, Torso 30%, Observed Tail 50% among positive-frequency items; Cold-start separate",
            "frozen_mild": {
                "blair_weight": 0.25,
                "frequency_alpha": 0.10,
                "quota_profile": "tail_20",
                "quota_strength": 0.01,
            },
            "frozen_aggressive": {
                "blair_weight": 0.25,
                "frequency_alpha": 0.20,
                "quota_profile": "balanced",
                "quota_strength": 0.05,
            },
        },
        "categories": categories,
        "transport_conclusion": {
            "verdict": verdict,
            "statement": statement,
            "checks": transport_checks,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output_dir, summary)
    print(json.dumps(summary["transport_conclusion"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
