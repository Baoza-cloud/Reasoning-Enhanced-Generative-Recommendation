from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from regenrec.metrics import mean_confidence_interval


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2026, 2027, 2028)
CONFIG = Path("configs/v2_4_adaptive_forgetting.yaml")
DYNAMIC_ROOT = Path("artifacts/v2_4_adaptive_forgetting")
SCORER_ROOT = Path("artifacts/v2_4_hybrid_scorer_multiseed")
FROZEN_DEFAULT_ROOT = Path("artifacts/candidate_reranking_mild_v1/test")
DEFAULT_ARTIFACT_ROOT = Path(
    "artifacts/amazon2023_musical_instruments_long_tail_training_v1/"
    "formal/token_loss_010"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot, test-only confirmation of frozen V2.4 adaptive forgetting and "
            "validation-fitted hybrid scorers"
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v2_4_test_confirmation"),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def run(command: list[str], log_path: Path, threads: int) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = str(threads)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def validate_frozen_inputs(seed: int) -> dict[str, str]:
    default_report_path = ROOT / FROZEN_DEFAULT_ROOT / f"seed_{seed}" / "report.json"
    default_indicator_path = (
        ROOT / FROZEN_DEFAULT_ROOT / f"seed_{seed}" / "hit_indicators.jsonl"
    )
    default_model_path = ROOT / DEFAULT_ARTIFACT_ROOT / f"seed_{seed}" / "generator.pt"
    dynamic_dir = ROOT / DYNAMIC_ROOT / f"seed_{seed}"
    dynamic_model_path = dynamic_dir / "generator.pt"
    dynamic_catalog_path = dynamic_dir / "catalog.json"
    scorer_path = ROOT / SCORER_ROOT / f"seed_{seed}" / "hybrid_scorer.json"
    required = (
        default_report_path,
        default_indicator_path,
        default_model_path,
        dynamic_model_path,
        dynamic_catalog_path,
        scorer_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"seed {seed} missing frozen inputs: {missing}")
    default_report = json.loads(default_report_path.read_text(encoding="utf-8"))
    if default_report.get("split") != "test" or int(default_report["seed"]) != seed:
        raise ValueError(f"invalid frozen default Test report: {default_report_path}")
    default_model_sha = sha256_file(default_model_path)
    if default_report["provenance"]["model_sha256"] != default_model_sha:
        raise ValueError(f"default checkpoint provenance mismatch: seed {seed}")
    return {
        "default_report": str(default_report_path.resolve()),
        "default_indicators": str(default_indicator_path.resolve()),
        "default_model_sha256": default_model_sha,
        "dynamic_model_sha256": sha256_file(dynamic_model_path),
        "dynamic_catalog_sha256": sha256_file(dynamic_catalog_path),
        "hybrid_scorer_sha256": sha256_file(scorer_path),
    }


def validate_test_report(path: Path, seed: int, provenance: dict[str, str]) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if int(report.get("seed", -1)) != seed or report.get("split") != "test":
        raise ValueError(f"wrong Test report identity: {path}")
    if report.get("mode") != "evaluate":
        raise ValueError(f"Test report was not produced in evaluate mode: {path}")
    hybrid = report.get("hybrid_scorer", {})
    if hybrid.get("mode") != "evaluate":
        raise ValueError(f"frozen hybrid scorer was not loaded for Test: {path}")
    if hybrid.get("test_labels_used_for_fitting") is not False:
        raise ValueError(f"Test labels leaked into hybrid fitting: {path}")
    if report["provenance"]["model_sha256"] != provenance["dynamic_model_sha256"]:
        raise ValueError(f"dynamic checkpoint provenance mismatch: seed {seed}")
    if report.get("synthetic_data_used") is not False:
        raise ValueError(f"non-real data report rejected: {path}")
    indicator_path = path.parent / "hit_indicators.jsonl"
    if not indicator_path.is_file():
        raise FileNotFoundError(indicator_path)
    return report


def metric_values(metrics: dict[str, object], latency: float) -> dict[str, float]:
    target_groups = metrics["target_groups"]
    strata = metrics["strata_coverage"]["strata"]
    return {
        "recall_at_5": float(metrics["ranking"]["recall_at_5"]),
        "ndcg_at_5": float(metrics["ranking"]["ndcg_at_5"]),
        "recall_at_10": float(metrics["ranking"]["recall_at_10"]),
        "ndcg_at_10": float(metrics["ranking"]["ndcg_at_10"]),
        "observed_tail_recall_at_5": float(
            target_groups["observed_tail"]["recall_at_5"]
        ),
        "observed_tail_recall_at_10": float(
            target_groups["observed_tail"]["recall_at_10"]
        ),
        "catalog_coverage": float(metrics["coverage"]["catalog_coverage"]),
        "observed_tail_catalog_coverage": float(
            strata["observed_tail"]["catalog_coverage"]
        ),
        "observed_tail_recommendation_share": float(
            strata["observed_tail"]["recommendation_share"]
        ),
        "latency_ms_per_example": latency,
    }


def report_model_values(
    default_report: dict[str, object],
    confirmation_report: dict[str, object],
) -> dict[str, dict[str, float]]:
    default_examples = int(default_report["examples"])
    confirmation_examples = int(confirmation_report["examples"])
    if default_examples != confirmation_examples:
        raise ValueError("default and V2.4 Test reports have different example counts")
    default_latency = (
        1000.0 * float(default_report["timing"]["generator_seconds"]) / default_examples
        + float(default_report["baseline"]["rerank_efficiency"]["milliseconds_per_example"])
    )
    adaptive_latency = (
        1000.0
        * float(confirmation_report["timing"]["generator_seconds"])
        / confirmation_examples
        + float(
            confirmation_report["baseline"]["rerank_efficiency"][
                "milliseconds_per_example"
            ]
        )
    )
    hybrid_latency = float(
        confirmation_report["timing"]["selected_total_milliseconds_per_example"]
    )
    return {
        "v2_default_token_loss_010": metric_values(
            default_report["baseline"]["metrics"], default_latency
        ),
        "adaptive_forgetting": metric_values(
            confirmation_report["baseline"]["metrics"], adaptive_latency
        ),
        "learned_hybrid": metric_values(
            confirmation_report["selected"]["metrics"], hybrid_latency
        ),
    }


def summarize(
    output_dir: Path, provenance: dict[int, dict[str, str]]
) -> dict[str, object]:
    values: dict[int, dict[str, dict[str, float]]] = {}
    per_seed: dict[str, object] = {}
    for seed in SEEDS:
        default_report_path = ROOT / FROZEN_DEFAULT_ROOT / f"seed_{seed}" / "report.json"
        confirmation_report_path = output_dir / f"seed_{seed}" / "report.json"
        default_report = json.loads(default_report_path.read_text(encoding="utf-8"))
        confirmation_report = validate_test_report(
            confirmation_report_path, seed, provenance[seed]
        )
        values[seed] = report_model_values(default_report, confirmation_report)
        per_seed[str(seed)] = {
            "models": values[seed],
            "default_report": str(default_report_path.resolve()),
            "confirmation_report": str(confirmation_report_path.resolve()),
            "provenance": provenance[seed],
        }
    models = tuple(values[SEEDS[0]])
    metrics = tuple(values[SEEDS[0]][models[0]])
    aggregate_models = {
        model: {
            metric: mean_confidence_interval(
                [values[seed][model][metric] for seed in SEEDS]
            )
            for metric in metrics
        }
        for model in models
    }
    comparisons: dict[str, object] = {}
    for label, treatment in (
        ("adaptive_minus_v2_default", "adaptive_forgetting"),
        ("hybrid_minus_v2_default", "learned_hybrid"),
        ("hybrid_minus_adaptive", "learned_hybrid"),
    ):
        control = (
            "adaptive_forgetting"
            if label == "hybrid_minus_adaptive"
            else "v2_default_token_loss_010"
        )
        comparisons[label] = {
            metric: mean_confidence_interval(
                [
                    values[seed][treatment][metric]
                    - values[seed][control][metric]
                    for seed in SEEDS
                ]
            )
            for metric in metrics
        }
    default_r10 = aggregate_models["v2_default_token_loss_010"]["recall_at_10"]["mean"]
    decision = {
        "pre_test_decision_frozen": True,
        "default_remains": "v2_default_token_loss_010",
        "test_used_for_training_or_selection": False,
        "adaptive_recall_at_10_retention": (
            aggregate_models["adaptive_forgetting"]["recall_at_10"]["mean"]
            / default_r10
        ),
        "hybrid_recall_at_10_retention": (
            aggregate_models["learned_hybrid"]["recall_at_10"]["mean"] / default_r10
        ),
        "interpretation": (
            "Post-freeze one-shot confirmation only. Test outcomes do not trigger "
            "retraining, reselection, or a change to the frozen V2 default."
        ),
    }
    return {
        "status": "completed",
        "created_at": utc_now(),
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "category": "Musical_Instruments",
        "split": "test",
        "synthetic_data_used": False,
        "seeds": list(SEEDS),
        "protocol": {
            "beam_size": 20,
            "generator_candidates": 20,
            "blair_observed_tail_candidates": 50,
            "top_k": 10,
            "adaptive_checkpoint": "frozen Validation-trained generator",
            "hybrid_scorer": "frozen Validation-fit scorer",
            "test_labels_used_for_selection": False,
        },
        "interval": "paired two-sided 95% Student-t CI across three seeds",
        "per_seed": per_seed,
        "models": aggregate_models,
        "comparisons": comparisons,
        "decision": decision,
    }


def load_indicators(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def exact_mcnemar(control: np.ndarray, treatment: np.ndarray) -> dict[str, float | int]:
    gains = int(np.sum((control == 0) & (treatment == 1)))
    losses = int(np.sum((control == 1) & (treatment == 0)))
    discordant = gains + losses
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(gains, losses)
        # Compute 2 * P[Binomial(discordant, 0.5) <= lower] in log space.
        # Directly converting the exact integer combination sum to float overflows
        # for full Test sets with thousands of discordant user outcomes.
        log_probabilities = [
            math.lgamma(discordant + 1)
            - math.lgamma(index + 1)
            - math.lgamma(discordant - index + 1)
            - discordant * math.log(2.0)
            for index in range(lower + 1)
        ]
        maximum = max(log_probabilities)
        log_tail = maximum + math.log(
            sum(math.exp(value - maximum) for value in log_probabilities)
        )
        p_value = min(1.0, 2.0 * math.exp(log_tail))
    return {
        "treatment_only_hits": gains,
        "control_only_hits": losses,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
    }


def bootstrap_ci(
    differences: np.ndarray, *, samples: int, seed: int
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    # A paired binary-outcome delta can only be -1, 0, or +1. Drawing the
    # bootstrap category counts from their multinomial distribution is exactly
    # equivalent to materializing samples * users indices, while avoiding
    # billions of temporary integers on the full Test set.
    categories, counts = np.unique(differences, return_counts=True)
    draws = rng.multinomial(
        len(differences), counts.astype(np.float64) / len(differences), size=samples
    )
    values = draws @ categories.astype(np.float64) / len(differences)
    return {
        "samples": samples,
        "mean_delta": float(differences.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def paired_tests(output_dir: Path, bootstrap_samples: int) -> dict[str, object]:
    output: dict[str, object] = {
        "bootstrap_samples": bootstrap_samples,
        "unit": "one leave-last-out Test example per user",
        "per_seed": {},
    }
    for seed in SEEDS:
        default_path = ROOT / FROZEN_DEFAULT_ROOT / f"seed_{seed}" / "hit_indicators.jsonl"
        confirmation_path = output_dir / f"seed_{seed}" / "hit_indicators.jsonl"
        default_rows = load_indicators(default_path)
        confirmation_rows = load_indicators(confirmation_path)
        if len(default_rows) != len(confirmation_rows):
            raise ValueError(f"indicator count mismatch for seed {seed}")
        for default_row, confirmation_row in zip(
            default_rows, confirmation_rows, strict=True
        ):
            identity = ("example_index", "user_hash", "target_item_id")
            if any(default_row[key] != confirmation_row[key] for key in identity):
                raise ValueError(f"indicator alignment failed for seed {seed}")
        groups = {
            "overall": np.ones(len(default_rows), dtype=bool),
            "observed_tail": np.asarray(
                [row["target_group"] == "observed_tail" for row in default_rows],
                dtype=bool,
            ),
        }
        comparisons = {
            "adaptive_vs_v2_default": (
                [row["baseline_rank"] for row in default_rows],
                [row["baseline_rank"] for row in confirmation_rows],
            ),
            "hybrid_vs_v2_default": (
                [row["baseline_rank"] for row in default_rows],
                [row["selected_rank"] for row in confirmation_rows],
            ),
        }
        seed_result: dict[str, object] = {}
        for comparison_index, (name, (control_ranks, treatment_ranks)) in enumerate(
            comparisons.items()
        ):
            comparison_result: dict[str, object] = {}
            for group_index, (group_name, keep) in enumerate(groups.items()):
                group_result: dict[str, object] = {}
                for cutoff in (5, 10):
                    control = np.asarray(
                        [rank is not None and int(rank) <= cutoff for rank in control_ranks],
                        dtype=np.int8,
                    )[keep]
                    treatment = np.asarray(
                        [
                            rank is not None and int(rank) <= cutoff
                            for rank in treatment_ranks
                        ],
                        dtype=np.int8,
                    )[keep]
                    differences = treatment.astype(np.float64) - control.astype(np.float64)
                    group_result[f"hit_at_{cutoff}"] = {
                        "examples": int(len(differences)),
                        "mcnemar": exact_mcnemar(control, treatment),
                        "paired_user_bootstrap": bootstrap_ci(
                            differences,
                            samples=bootstrap_samples,
                            seed=(
                                seed
                                + 100000 * comparison_index
                                + 10000 * group_index
                                + 100 * cutoff
                            ),
                        ),
                    }
                comparison_result[group_name] = group_result
            seed_result[name] = comparison_result
        output["per_seed"][str(seed)] = seed_result
    return output


def write_report(output_dir: Path, summary: dict[str, object]) -> None:
    models = summary["models"]
    comparisons = summary["comparisons"]
    lines = [
        "# V2.4 冻结后一次性 Test 确认",
        "",
        "本评测只加载冻结 checkpoint 与 Validation 拟合的 scorer；Test 不参与训练、选参或重选。",
        "原 V2 默认方案在评测前已经冻结，结果不会触发重新训练或改变默认方案。",
        "",
        "## 三种子 Test 均值",
        "",
        "| 模型 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Tail R@10 | Catalog coverage | Tail coverage | Tail share | ms/example |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "v2_default_token_loss_010": "原 V2 默认",
        "adaptive_forgetting": "动态遗忘",
        "learned_hybrid": "Learned hybrid",
    }
    for model, label in labels.items():
        row = models[model]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                f"{row[name]['mean']:.6f}"
                for name in (
                    "recall_at_5",
                    "ndcg_at_5",
                    "recall_at_10",
                    "ndcg_at_10",
                    "observed_tail_recall_at_10",
                    "catalog_coverage",
                    "observed_tail_catalog_coverage",
                    "observed_tail_recommendation_share",
                    "latency_ms_per_example",
                )
            )
            + " |"
        )
    hybrid_latency_delta = comparisons["hybrid_minus_adaptive"][
        "latency_ms_per_example"
    ]
    lines.extend(
        [
            "",
            "延迟说明：原 V2 数值来自此前冻结运行，V2.4 数值来自本次三种子并行运行，绝对值不作跨运行机制归因。"
            f"同一次 V2.4 运行内，hybrid 相对动态遗忘增加 {hybrid_latency_delta['mean']:.3f} ms/example。",
            "",
            "## 配对 Recall@10 增量",
            "",
            "| 比较 | 均值 | 95% Student-t CI |",
            "|---|---:|---:|",
        ]
    )
    for name in ("adaptive_minus_v2_default", "hybrid_minus_v2_default"):
        row = comparisons[name]["recall_at_10"]
        lines.append(
            f"| {name} | {row['mean']:+.6f} | "
            f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "完整指标见 `summary.json`；逐种子 McNemar 与用户 bootstrap 见 `paired_tests.json`。",
            "",
            "## 冻结解释",
            "",
            "这是 post-freeze confirmation，不是第二轮模型选择。原 V2 "
            "`token_loss_010 + Beam-20/Top-10` 保持最终默认方案。",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.threads_per_worker <= 0:
        raise ValueError("worker counts must be positive")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    provenance = {seed: validate_frozen_inputs(seed) for seed in SEEDS}
    write_json(
        output_dir / "protocol.json",
        {
            "status": "frozen_before_test",
            "created_at": utc_now(),
            "seeds": list(SEEDS),
            "split": "test",
            "test_used_for_selection": False,
            "beam_size": 20,
            "candidate_count": 20,
            "retrieval_count": 50,
            "top_k": 10,
            "provenance": provenance,
        },
    )
    completed: list[str] = []
    tasks: list[tuple[int, list[str], Path, Path]] = []
    for seed in SEEDS:
        seed_dir = output_dir / f"seed_{seed}"
        report_path = seed_dir / "report.json"
        step = f"evaluate_test_seed_{seed}"
        if report_path.is_file():
            validate_test_report(report_path, seed, provenance[seed])
            completed.append(step)
            continue
        scorer_path = ROOT / SCORER_ROOT / f"seed_{seed}" / "hybrid_scorer.json"
        tasks.append(
            (
                seed,
                [
                    sys.executable,
                    "scripts/analyze_candidate_reranking.py",
                    "--config",
                    str(CONFIG),
                    "--artifact-dir",
                    str(ROOT / DYNAMIC_ROOT / f"seed_{seed}"),
                    "--seed",
                    str(seed),
                    "--split",
                    "test",
                    "--mode",
                    "evaluate",
                    "--beam-size",
                    "20",
                    "--candidate-count",
                    "20",
                    "--retrieval-count",
                    "50",
                    "--top-k",
                    "10",
                    "--hybrid-scorer-mode",
                    "evaluate",
                    "--hybrid-scorer-path",
                    str(scorer_path),
                    "--output-dir",
                    str(seed_dir),
                ],
                output_dir / "logs" / f"{step}.log",
                report_path,
            )
        )
    try:
        write_json(
            state_path,
            {
                "updated_at": utc_now(),
                "status": "running",
                "phase": "frozen_test_evaluation",
                "completed": completed,
                "pending": [f"evaluate_test_seed_{seed}" for seed, *_ in tasks],
            },
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run, command, log_path, args.threads_per_worker): (
                    seed,
                    report_path,
                )
                for seed, command, log_path, report_path in tasks
            }
            for future in as_completed(futures):
                seed, report_path = futures[future]
                future.result()
                validate_test_report(report_path, seed, provenance[seed])
                completed.append(f"evaluate_test_seed_{seed}")
                write_json(
                    state_path,
                    {
                        "updated_at": utc_now(),
                        "status": "running",
                        "phase": "frozen_test_evaluation",
                        "completed": completed,
                    },
                )
        write_json(
            state_path,
            {
                "updated_at": utc_now(),
                "status": "running",
                "phase": "summarize_and_paired_tests",
                "completed": completed,
            },
        )
        summary = summarize(output_dir, provenance)
        write_json(output_dir / "summary.json", summary)
        tests = paired_tests(output_dir, args.bootstrap_samples)
        write_json(output_dir / "paired_tests.json", tests)
        write_report(output_dir, summary)
        write_json(
            state_path,
            {
                "updated_at": utc_now(),
                "status": "completed",
                "phase": None,
                "completed": completed + ["summarize", "paired_tests", "report"],
            },
        )
    except Exception as error:
        write_json(
            state_path,
            {
                "updated_at": utc_now(),
                "status": "failed",
                "phase": "frozen_test_evaluation",
                "completed": completed,
                "error": repr(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
