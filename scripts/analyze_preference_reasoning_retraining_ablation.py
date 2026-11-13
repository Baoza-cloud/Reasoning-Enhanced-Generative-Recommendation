from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from regenrec.config import load_config
from regenrec.data import load_sequence_examples_jsonl
from regenrec.metrics import mean_confidence_interval


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2026, 2027, 2028)
MODES = ("full", "fixed_0p5_gate", "no_event_attention", "no_fusion_mlp")
ABLATIONS = MODES[1:]
GROUPS = ("all", "le_10", "11_20", "gt_20", "gt_10")
METRICS = ("recall_at_5", "ndcg_at_5", "recall_at_10", "ndcg_at_10")
MODE_LABELS = {
    "full": "Learned gate (full)",
    "fixed_0p5_gate": "Fixed 0.5 gate",
    "no_event_attention": "No event attention",
    "no_fusion_mlp": "No fusion MLP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize independently retrained Preference Reasoning ablations"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/preference_reasoning_retraining_ablation_v1"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def full_artifact(seed: int) -> Path:
    return ROOT / (
        "artifacts/amazon2023_musical_instruments_long_tail_training_v1/"
        f"formal/token_loss_010/seed_{seed}"
    )


def full_indicator(seed: int) -> Path:
    return ROOT / (
        "artifacts/candidate_reranking_mild_v1/test/"
        f"seed_{seed}/hit_indicators.jsonl"
    )


def variant_artifact(output_dir: Path, mode: str, seed: int) -> Path:
    return output_dir / mode / f"seed_{seed}"


def load_ranks(path: Path, field: str, targets: list[str]) -> np.ndarray:
    ranks = np.zeros(len(targets), dtype=np.int16)
    seen = np.zeros(len(targets), dtype=np.bool_)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["example_index"])
            if index < 0 or index >= len(targets):
                raise ValueError(f"example index {index} outside expected range in {path}")
            if str(row["target_item_id"]) != targets[index]:
                raise ValueError(f"target mismatch at example {index} in {path}")
            rank = row.get(field)
            ranks[index] = 0 if rank is None else int(rank)
            seen[index] = True
    if not bool(seen.all()):
        raise ValueError(f"incomplete hit indicators: {path}")
    return ranks


def group_indices(lengths: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all": np.arange(len(lengths)),
        "le_10": np.flatnonzero(lengths <= 10),
        "11_20": np.flatnonzero((lengths >= 11) & (lengths <= 20)),
        "gt_20": np.flatnonzero(lengths > 20),
        "gt_10": np.flatnonzero(lengths > 10),
    }


def ranking_metrics(ranks: np.ndarray) -> dict[str, float]:
    output: dict[str, float] = {}
    for k in (5, 10):
        hits = (ranks > 0) & (ranks <= k)
        discounts = np.zeros(len(ranks), dtype=np.float64)
        discounts[hits] = 1.0 / np.log2(ranks[hits].astype(np.float64) + 1.0)
        output[f"recall_at_{k}"] = float(hits.mean())
        output[f"ndcg_at_{k}"] = float(discounts.mean())
    return output


def exact_mcnemar(full_ranks: np.ndarray, ablated_ranks: np.ndarray) -> dict[str, float | int]:
    full = (full_ranks > 0) & (full_ranks <= 10)
    ablated = (ablated_ranks > 0) & (ablated_ranks <= 10)
    full_only = int(np.sum(full & ~ablated))
    ablated_only = int(np.sum(~full & ablated))
    discordant = full_only + ablated_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(full_only, ablated_only)
        tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / 2**discordant
        p_value = min(1.0, 2.0 * tail)
    return {
        "full_only_hits": full_only,
        "ablation_only_hits": ablated_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def categorical_bootstrap_delta(
    full_ranks: np.ndarray,
    ablated_ranks: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap paired Top-10 hit deltas without materializing index matrices."""
    full = ((full_ranks > 0) & (full_ranks <= 10)).astype(np.int8)
    ablated = ((ablated_ranks > 0) & (ablated_ranks <= 10)).astype(np.int8)
    delta = full - ablated
    values = np.asarray((-1, 0, 1), dtype=np.int8)
    counts = np.asarray([np.sum(delta == value) for value in values], dtype=np.int64)
    probabilities = counts / counts.sum()
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(delta), probabilities, size=samples)
    means = (draws @ values.astype(np.float64)) / len(delta)
    return {
        "samples": samples,
        "examples": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def validation_metrics(path: Path, expected_mode: str, seed: int) -> dict[str, float]:
    report = read_json(path)
    if int(report["seed"]) != seed:
        raise ValueError(f"seed mismatch in {path}")
    actual_mode = str(report["generator_config"].get("preference_reasoning_mode", "full"))
    if actual_mode != expected_mode:
        raise ValueError(
            f"reasoning mode mismatch in {path}: expected {expected_mode}, got {actual_mode}"
        )
    audit = report.get("data_audit", {})
    if audit.get("status") != "passed" or audit.get("synthetic_data_used") is not False:
        raise ValueError(f"real-data audit failed in {path}")
    validation = report["validation"]
    return {metric: float(validation[metric]) for metric in METRICS}


def result_latency(path: Path) -> float:
    report = read_json(path)
    for row in report["results"]:
        if (
            int(row["beam_size"]) == 20
            and int(row["return_count"]) == 10
            and float(row.get("diversity_weight", 0.0)) == 0.0
        ):
            return float(row["efficiency"]["milliseconds_per_example"])
    raise ValueError(f"missing Beam-20/Top-10 result in {path}")


def load_all_ranks(
    output_dir: Path, targets: list[str]
) -> dict[int, dict[str, np.ndarray]]:
    output: dict[int, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        seed_rows = {
            "full": load_ranks(full_indicator(seed), "baseline_rank", targets)
        }
        for mode in ABLATIONS:
            path = (
                variant_artifact(output_dir, mode, seed)
                / "hit_indicators/beam_20_diversity_0p0.jsonl"
            )
            seed_rows[mode] = load_ranks(path, "top_k_rank", targets)
        output[seed] = seed_rows
    return output


def ci_text(row: dict[str, object], *, signed: bool = False) -> str:
    if signed:
        return (
            f"{float(row['mean']):+.6f} "
            f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}]"
        )
    return (
        f"{float(row['mean']):.6f} "
        f"[{float(row['ci95_low']):.6f}, {float(row['ci95_high']):.6f}]"
    )


def write_report(summary: dict[str, object], path: Path) -> None:
    validation = summary["aggregate"]["validation"]
    test = summary["aggregate"]["test"]
    lines = [
        "# Preference Reasoning 细粒度结构独立重训消融",
        "",
        "## 设计",
        "",
        "- 数据：真实 Amazon Reviews 2023 / Musical Instruments；无合成数据。",
        "- full 及三个变体均为 seed 2026/2027/2028 的独立 10-epoch 训练；变体不加载 full 权重。",
        "- BLaIR、RQ-VAE Semantic ID、token-loss 0.10、优化器和 Beam-20/Top-10 协议保持一致。",
        "- 结构变体预先指定；训练阶段只评估 Validation，Test 在配置冻结后统一评估一次。",
        "- 区间为相同 seed 配对差值的双侧 95% Student-t CI；逐种子 Top-10 另报告 McNemar 与用户样本 bootstrap。",
        "",
        "## Validation",
        "",
        "| 模式 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Full−模式 ΔR@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = validation[mode]
        lines.append(
            f"| {MODE_LABELS[mode]} | {ci_text(row['metrics']['recall_at_5'])} | "
            f"{ci_text(row['metrics']['ndcg_at_5'])} | "
            f"{ci_text(row['metrics']['recall_at_10'])} | "
            f"{ci_text(row['metrics']['ndcg_at_10'])} | "
            f"{ci_text(row['full_minus_mode']['recall_at_10'], signed=True)} |"
        )
    lines.extend(
        [
            "",
            "## Test 全量",
            "",
            "| 模式 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Full−模式 ΔR@10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        row = test["all"][mode]
        lines.append(
            f"| {MODE_LABELS[mode]} | {ci_text(row['metrics']['recall_at_5'])} | "
            f"{ci_text(row['metrics']['ndcg_at_5'])} | "
            f"{ci_text(row['metrics']['recall_at_10'])} | "
            f"{ci_text(row['metrics']['ndcg_at_10'])} | "
            f"{ci_text(row['full_minus_mode']['recall_at_10'], signed=True)} |"
        )
    lines.extend(
        [
            "",
            "## 长历史用户",
            "",
            "| 模式 | history>10 R@10 | Full−模式 ΔR@10 | history>20 R@10 | Full−模式 ΔR@10 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        gt10 = test["gt_10"][mode]
        gt20 = test["gt_20"][mode]
        lines.append(
            f"| {MODE_LABELS[mode]} | {ci_text(gt10['metrics']['recall_at_10'])} | "
            f"{ci_text(gt10['full_minus_mode']['recall_at_10'], signed=True)} | "
            f"{ci_text(gt20['metrics']['recall_at_10'])} | "
            f"{ci_text(gt20['full_minus_mode']['recall_at_10'], signed=True)} |"
        )
    lines.extend(
        [
            "",
            "## 结论门禁",
            "",
            "只有配对 Student-t CI 完全位于 0 的一侧，才报告稳定方向。",
            "",
            "| 对照 | 全量 Test ΔR@10 | 判定 | history>10 ΔR@10 | 判定 |",
            "|---|---:|---|---:|---|",
        ]
    )
    for mode in ABLATIONS:
        all_row = summary["conclusions"][mode]["all"]
        long_row = summary["conclusions"][mode]["gt_10"]
        lines.append(
            f"| Full vs {MODE_LABELS[mode]} | "
            f"{all_row['mean_delta_recall_at_10']:+.6f} "
            f"[{all_row['ci95_low']:+.6f}, {all_row['ci95_high']:+.6f}] | "
            f"{all_row['decision']} | "
            f"{long_row['mean_delta_recall_at_10']:+.6f} "
            f"[{long_row['ci95_low']:+.6f}, {long_row['ci95_high']:+.6f}] | "
            f"{long_row['decision']} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "这是受控架构比较，能够检验替代结构在重新优化后是否恢复性能；仍然只是离线预测证据，不能解释成用户偏好的因果效应。三种子区间的样本量仍很小，所有不跨 0 的结论也应限定在当前数据、训练预算与解码协议内。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(ROOT / "configs/v2_preference_reasoning_retraining.yaml")
    examples = load_sequence_examples_jsonl(ROOT / config["data"]["test_path"])
    targets = [example.target_item_id for example in examples]
    lengths = np.asarray([len(example.history) for example in examples], dtype=np.int16)
    indices_by_group = group_indices(lengths)
    ranks = load_all_ranks(output_dir, targets)

    per_seed: dict[str, object] = {}
    for seed in SEEDS:
        validation_by_mode = {
            "full": validation_metrics(full_artifact(seed) / "report.json", "full", seed)
        }
        for mode in ABLATIONS:
            validation_by_mode[mode] = validation_metrics(
                variant_artifact(output_dir, mode, seed) / "report.json", mode, seed
            )

        test_by_group: dict[str, object] = {}
        for group, indices in indices_by_group.items():
            full_metrics = ranking_metrics(ranks[seed]["full"][indices])
            mode_rows: dict[str, object] = {}
            for mode in MODES:
                metrics = ranking_metrics(ranks[seed][mode][indices])
                row: dict[str, object] = {
                    "metrics": metrics,
                    "full_minus_mode": {
                        metric: full_metrics[metric] - metrics[metric]
                        for metric in METRICS
                    },
                }
                if mode != "full":
                    row["mcnemar_at_10_vs_full"] = exact_mcnemar(
                        ranks[seed]["full"][indices], ranks[seed][mode][indices]
                    )
                    if group in {"all", "gt_10"}:
                        row["paired_user_bootstrap_at_10_vs_full"] = (
                            categorical_bootstrap_delta(
                                ranks[seed]["full"][indices],
                                ranks[seed][mode][indices],
                                samples=args.bootstrap_samples,
                                seed=seed * 100 + ABLATIONS.index(mode),
                            )
                        )
                mode_rows[mode] = row
            test_by_group[group] = {
                "examples": int(len(indices)),
                "modes": mode_rows,
            }
        per_seed[str(seed)] = {
            "validation": {
                mode: {
                    "metrics": metrics,
                    "full_minus_mode": {
                        metric: validation_by_mode["full"][metric] - metrics[metric]
                        for metric in METRICS
                    },
                }
                for mode, metrics in validation_by_mode.items()
            },
            "test": test_by_group,
        }

    aggregate_validation: dict[str, object] = {}
    for mode in MODES:
        aggregate_validation[mode] = {
            "metrics": {
                metric: mean_confidence_interval(
                    [
                        per_seed[str(seed)]["validation"][mode]["metrics"][metric]
                        for seed in SEEDS
                    ]
                )
                for metric in METRICS
            },
            "full_minus_mode": {
                metric: mean_confidence_interval(
                    [
                        per_seed[str(seed)]["validation"][mode]["full_minus_mode"][
                            metric
                        ]
                        for seed in SEEDS
                    ]
                )
                for metric in METRICS
            },
        }

    aggregate_test: dict[str, object] = {}
    for group in GROUPS:
        aggregate_test[group] = {"examples": int(len(indices_by_group[group]))}
        for mode in MODES:
            aggregate_test[group][mode] = {
                "metrics": {
                    metric: mean_confidence_interval(
                        [
                            per_seed[str(seed)]["test"][group]["modes"][mode][
                                "metrics"
                            ][metric]
                            for seed in SEEDS
                        ]
                    )
                    for metric in METRICS
                },
                "full_minus_mode": {
                    metric: mean_confidence_interval(
                        [
                            per_seed[str(seed)]["test"][group]["modes"][mode][
                                "full_minus_mode"
                            ][metric]
                            for seed in SEEDS
                        ]
                    )
                    for metric in METRICS
                },
            }

    conclusions: dict[str, object] = {}
    for mode in ABLATIONS:
        conclusions[mode] = {}
        for group in ("all", "gt_10", "gt_20"):
            interval = aggregate_test[group][mode]["full_minus_mode"]["recall_at_10"]
            if interval["ci95_low"] > 0:
                decision = "Full 稳定更高"
            elif interval["ci95_high"] < 0:
                decision = f"{MODE_LABELS[mode]} 稳定更高"
            else:
                decision = "无稳定差异"
            conclusions[mode][group] = {
                "mean_delta_recall_at_10": interval["mean"],
                "ci95_low": interval["ci95_low"],
                "ci95_high": interval["ci95_high"],
                "decision": decision,
            }

    latency = {
        mode: mean_confidence_interval(
            [
                result_latency(
                    variant_artifact(output_dir, mode, seed) / "beam20_test.json"
                )
                for seed in SEEDS
            ]
        )
        for mode in ABLATIONS
    }
    summary: dict[str, object] = {
        "status": "completed",
        "experiment": "preference_reasoning_retraining_ablation_v1",
        "protocol": {
            "design": "independent from-scratch architecture retraining ablation",
            "seeds": list(SEEDS),
            "epochs": 10,
            "beam_size": 20,
            "top_k": 10,
            "validation_only_during_training": True,
            "test_evaluated_after_freeze": True,
            "synthetic_data_used": False,
            "paired_seed_interval": "two-sided 95% Student-t",
            "bootstrap_samples": args.bootstrap_samples,
        },
        "mode_definitions": {
            "full": "learned event attention + learned gate + fusion MLP",
            "fixed_0p5_gate": "learned event attention + fixed equal gate + fusion MLP",
            "no_event_attention": "base behavior/recency weights + learned gate + fusion MLP",
            "no_fusion_mlp": "learned event attention + learned gate; gated mixture directly",
        },
        "history_counts": {key: int(len(value)) for key, value in indices_by_group.items()},
        "per_seed": per_seed,
        "aggregate": {
            "validation": aggregate_validation,
            "test": aggregate_test,
        },
        "conclusions": conclusions,
        "latency_ms_per_example": latency,
        "provenance": {
            "dataset": config["data"]["test_path"],
            "dataset_sha256": sha256_file(ROOT / config["data"]["test_path"]),
            "real_data": True,
            "synthetic_data_used": False,
            "full_checkpoints": {
                str(seed): sha256_file(full_artifact(seed) / "generator.pt")
                for seed in SEEDS
            },
            "variant_checkpoints": {
                mode: {
                    str(seed): sha256_file(
                        variant_artifact(output_dir, mode, seed) / "generator.pt"
                    )
                    for seed in SEEDS
                }
                for mode in ABLATIONS
            },
            "test_reports": {
                mode: {
                    str(seed): sha256_file(
                        variant_artifact(output_dir, mode, seed) / "beam20_test.json"
                    )
                    for seed in SEEDS
                }
                for mode in ABLATIONS
            },
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(summary, output_dir / "REPORT.md")
    print(f"completed: {output_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
