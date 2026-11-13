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
MODES = (
    "learned_gate_full",
    "fixed_0p5_gate",
    "long_term_only",
    "short_term_only",
    "no_event_attention",
    "no_fusion_mlp",
    "mean_pool_coarse",
)
NEW_MODES = (
    "fixed_0p5_gate",
    "long_term_only",
    "short_term_only",
    "no_event_attention",
    "no_fusion_mlp",
)
GROUPS = ("all", "le_10", "11_20", "gt_20", "gt_10")
METRICS = ("recall_at_5", "ndcg_at_5", "recall_at_10", "ndcg_at_10")
MODE_LABELS = {
    "learned_gate_full": "Learned gate (full)",
    "fixed_0p5_gate": "Fixed 0.5 gate",
    "long_term_only": "Long-term only",
    "short_term_only": "Short-term only",
    "no_event_attention": "No event attention",
    "no_fusion_mlp": "No fusion MLP",
    "mean_pool_coarse": "Mean-pool (coarse)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize frozen-checkpoint Preference Reasoning mechanism interventions"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/preference_reasoning_mechanism_ablation_v1"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def full_indicator_path(seed: int) -> Path:
    return ROOT / (
        "artifacts/candidate_reranking_mild_v1/test/"
        f"seed_{seed}/hit_indicators.jsonl"
    )


def mean_pool_indicator_path(seed: int) -> Path:
    return ROOT / (
        "artifacts/preference_reasoning_ablation_v1/mean_pool/"
        f"seed_{seed}/hit_indicators/beam_20_diversity_0p0.jsonl"
    )


def mode_indicator_path(output_dir: Path, mode: str, seed: int) -> Path:
    return output_dir / mode / f"seed_{seed}/hit_indicators/beam_20_diversity_0p0.jsonl"


def mode_report_path(output_dir: Path, mode: str, seed: int) -> Path:
    return output_dir / mode / f"seed_{seed}/beam20_test.json"


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
            value = row.get(field)
            ranks[index] = 0 if value is None else int(value)
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


def exact_mcnemar(full_ranks: np.ndarray, mode_ranks: np.ndarray) -> dict[str, float | int]:
    full = (full_ranks > 0) & (full_ranks <= 10)
    mode = (mode_ranks > 0) & (mode_ranks <= 10)
    full_only = int(np.sum(full & ~mode))
    mode_only = int(np.sum(~full & mode))
    discordant = full_only + mode_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(full_only, mode_only)
        tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / 2**discordant
        p_value = min(1.0, 2.0 * tail)
    return {
        "full_only_hits": full_only,
        "mode_only_hits": mode_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def load_all_ranks(
    output_dir: Path,
    targets: list[str],
) -> dict[int, dict[str, np.ndarray]]:
    ranks: dict[int, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        seed_ranks = {
            "learned_gate_full": load_ranks(
                full_indicator_path(seed), "baseline_rank", targets
            ),
            "mean_pool_coarse": load_ranks(
                mean_pool_indicator_path(seed), "top_k_rank", targets
            ),
        }
        for mode in NEW_MODES:
            seed_ranks[mode] = load_ranks(
                mode_indicator_path(output_dir, mode, seed), "top_k_rank", targets
            )
        ranks[seed] = seed_ranks
    return ranks


def summarize(
    ranks: dict[int, dict[str, np.ndarray]],
    indices_by_group: dict[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    per_seed: dict[str, object] = {}
    for seed in SEEDS:
        seed_rows: dict[str, object] = {}
        for group in GROUPS:
            indices = indices_by_group[group]
            full_metrics = ranking_metrics(ranks[seed]["learned_gate_full"][indices])
            mode_rows: dict[str, object] = {}
            for mode in MODES:
                metrics = ranking_metrics(ranks[seed][mode][indices])
                mode_rows[mode] = {
                    "metrics": metrics,
                    "full_minus_mode": {
                        metric: full_metrics[metric] - metrics[metric] for metric in METRICS
                    },
                    "mcnemar_at_10_vs_full": exact_mcnemar(
                        ranks[seed]["learned_gate_full"][indices],
                        ranks[seed][mode][indices],
                    ),
                }
            seed_rows[group] = {"examples": int(len(indices)), "modes": mode_rows}
        per_seed[str(seed)] = seed_rows

    aggregate: dict[str, object] = {}
    for group in GROUPS:
        group_rows: dict[str, object] = {"examples": int(len(indices_by_group[group]))}
        for mode in MODES:
            group_rows[mode] = {
                "metrics": {
                    metric: mean_confidence_interval(
                        [
                            per_seed[str(seed)][group]["modes"][mode]["metrics"][metric]
                            for seed in SEEDS
                        ]
                    )
                    for metric in METRICS
                },
                "full_minus_mode": {
                    metric: mean_confidence_interval(
                        [
                            per_seed[str(seed)][group]["modes"][mode]["full_minus_mode"][
                                metric
                            ]
                            for seed in SEEDS
                        ]
                    )
                    for metric in METRICS
                },
            }
        aggregate[group] = group_rows

    attribution_map = {
        "learned_gate_contribution": "fixed_0p5_gate",
        "event_attention_contribution": "no_event_attention",
        "fusion_mlp_contribution": "no_fusion_mlp",
    }
    attribution: dict[str, object] = {}
    for name, ablation in attribution_map.items():
        attribution[name] = {
            "definition": f"learned_gate_full minus {ablation}",
            "groups": {
                group: aggregate[group][ablation]["full_minus_mode"] for group in GROUPS
            },
        }

    conclusions: dict[str, object] = {}
    for name, ablation in attribution_map.items():
        conclusions[name] = {}
        for group in ("all", "gt_10", "gt_20"):
            interval = aggregate[group][ablation]["full_minus_mode"]["recall_at_10"]
            conclusions[name][group] = {
                "mean_delta_recall_at_10": interval["mean"],
                "ci95_low": interval["ci95_low"],
                "ci95_high": interval["ci95_high"],
                "stable_positive": bool(interval["ci95_low"] > 0),
                "stable_negative": bool(interval["ci95_high"] < 0),
            }
    return per_seed, aggregate, {"attribution": attribution, "conclusions": conclusions}


def ci_text(row: dict[str, object]) -> str:
    return (
        f"{float(row['mean']):.6f} "
        f"[{float(row['ci95_low']):.6f}, {float(row['ci95_high']):.6f}]"
    )


def write_report(summary: dict[str, object], path: Path) -> None:
    aggregate = summary["aggregate"]
    lines = [
        "# Preference Reasoning 细粒度机制消融",
        "",
        "## 设计",
        "",
        "- 数据：真实 Amazon Reviews 2023 / Musical Instruments Test，57,439 条；无合成数据。",
        "- 复用 seed 2026/2027/2028 的同一批 full 模型检查点，不重新训练。",
        "- 所有干预均为冻结检查点上的 Beam-20/Top-10 推理；因此组件差异不混入训练随机性。",
        "- long/short-only 是信息源充分性测试；fixed gate、no attention、no fusion 才是对应组件的单因素归因。",
        "- 区间是三独立模型种子的配对 Student-t 95% CI。",
        "",
        "## 全量测试集",
        "",
        "| 模式 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Full−模式 ΔR@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = aggregate["all"][mode]
        lines.append(
            f"| {MODE_LABELS[mode]} | {ci_text(row['metrics']['recall_at_5'])} | "
            f"{ci_text(row['metrics']['ndcg_at_5'])} | "
            f"{ci_text(row['metrics']['recall_at_10'])} | "
            f"{ci_text(row['metrics']['ndcg_at_10'])} | "
            f"{ci_text(row['full_minus_mode']['recall_at_10'])} |"
        )
    lines.extend(
        [
            "",
            "## 长历史用户（history>10）",
            "",
            "| 模式 | Recall@10 | NDCG@10 | Full−模式 ΔR@10 | Full−模式 ΔNDCG@10 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        row = aggregate["gt_10"][mode]
        lines.append(
            f"| {MODE_LABELS[mode]} | {ci_text(row['metrics']['recall_at_10'])} | "
            f"{ci_text(row['metrics']['ndcg_at_10'])} | "
            f"{ci_text(row['full_minus_mode']['recall_at_10'])} | "
            f"{ci_text(row['full_minus_mode']['ndcg_at_10'])} |"
        )
    lines.extend(
        [
            "",
            "## 单组件归因",
            "",
            "正值表示完整模型优于移除/固定该组件后的模型。",
            "",
            "| 组件 | 全量 ΔR@10 | history>10 ΔR@10 | history>20 ΔR@10 |",
            "|---|---:|---:|---:|",
        ]
    )
    component_rows = (
        ("Learned gate vs fixed 0.5", "fixed_0p5_gate"),
        ("Event attention", "no_event_attention"),
        ("Fusion MLP", "no_fusion_mlp"),
    )
    for label, mode in component_rows:
        lines.append(
            f"| {label} | "
            f"{ci_text(aggregate['all'][mode]['full_minus_mode']['recall_at_10'])} | "
            f"{ci_text(aggregate['gt_10'][mode]['full_minus_mode']['recall_at_10'])} | "
            f"{ci_text(aggregate['gt_20'][mode]['full_minus_mode']['recall_at_10'])} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "这是同一已训练模型内的机制干预，回答各模块对当前模型预测是否必要；它不等价于分别重训各架构后的容量比较，也不构成用户偏好的因果识别。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs/v2_final_default.yaml")
    test_path = ROOT / config["data"]["test_path"]
    examples = load_sequence_examples_jsonl(test_path)
    targets = [example.target_item_id for example in examples]
    lengths = np.asarray([len(example.history) for example in examples], dtype=np.int16)
    indices_by_group = group_indices(lengths)
    ranks = load_all_ranks(output_dir, targets)
    per_seed, aggregate, inference = summarize(ranks, indices_by_group)

    latency: dict[str, object] = {}
    for mode in NEW_MODES:
        values = []
        for seed in SEEDS:
            report = json.loads(mode_report_path(output_dir, mode, seed).read_text())
            values.append(float(report["results"][0]["efficiency"]["milliseconds_per_example"]))
        latency[mode] = mean_confidence_interval(values)

    summary: dict[str, object] = {
        "status": "completed",
        "experiment": "preference_reasoning_mechanism_ablation_v1",
        "protocol": {
            "design": "inference-time frozen-checkpoint mechanism intervention",
            "seeds": list(SEEDS),
            "beam_size": 20,
            "top_k": 10,
            "retrained": False,
            "synthetic_data_used": False,
            "mode_definitions": {
                "learned_gate_full": "learned event attention + learned gate + fusion MLP",
                "fixed_0p5_gate": "event attention + fixed equal gate + fusion MLP",
                "long_term_only": "long-term state directly; source-sufficiency endpoint",
                "short_term_only": "short-term state directly; source-sufficiency endpoint",
                "no_event_attention": "base intensity/recency weights + learned gate + fusion MLP",
                "no_fusion_mlp": "event attention + learned gate; gated mix directly",
                "mean_pool_coarse": "no learned attention/gate/fusion; separately trained coarse ablation",
            },
        },
        "history_counts": {key: int(len(value)) for key, value in indices_by_group.items()},
        "per_seed": per_seed,
        "aggregate": aggregate,
        **inference,
        "latency_ms_per_example": latency,
        "provenance": {
            "dataset": str(test_path.relative_to(ROOT)),
            "dataset_sha256": sha256_file(test_path),
            "real_data": True,
            "synthetic_data_used": False,
            "full_checkpoints": {
                str(seed): sha256_file(
                    ROOT
                    / "artifacts/amazon2023_musical_instruments_long_tail_training_v1/"
                    / f"formal/token_loss_010/seed_{seed}/generator.pt"
                )
                for seed in SEEDS
            },
            "reports": {
                mode: {
                    str(seed): sha256_file(mode_report_path(output_dir, mode, seed))
                    for seed in SEEDS
                }
                for mode in NEW_MODES
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
