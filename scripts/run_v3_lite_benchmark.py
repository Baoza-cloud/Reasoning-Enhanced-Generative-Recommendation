from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from regenrec.causal_benchmark import (
    UPLIFT_LEARNERS,
    cross_fitted_uplift_scores,
    evaluate_score_mapping,
    outcome_calibration_metrics,
    percentile_rank,
    policy_pareto_table,
    stratified_poisson_bootstrap_difference,
    stratified_poisson_bootstrap_metrics,
    uplift_policy_metrics,
)
from regenrec.config import load_config
from regenrec.datasets.criteo_uplift import load_criteo_uplift_tensors, sha256_file
from regenrec.metrics import mean_confidence_interval


KEY_METRICS = (
    "auuc",
    "qini_per_row",
    "uplift_at_10_percent",
    "policy_incremental_at_10_percent",
    "incremental_conversions_per_1000_at_10_percent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the validation-selected V3-lite Criteo uplift benchmark"
    )
    parser.add_argument("--config", default="configs/v3_lite_smoke.yaml")
    return parser.parse_args()


def deterministic_split_indices(num_rows: int, seed: int) -> dict[str, np.ndarray]:
    indices = np.arange(num_rows, dtype=np.int64)
    buckets = ((indices * 1_103_515_245 + seed) % 2_147_483_647) % 10
    return {
        "train": indices[buckets < 8],
        "validation": indices[buckets == 8],
        "test": indices[buckets == 9],
    }


def standardize(
    features: np.ndarray,
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    fit_values = features[fit_indices].astype(np.float64)
    mean = fit_values.mean(axis=0)
    std = fit_values.std(axis=0)
    std[std < 1e-6] = 1.0
    normalized_fit = ((fit_values - mean) / std).astype(np.float32)
    normalized_evaluation = ((features[evaluation_indices] - mean) / std).astype(np.float32)
    return normalized_fit, normalized_evaluation, {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def load_or_create_sample(
    config: Mapping[str, object],
    audit: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    data_config = config["data"]
    if not isinstance(data_config, Mapping):
        raise ValueError("data config must be a mapping")
    cache_path = Path(str(data_config["sample_cache_path"]))
    max_rows = int(data_config["max_rows"])
    sample_seed = int(data_config["sample_seed"])
    expected_manifest = {
        "source_sha256": audit["source_sha256"],
        "audited_rows": int(audit["statistics"]["rows"]),
        "max_rows": max_rows,
        "sample_seed": sample_seed,
        "sampling": "uniform reservoir over every audited source row",
    }
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.get("manifest") == expected_manifest:
            return (
                cached["features"],
                cached["treatment"],
                cached["outcome"],
                {**expected_manifest, "cache_reused": True, "cache_path": str(cache_path.resolve())},
            )
    features, treatment, outcome = load_criteo_uplift_tensors(
        str(data_config["path"]),
        expected_rows=expected_manifest["audited_rows"],
        max_rows=max_rows,
        sample_seed=sample_seed,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "manifest": expected_manifest,
            "features": features,
            "treatment": treatment,
            "outcome": outcome,
        },
        cache_path,
    )
    return (
        features,
        treatment,
        outcome,
        {**expected_manifest, "cache_reused": False, "cache_path": str(cache_path.resolve())},
    )


def summarize_seed_metrics(
    seed_reports: Sequence[Mapping[str, object]],
    score_names: Sequence[str],
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for score_name in score_names:
        summary[score_name] = {}
        for metric in KEY_METRICS:
            values = [
                float(report["metrics"][score_name][metric])
                for report in seed_reports
            ]
            if len(values) >= 2:
                summary[score_name][metric] = mean_confidence_interval(values)
            else:
                summary[score_name][metric] = {"n": 1, "mean": values[0], "ci95": None}
    return summary


def mean_scores(score_sets: Sequence[Mapping[str, np.ndarray]], name: str) -> np.ndarray:
    return np.mean(np.stack([scores[name] for scores in score_sets], axis=0), axis=0)


def summarize_metric_values(values: Sequence[float]) -> dict[str, object]:
    normalized = [float(value) for value in values]
    if len(normalized) >= 2:
        return mean_confidence_interval(normalized)
    return {"n": 1, "mean": normalized[0], "ci95": None}


def paired_seed_metric_summary(
    candidate: Sequence[Mapping[str, float]],
    baseline: Sequence[Mapping[str, float]],
) -> dict[str, object]:
    return {
        metric: summarize_metric_values(
            [
                float(candidate_row[metric]) - float(baseline_row[metric])
                for candidate_row, baseline_row in zip(candidate, baseline, strict=True)
            ]
        )
        for metric in KEY_METRICS
    }


def run_phase(
    *,
    phase: str,
    features: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    seeds: Sequence[int],
    folds: int,
    max_iter: int,
    alpha: float,
    fractions: Sequence[float],
) -> tuple[list[dict[str, object]], list[dict[str, np.ndarray]], dict[str, object]]:
    fit_features, evaluation_features, normalization = standardize(
        features, fit_indices, evaluation_indices
    )
    fit_treatment = treatment[fit_indices]
    fit_outcome = outcome[fit_indices]
    evaluation_treatment = treatment[evaluation_indices]
    evaluation_outcome = outcome[evaluation_indices]
    propensity = float(fit_treatment.mean())
    seed_reports: list[dict[str, object]] = []
    seed_scores: list[dict[str, np.ndarray]] = []
    for seed in seeds:
        print(json.dumps({"phase": phase, "seed": seed, "status": "training"}), flush=True)
        fitted = cross_fitted_uplift_scores(
            fit_features,
            fit_treatment,
            fit_outcome,
            evaluation_features,
            propensity=propensity,
            folds=folds,
            seed=int(seed),
            max_iter=max_iter,
            alpha=alpha,
        )
        scores = fitted["scores"]
        metrics = evaluate_score_mapping(
            scores,
            evaluation_treatment,
            evaluation_outcome,
            propensity=propensity,
            fractions=fractions,
        )
        calibration = {
            name: outcome_calibration_metrics(
                values["control"],
                values["treated"],
                evaluation_treatment,
                evaluation_outcome,
            )
            for name, values in fitted["potential_outcomes"].items()
        }
        seed_reports.append(
            {
                "seed": int(seed),
                "fit_rows": int(fit_indices.size),
                "evaluation_rows": int(evaluation_indices.size),
                "fit_treatment_rate": propensity,
                "evaluation_treatment_rate": float(evaluation_treatment.mean()),
                "fit_conversions": int(fit_outcome.sum()),
                "evaluation_conversions": int(evaluation_outcome.sum()),
                "training_diagnostics": fitted["training_diagnostics"],
                "calibration": calibration,
                "metrics": metrics,
            }
        )
        seed_scores.append(scores)
        print(
            json.dumps(
                {
                    "phase": phase,
                    "seed": seed,
                    "status": "complete",
                    "dr_qini_per_row": metrics["dr_learner"]["qini_per_row"],
                    "t_qini_per_row": metrics["t_learner"]["qini_per_row"],
                }
            ),
            flush=True,
        )
    return seed_reports, seed_scores, {
        "propensity": propensity,
        "normalization": normalization,
        "fit_rows": int(fit_indices.size),
        "evaluation_rows": int(evaluation_indices.size),
    }


def select_model(validation_summary: Mapping[str, Mapping[str, object]]) -> str:
    return max(
        UPLIFT_LEARNERS,
        key=lambda name: (
            float(validation_summary[name]["qini_per_row"]["mean"]),
            float(validation_summary[name]["auuc"]["mean"]),
            name,
        ),
    )


def select_pareto_weight(rows: Sequence[Mapping[str, float]]) -> float:
    eligible = [row for row in rows if row["response_proxy_retention"] >= 0.97]
    pool = eligible or list(rows)
    selected = max(
        pool,
        key=lambda row: (
            row["policy_incremental_at_target_fraction"],
            -row["uplift_weight"],
        ),
    )
    return float(selected["uplift_weight"])


def report_markdown(report: Mapping[str, object]) -> str:
    selected = str(report["selection"]["uplift_learner"])
    weight = float(report["selection"]["uplift_weight"])
    official = report["test"]["official_policy_metrics"]
    versus_response = report["test"]["paired_bootstrap_comparisons"][
        "vs_response_model"
    ]["metrics"]
    gate = report["acceptance_gate"]
    lines = [
        "# V3-lite 因果增量与策略重排报告",
        "",
        "## 冻结结论",
        "",
        str(report["conclusion"]),
        "",
        "## 数据与协议",
        "",
        f"- 数据：`{report['dataset']['name']}`，真实数据，未使用合成样本。",
        f"- 正式样本：{report['dataset']['sample']['max_rows']:,} 行，固定 reservoir seed。",
        "- 模型选择：只使用 Validation；Test 在 learner 与策略权重冻结后评估。",
        "- 学习器：S/T/X/DR learner；nuisance 使用 OOF cross-fitting。",
        "- 不确定性：三训练种子 Student-t CI + Test 分层 Poisson row bootstrap。",
        "",
        "## 冻结方案",
        "",
        f"- uplift learner：`{selected}`",
        f"- 策略：`response_rank + {weight:g} * uplift_rank`",
        f"- Test Qini/row：{official['qini_per_row']:.8f}",
        f"- Test policy incremental@10%：{official['policy_incremental_at_10_percent']:.8f}",
        "- Criteo 下采样尺度每千样本估计增量转化："
        f"{official['incremental_conversions_per_1000_at_10_percent']:.4f}",
        "- 相对 response-only 的 policy incremental@10% 差值 CI："
        f"[{versus_response['policy_incremental_at_10_percent']['ci95_low']:.8f}, "
        f"{versus_response['policy_incremental_at_10_percent']['ci95_high']:.8f}]",
        "",
        "## 验收门",
        "",
        f"- Criteo 随机实验资格：`{gate['criteo_randomized_data_gate']}`",
        f"- Amazon 候选级可识别：`{gate['amazon_candidate_causal_gate']}`",
        f"- Test 稳定增量门：`{gate['stable_auxiliary_uplift_gate']}`",
        f"- V3-lite 最终状态：`{gate['v3_lite_status']}`",
        "",
        "## 科学边界",
        "",
        "Criteo 是营销处理的随机实验 benchmark，不含 Amazon 商品候选集合和推荐相关性标签。",
        "因此这里验证的是 uplift 学习与离线 targeting policy，不把结果改写成 Amazon 商品级",
        "增量转化提升；Amazon 仍需真实候选曝光、对照、propensity 与固定转化窗口。",
        "",
        "完整逐种子指标、校准、bootstrap、Pareto 表和数据哈希见 `report.json`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    evaluation_config = config["evaluation"]
    training_config = config["training"]
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = Path(str(data_config["audit_path"]))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("synthetic_data_used") is not False:
        raise ValueError("V3-lite requires a passed real-data Criteo audit")
    source_path = Path(str(data_config["path"]))
    if source_path.stat().st_size != int(audit["source_bytes"]):
        raise ValueError("Criteo source length differs from the audited file")
    if sha256_file(source_path) != audit["source_sha256"]:
        raise ValueError("Criteo source SHA-256 differs from the audited file")

    started = time.perf_counter()
    feature_tensor, treatment_tensor, outcome_tensor, sample_manifest = load_or_create_sample(
        config, audit
    )
    features = feature_tensor.numpy()
    treatment = treatment_tensor.numpy().astype(np.int8)
    outcome = outcome_tensor.numpy().astype(np.int8)
    splits = deterministic_split_indices(features.shape[0], int(data_config["split_seed"]))
    train_validation_indices = np.sort(
        np.concatenate((splits["train"], splits["validation"]))
    )
    seeds = tuple(int(value) for value in training_config["seeds"])
    fractions = tuple(float(value) for value in evaluation_config["fractions"])

    validation_reports, validation_scores, validation_stage = run_phase(
        phase="validation",
        features=features,
        treatment=treatment,
        outcome=outcome,
        fit_indices=splits["train"],
        evaluation_indices=splits["validation"],
        seeds=seeds,
        folds=int(training_config["cross_fitting_folds"]),
        max_iter=int(training_config["max_iter"]),
        alpha=float(training_config["alpha"]),
        fractions=fractions,
    )
    score_names = tuple(validation_scores[0])
    validation_summary = summarize_seed_metrics(validation_reports, score_names)
    selected_learner = select_model(validation_summary)
    validation_selected_scores = mean_scores(validation_scores, selected_learner)
    validation_response_scores = mean_scores(validation_scores, "response_model")
    validation_pareto = policy_pareto_table(
        validation_response_scores,
        validation_selected_scores,
        treatment[splits["validation"]],
        outcome[splits["validation"]],
        propensity=float(validation_stage["propensity"]),
        uplift_weights=tuple(float(value) for value in evaluation_config["uplift_weights"]),
        top_fraction=float(evaluation_config["policy_top_fraction"]),
    )
    selected_weight = select_pareto_weight(validation_pareto)
    selection = {
        "source": "validation_only",
        "primary_metric": "mean qini_per_row across configured training seeds",
        "uplift_learner": selected_learner,
        "uplift_weight": selected_weight,
        "response_proxy_retention_floor": 0.97,
        "validation_pareto": validation_pareto,
        "frozen_before_test": True,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selection_frozen": selection}, ensure_ascii=False), flush=True)

    test_reports, test_scores, test_stage = run_phase(
        phase="test_after_freeze",
        features=features,
        treatment=treatment,
        outcome=outcome,
        fit_indices=train_validation_indices,
        evaluation_indices=splits["test"],
        seeds=seeds,
        folds=int(training_config["cross_fitting_folds"]),
        max_iter=int(training_config["max_iter"]),
        alpha=float(training_config["alpha"]),
        fractions=fractions,
    )
    test_summary = summarize_seed_metrics(test_reports, score_names)
    test_selected_scores = mean_scores(test_scores, selected_learner)
    test_response_scores = mean_scores(test_scores, "response_model")
    test_random_scores = mean_scores(test_scores, "random")
    combined_test_scores = percentile_rank(test_response_scores) + selected_weight * percentile_rank(
        test_selected_scores
    )
    test_treatment = treatment[splits["test"]]
    test_outcome = outcome[splits["test"]]
    test_propensity = float(test_stage["propensity"])
    official_metrics = uplift_policy_metrics(
        combined_test_scores,
        test_treatment,
        test_outcome,
        propensity=test_propensity,
        fractions=fractions,
    )
    test_pareto = policy_pareto_table(
        test_response_scores,
        test_selected_scores,
        test_treatment,
        test_outcome,
        propensity=test_propensity,
        uplift_weights=tuple(float(value) for value in evaluation_config["uplift_weights"]),
        top_fraction=float(evaluation_config["policy_top_fraction"]),
    )
    official_seed_metrics = [
        uplift_policy_metrics(
            percentile_rank(scores["response_model"])
            + selected_weight * percentile_rank(scores[selected_learner]),
            test_treatment,
            test_outcome,
            propensity=test_propensity,
            fractions=fractions,
        )
        for scores in test_scores
    ]
    official_seed_summary = {
        metric: summarize_metric_values(
            [float(seed_metrics[metric]) for seed_metrics in official_seed_metrics]
        )
        for metric in KEY_METRICS
    }
    response_seed_metrics = [
        report["metrics"]["response_model"] for report in test_reports
    ]
    random_seed_metrics = [report["metrics"]["random"] for report in test_reports]
    bootstrap_metric_names = (
        "qini_per_row",
        "policy_incremental_at_10_percent",
        "incremental_conversions_per_1000_at_10_percent",
    )
    bootstrap_samples = int(evaluation_config["bootstrap_samples"])
    bootstrap_seed = int(evaluation_config["bootstrap_seed"])
    official_intervals = stratified_poisson_bootstrap_metrics(
        combined_test_scores,
        test_treatment,
        test_outcome,
        propensity=test_propensity,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        metric_names=bootstrap_metric_names,
        fractions=(float(evaluation_config["policy_top_fraction"]),),
    )
    comparisons = {
        "vs_random": stratified_poisson_bootstrap_difference(
            combined_test_scores,
            test_random_scores,
            test_treatment,
            test_outcome,
            propensity=test_propensity,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 1,
            metric_names=bootstrap_metric_names,
            fractions=(float(evaluation_config["policy_top_fraction"]),),
        ),
        "vs_response_model": stratified_poisson_bootstrap_difference(
            combined_test_scores,
            test_response_scores,
            test_treatment,
            test_outcome,
            propensity=test_propensity,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 2,
            metric_names=bootstrap_metric_names,
            fractions=(float(evaluation_config["policy_top_fraction"]),),
        ),
    }
    interval_metrics = official_intervals["metrics"]
    comparison_metrics = comparisons["vs_random"]["metrics"]
    targeting_signal_gate = (
        interval_metrics["qini_per_row"]["ci95_low"] > 0
        and interval_metrics["policy_incremental_at_10_percent"]["ci95_low"] > 0
        and comparison_metrics["qini_per_row"]["ci95_low"] > 0
    )
    response_comparison_metrics = comparisons["vs_response_model"]["metrics"]
    uplift_over_response_gate = (
        response_comparison_metrics["qini_per_row"]["ci95_low"] > 0
        and response_comparison_metrics["policy_incremental_at_10_percent"][
            "ci95_low"
        ]
        > 0
    )
    stable_gate = targeting_signal_gate and uplift_over_response_gate
    amazon_audit_path = Path(str(config["amazon_readiness_path"]))
    amazon_audit = json.loads(amazon_audit_path.read_text(encoding="utf-8"))
    gate = {
        "criteo_randomized_data_gate": "passed",
        "amazon_candidate_causal_gate": amazon_audit["status"],
        "stable_auxiliary_uplift_gate": "passed" if stable_gate else "failed",
        "targeting_signal_vs_random_gate": "passed"
        if targeting_signal_gate
        else "failed",
        "uplift_increment_over_response_gate": "passed"
        if uplift_over_response_gate
        else "failed",
        "criteria": {
            "official_qini_per_row_bootstrap_low_gt_zero": interval_metrics[
                "qini_per_row"
            ]["ci95_low"]
            > 0,
            "official_policy_incremental_at_10_bootstrap_low_gt_zero": interval_metrics[
                "policy_incremental_at_10_percent"
            ]["ci95_low"]
            > 0,
            "qini_per_row_delta_vs_random_bootstrap_low_gt_zero": comparison_metrics[
                "qini_per_row"
            ]["ci95_low"]
            > 0,
            "qini_per_row_delta_vs_response_bootstrap_low_gt_zero": (
                response_comparison_metrics["qini_per_row"]["ci95_low"] > 0
            ),
            "policy_incremental_delta_vs_response_bootstrap_low_gt_zero": (
                response_comparison_metrics["policy_incremental_at_10_percent"][
                    "ci95_low"
                ]
                > 0
            ),
        },
        "v3_lite_status": "completed_with_stable_auxiliary_gain"
        if stable_gate
        else "completed_without_stable_incremental_uplift_over_response",
    }
    conclusion = (
        "V3-lite 已完成。Criteo 随机实验上的验证集冻结策略通过稳定增量门；该证据仅支持"
        "辅助 uplift benchmark，不构成 Amazon 商品级因果提升。"
        if stable_gate
        else "V3-lite 已完成。冻结策略在 Criteo 上稳定优于 random targeting，但相对 "
        "response-only 基线的增量区间跨 0，因此没有证据证明 uplift 模块带来额外稳定收益；"
        "Amazon 商品级因果收益同样不作声明。"
    )
    report = {
        "status": "completed",
        "synthetic_data_used": False,
        "dataset": {
            "name": audit["source_dataset"],
            "source_file": str(source_path.resolve()),
            "source_bytes": audit["source_bytes"],
            "source_sha256": audit["source_sha256"],
            "audit_path": str(audit_path.resolve()),
            "sample": sample_manifest,
            "loaded_rows": int(features.shape[0]),
            "split": {name: int(values.size) for name, values in splits.items()},
            "split_seed": int(data_config["split_seed"]),
            "sampling_note": (
                "The formal run uses an exact uniform reservoir sample over every audited row; "
                "it is not a prefix sample and is never described as the full dataset."
            ),
        },
        "protocol": {
            "training_seeds": list(seeds),
            "cross_fitting_folds": int(training_config["cross_fitting_folds"]),
            "max_iter": int(training_config["max_iter"]),
            "alpha": float(training_config["alpha"]),
            "learners": list(UPLIFT_LEARNERS),
            "baselines": ["random", "response_model"],
            "selection": "Validation mean Qini/row; Test isolated until selection.json is written",
            "bootstrap": "paired stratified Poisson row bootstrap",
            "test_evaluated_after_freeze": True,
        },
        "validation": {
            "stage": validation_stage,
            "per_seed": validation_reports,
            "three_seed_summary": validation_summary,
        },
        "selection": selection,
        "test": {
            "stage": test_stage,
            "per_seed": test_reports,
            "three_seed_summary": test_summary,
            "official_policy_formula": f"response_rank + {selected_weight:g} * uplift_rank",
            "official_policy_metrics": official_metrics,
            "official_policy_per_seed": official_seed_metrics,
            "official_policy_three_seed_summary": official_seed_summary,
            "official_policy_paired_seed_differences": {
                "vs_random": paired_seed_metric_summary(
                    official_seed_metrics, random_seed_metrics
                ),
                "vs_response_model": paired_seed_metric_summary(
                    official_seed_metrics, response_seed_metrics
                ),
            },
            "official_policy_bootstrap": official_intervals,
            "paired_bootstrap_comparisons": comparisons,
            "diagnostic_pareto_table": test_pareto,
        },
        "acceptance_gate": gate,
        "amazon_candidate_readiness": amazon_audit,
        "elapsed_seconds": time.perf_counter() - started,
        "conclusion": conclusion,
        "interpretation_guardrail": audit["identifiability_note"],
    }
    np.savez_compressed(
        output_dir / "official_test_scores.npz",
        treatment=test_treatment,
        outcome=test_outcome,
        response_score=test_response_scores,
        uplift_score=test_selected_scores,
        official_policy_score=combined_test_scores,
        random_score=test_random_scores,
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(report_markdown(report), encoding="utf-8")
    print(json.dumps({"status": "completed", "acceptance_gate": gate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
