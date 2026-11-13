from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor


UPLIFT_LEARNERS = ("s_learner", "t_learner", "x_learner", "dr_learner")
UPLIFT_BASELINES = ("random", "response_model")


@dataclass
class _ConstantClassifier:
    probability: float

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = np.full(features.shape[0], self.probability, dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


@dataclass
class _ConstantRegressor:
    value: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.value, dtype=np.float64)


def _validate_arrays(
    features: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    treatment = np.asarray(treatment, dtype=np.int8)
    outcome = np.asarray(outcome, dtype=np.int8)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if treatment.shape != outcome.shape or treatment.shape != (features.shape[0],):
        raise ValueError("features, treatment and outcome must align")
    if features.shape[0] == 0:
        raise ValueError("causal arrays cannot be empty")
    if not np.isfinite(features).all():
        raise ValueError("features must be finite")
    if not np.isin(treatment, (0, 1)).all() or np.unique(treatment).size != 2:
        raise ValueError("treatment must contain binary treated and control observations")
    if not np.isin(outcome, (0, 1)).all():
        raise ValueError("outcome must be binary")
    return features, treatment, outcome


def _fit_classifier(
    features: np.ndarray,
    outcome: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    alpha: float,
) -> SGDClassifier | _ConstantClassifier:
    classes = np.unique(outcome)
    if classes.size == 1:
        return _ConstantClassifier(float(classes[0]))
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=max_iter,
        tol=None,
        shuffle=True,
        random_state=seed,
        average=True,
    )
    model.fit(features, outcome)
    return model


def _fit_regressor(
    features: np.ndarray,
    target: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    alpha: float,
) -> SGDRegressor | _ConstantRegressor:
    if np.allclose(target, target[0]):
        return _ConstantRegressor(float(target[0]))
    model = SGDRegressor(
        loss="huber",
        epsilon=1.35,
        penalty="l2",
        alpha=alpha,
        max_iter=max_iter,
        tol=None,
        shuffle=True,
        random_state=seed,
        average=True,
    )
    model.fit(features, target)
    return model


def _potential_outcomes_s(
    model: SGDClassifier | _ConstantClassifier,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    control_features = np.column_stack(
        (features, np.zeros(features.shape[0], dtype=np.float32))
    )
    treated_features = control_features.copy()
    treated_features[:, -1] = 1.0
    return model.predict_proba(control_features)[:, 1], model.predict_proba(treated_features)[:, 1]


def _fit_s_learner(
    features: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    alpha: float,
) -> SGDClassifier | _ConstantClassifier:
    augmented = np.column_stack((features, treatment.astype(np.float32)))
    return _fit_classifier(
        augmented,
        outcome,
        seed=seed,
        max_iter=max_iter,
        alpha=alpha,
    )


def _fit_t_learner(
    features: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    alpha: float,
) -> tuple[SGDClassifier | _ConstantClassifier, SGDClassifier | _ConstantClassifier]:
    control = treatment == 0
    treated = treatment == 1
    return (
        _fit_classifier(
            features[control],
            outcome[control],
            seed=seed + 1,
            max_iter=max_iter,
            alpha=alpha,
        ),
        _fit_classifier(
            features[treated],
            outcome[treated],
            seed=seed + 2,
            max_iter=max_iter,
            alpha=alpha,
        ),
    )


def cross_fitted_uplift_scores(
    train_features: np.ndarray,
    train_treatment: np.ndarray,
    train_outcome: np.ndarray,
    evaluation_features: np.ndarray,
    *,
    propensity: float,
    folds: int,
    seed: int,
    max_iter: int,
    alpha: float,
) -> dict[str, object]:
    """Fit S/T/X/DR learners with out-of-fold nuisance predictions.

    Nuisance outcomes are predicted out of fold for every training row. The X and
    DR second stages are then trained on those honest pseudo-labels. Evaluation rows
    never enter nuisance fitting, learner selection, or normalization.
    """
    train_features, train_treatment, train_outcome = _validate_arrays(
        train_features, train_treatment, train_outcome
    )
    evaluation_features = np.asarray(evaluation_features, dtype=np.float32)
    if evaluation_features.ndim != 2 or evaluation_features.shape[1] != train_features.shape[1]:
        raise ValueError("evaluation features must match the training feature dimension")
    if not np.isfinite(evaluation_features).all():
        raise ValueError("evaluation features must be finite")
    if not 0.01 <= propensity <= 0.99:
        raise ValueError("propensity violates overlap bounds")
    if folds < 2 or folds > train_features.shape[0]:
        raise ValueError("folds must be between 2 and the number of training rows")
    if max_iter <= 0 or alpha <= 0:
        raise ValueError("max_iter and alpha must be positive")

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(train_features.shape[0])
    fold_assignment = np.empty(train_features.shape[0], dtype=np.int16)
    fold_assignment[permutation] = np.arange(train_features.shape[0]) % folds
    oof_s0 = np.empty(train_features.shape[0], dtype=np.float64)
    oof_s1 = np.empty_like(oof_s0)
    oof_mu0 = np.empty_like(oof_s0)
    oof_mu1 = np.empty_like(oof_s0)
    fold_rows: list[dict[str, int]] = []

    for fold in range(folds):
        held_out = fold_assignment == fold
        fitting = ~held_out
        s_model = _fit_s_learner(
            train_features[fitting],
            train_treatment[fitting],
            train_outcome[fitting],
            seed=seed + fold * 10,
            max_iter=max_iter,
            alpha=alpha,
        )
        oof_s0[held_out], oof_s1[held_out] = _potential_outcomes_s(
            s_model, train_features[held_out]
        )
        control_model, treated_model = _fit_t_learner(
            train_features[fitting],
            train_treatment[fitting],
            train_outcome[fitting],
            seed=seed + fold * 10 + 3,
            max_iter=max_iter,
            alpha=alpha,
        )
        oof_mu0[held_out] = control_model.predict_proba(train_features[held_out])[:, 1]
        oof_mu1[held_out] = treated_model.predict_proba(train_features[held_out])[:, 1]
        fold_rows.append(
            {
                "fold": fold,
                "fit_rows": int(fitting.sum()),
                "held_out_rows": int(held_out.sum()),
            }
        )

    factual_oof = np.where(train_treatment == 1, oof_mu1, oof_mu0)
    clipped_propensity = min(max(propensity, 0.01), 0.99)
    dr_target = (
        oof_mu1
        - oof_mu0
        + train_treatment * (train_outcome - oof_mu1) / clipped_propensity
        - (1 - train_treatment) * (train_outcome - oof_mu0) / (1 - clipped_propensity)
    )
    dr_model = _fit_regressor(
        train_features,
        dr_target,
        seed=seed + 50_000,
        max_iter=max_iter,
        alpha=alpha,
    )

    control_rows = train_treatment == 0
    treated_rows = train_treatment == 1
    x_control_target = oof_mu1[control_rows] - train_outcome[control_rows]
    x_treated_target = train_outcome[treated_rows] - oof_mu0[treated_rows]
    x_control_model = _fit_regressor(
        train_features[control_rows],
        x_control_target,
        seed=seed + 60_000,
        max_iter=max_iter,
        alpha=alpha,
    )
    x_treated_model = _fit_regressor(
        train_features[treated_rows],
        x_treated_target,
        seed=seed + 70_000,
        max_iter=max_iter,
        alpha=alpha,
    )

    final_s_model = _fit_s_learner(
        train_features,
        train_treatment,
        train_outcome,
        seed=seed + 80_000,
        max_iter=max_iter,
        alpha=alpha,
    )
    eval_s0, eval_s1 = _potential_outcomes_s(final_s_model, evaluation_features)
    final_control_model, final_treated_model = _fit_t_learner(
        train_features,
        train_treatment,
        train_outcome,
        seed=seed + 90_000,
        max_iter=max_iter,
        alpha=alpha,
    )
    eval_mu0 = final_control_model.predict_proba(evaluation_features)[:, 1]
    eval_mu1 = final_treated_model.predict_proba(evaluation_features)[:, 1]
    x_control = x_control_model.predict(evaluation_features)
    x_treated = x_treated_model.predict(evaluation_features)
    random_scores = np.random.default_rng(seed + 100_000).random(evaluation_features.shape[0])

    return {
        "scores": {
            "random": random_scores,
            "response_model": eval_mu1,
            "s_learner": eval_s1 - eval_s0,
            "t_learner": eval_mu1 - eval_mu0,
            "x_learner": propensity * x_control + (1.0 - propensity) * x_treated,
            "dr_learner": dr_model.predict(evaluation_features),
        },
        "potential_outcomes": {
            "s_learner": {"control": eval_s0, "treated": eval_s1},
            "t_learner": {"control": eval_mu0, "treated": eval_mu1},
        },
        "training_diagnostics": {
            "cross_fitting_folds": folds,
            "folds": fold_rows,
            "oof_factual_brier": float(np.mean((factual_oof - train_outcome) ** 2)),
            "oof_factual_log_loss": binary_log_loss(train_outcome, factual_oof),
            "dr_pseudo_outcome_mean": float(np.mean(dr_target)),
            "dr_pseudo_outcome_std": float(np.std(dr_target)),
            "elapsed_seconds": time.perf_counter() - started,
        },
    }


def binary_log_loss(outcome: np.ndarray, probability: np.ndarray) -> float:
    outcome = np.asarray(outcome, dtype=np.float64)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(outcome * np.log(probability) + (1 - outcome) * np.log(1 - probability)))


def outcome_calibration_metrics(
    control_probability: np.ndarray,
    treated_probability: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, float | int]:
    if bins <= 1:
        raise ValueError("bins must be greater than one")
    treatment = np.asarray(treatment, dtype=np.int8)
    outcome = np.asarray(outcome, dtype=np.int8)
    factual = np.where(treatment == 1, treated_probability, control_probability)
    factual = np.clip(np.asarray(factual, dtype=np.float64), 1e-7, 1 - 1e-7)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.searchsorted(boundaries, factual, side="right") - 1, bins - 1)
    expected_calibration_error = 0.0
    nonempty_bins = 0
    for bin_index in range(bins):
        mask = assignments == bin_index
        if not np.any(mask):
            continue
        nonempty_bins += 1
        expected_calibration_error += float(mask.mean()) * abs(
            float(factual[mask].mean()) - float(outcome[mask].mean())
        )
    return {
        "rows": int(outcome.size),
        "observed_outcome_rate": float(outcome.mean()),
        "mean_factual_probability": float(factual.mean()),
        "brier_score": float(np.mean((factual - outcome) ** 2)),
        "log_loss": binary_log_loss(outcome, factual),
        "expected_calibration_error": expected_calibration_error,
        "nonempty_calibration_bins": nonempty_bins,
    }


def _ordered_uplift_metrics(
    order: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    propensity: float,
    fractions: Sequence[float],
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    treatment = np.asarray(treatment, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    if weights is None:
        weights = np.ones(treatment.size, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    sorted_weights = weights[order]
    sorted_treatment = treatment[order]
    sorted_outcome = outcome[order]
    total_weight = float(sorted_weights.sum())
    treated_weight = np.cumsum(sorted_weights * sorted_treatment)
    control_weight = np.cumsum(sorted_weights * (1.0 - sorted_treatment))
    treated_outcomes = np.cumsum(sorted_weights * sorted_treatment * sorted_outcome)
    control_outcomes = np.cumsum(sorted_weights * (1.0 - sorted_treatment) * sorted_outcome)
    uplift_curve = np.divide(
        treated_outcomes,
        treated_weight,
        out=np.zeros_like(treated_outcomes),
        where=treated_weight > 0,
    ) - np.divide(
        control_outcomes,
        control_weight,
        out=np.zeros_like(control_outcomes),
        where=control_weight > 0,
    )
    qini_curve = treated_outcomes - np.divide(
        treated_weight * control_outcomes,
        control_weight,
        out=np.zeros_like(treated_outcomes),
        where=control_weight > 0,
    )
    cumulative_weight = np.cumsum(sorted_weights)
    final_qini = float(qini_curve[-1])
    random_qini = final_qini * cumulative_weight / total_weight
    averaging_weight = sorted_weights / total_weight
    result = {
        "auuc": float(np.sum(averaging_weight * uplift_curve)),
        "qini_coefficient": float(np.sum(averaging_weight * (qini_curve - random_qini))),
        "qini_per_row": float(np.sum(averaging_weight * (qini_curve - random_qini)) / total_weight),
        "overall_ate": float(uplift_curve[-1]),
    }
    transformed_outcome = (
        treatment * outcome / propensity - (1.0 - treatment) * outcome / (1.0 - propensity)
    )
    treat_none_value = float(
        np.sum(weights * (1.0 - treatment) * outcome / (1.0 - propensity)) / total_weight
    )
    for fraction in fractions:
        threshold = fraction * total_weight
        position = min(int(np.searchsorted(cumulative_weight, threshold, side="left")), len(order) - 1)
        selected = order[: position + 1]
        incremental = float(np.sum(weights[selected] * transformed_outcome[selected]) / total_weight)
        label = int(round(fraction * 100))
        result[f"uplift_at_{label}_percent"] = float(uplift_curve[position])
        result[f"policy_incremental_at_{label}_percent"] = incremental
        result[f"policy_value_at_{label}_percent"] = treat_none_value + incremental
        result[f"incremental_conversions_per_1000_at_{label}_percent"] = 1000.0 * incremental
    return result


def uplift_policy_metrics(
    scores: Sequence[float] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    outcome: Sequence[int] | np.ndarray,
    *,
    propensity: float,
    fractions: Sequence[float] = (0.01, 0.05, 0.1, 0.2, 0.5),
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.int8)
    outcome = np.asarray(outcome, dtype=np.int8)
    if scores.shape != treatment.shape or scores.shape != outcome.shape or scores.size == 0:
        raise ValueError("scores, treatment and outcome must be non-empty and aligned")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not np.isin(treatment, (0, 1)).all() or np.unique(treatment).size != 2:
        raise ValueError("treatment must contain both binary arms")
    if not np.isin(outcome, (0, 1)).all():
        raise ValueError("outcome must be binary")
    if not 0.01 <= propensity <= 0.99:
        raise ValueError("propensity violates overlap bounds")
    normalized_fractions = tuple(sorted(set(float(value) for value in fractions)))
    if not normalized_fractions or normalized_fractions[0] <= 0 or normalized_fractions[-1] > 1:
        raise ValueError("fractions must be in (0, 1]")
    order = np.argsort(-scores, kind="stable")
    return _ordered_uplift_metrics(
        order,
        treatment,
        outcome,
        propensity=propensity,
        fractions=normalized_fractions,
    )


def stratified_poisson_bootstrap_difference(
    candidate_scores: Sequence[float] | np.ndarray,
    baseline_scores: Sequence[float] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    outcome: Sequence[int] | np.ndarray,
    *,
    propensity: float,
    samples: int,
    seed: int,
    metric_names: Sequence[str],
    fractions: Sequence[float] = (0.1,),
) -> dict[str, object]:
    """Paired row bootstrap with shared Poisson weights and arm-size stabilization."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    baseline_scores = np.asarray(baseline_scores, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.int8)
    outcome = np.asarray(outcome, dtype=np.int8)
    if not (
        candidate_scores.shape == baseline_scores.shape == treatment.shape == outcome.shape
    ):
        raise ValueError("bootstrap arrays must align")
    candidate_order = np.argsort(-candidate_scores, kind="stable")
    baseline_order = np.argsort(-baseline_scores, kind="stable")
    rng = np.random.default_rng(seed)
    arm_masks = (treatment == 0, treatment == 1)
    differences = {str(name): np.empty(samples, dtype=np.float64) for name in metric_names}
    for sample_index in range(samples):
        weights = rng.poisson(1.0, size=treatment.size).astype(np.float64)
        for mask in arm_masks:
            observed = float(weights[mask].sum())
            if observed == 0.0:
                weights[mask] = 1.0
                observed = float(mask.sum())
            weights[mask] *= float(mask.sum()) / observed
        candidate = _ordered_uplift_metrics(
            candidate_order,
            treatment,
            outcome,
            propensity=propensity,
            fractions=fractions,
            weights=weights,
        )
        baseline = _ordered_uplift_metrics(
            baseline_order,
            treatment,
            outcome,
            propensity=propensity,
            fractions=fractions,
            weights=weights,
        )
        for name in differences:
            differences[name][sample_index] = candidate[name] - baseline[name]
    return {
        "method": "paired stratified Poisson row bootstrap",
        "samples": samples,
        "seed": seed,
        "metrics": {
            name: {
                "mean_delta": float(values.mean()),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "probability_delta_gt_zero": float(np.mean(values > 0)),
            }
            for name, values in differences.items()
        },
    }


def stratified_poisson_bootstrap_metrics(
    scores: Sequence[float] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    outcome: Sequence[int] | np.ndarray,
    *,
    propensity: float,
    samples: int,
    seed: int,
    metric_names: Sequence[str],
    fractions: Sequence[float] = (0.1,),
) -> dict[str, object]:
    """Stratified Poisson row-bootstrap intervals for one fixed scoring policy."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    scores = np.asarray(scores, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.int8)
    outcome = np.asarray(outcome, dtype=np.int8)
    if not (scores.shape == treatment.shape == outcome.shape):
        raise ValueError("bootstrap arrays must align")
    order = np.argsort(-scores, kind="stable")
    rng = np.random.default_rng(seed)
    arm_masks = (treatment == 0, treatment == 1)
    estimates = {str(name): np.empty(samples, dtype=np.float64) for name in metric_names}
    for sample_index in range(samples):
        weights = rng.poisson(1.0, size=treatment.size).astype(np.float64)
        for mask in arm_masks:
            observed = float(weights[mask].sum())
            if observed == 0.0:
                weights[mask] = 1.0
                observed = float(mask.sum())
            weights[mask] *= float(mask.sum()) / observed
        metrics = _ordered_uplift_metrics(
            order,
            treatment,
            outcome,
            propensity=propensity,
            fractions=fractions,
            weights=weights,
        )
        for name in estimates:
            estimates[name][sample_index] = metrics[name]
    return {
        "method": "stratified Poisson row bootstrap",
        "samples": samples,
        "seed": seed,
        "metrics": {
            name: {
                "mean": float(values.mean()),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "probability_gt_zero": float(np.mean(values > 0)),
            }
            for name, values in estimates.items()
        },
    }


def percentile_rank(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite non-empty vector")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(scores.size, dtype=np.float64)
    return ranks / max(1, scores.size - 1)


def policy_pareto_table(
    response_scores: Sequence[float] | np.ndarray,
    uplift_scores: Sequence[float] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    outcome: Sequence[int] | np.ndarray,
    *,
    propensity: float,
    uplift_weights: Sequence[float],
    top_fraction: float = 0.1,
) -> list[dict[str, float]]:
    """Evaluate response-proxy plus uplift policy scores on an RCT holdout."""
    response_scores = np.asarray(response_scores, dtype=np.float64)
    uplift_scores = np.asarray(uplift_scores, dtype=np.float64)
    if response_scores.shape != uplift_scores.shape:
        raise ValueError("response and uplift scores must align")
    response_rank = percentile_rank(response_scores)
    uplift_rank = percentile_rank(uplift_scores)
    selected_count = max(1, math.ceil(response_scores.size * top_fraction))
    baseline_order = np.argsort(-response_rank, kind="stable")
    baseline_response = float(response_scores[baseline_order[:selected_count]].mean())
    label = int(round(top_fraction * 100))
    table: list[dict[str, float]] = []
    for weight in uplift_weights:
        if weight < 0 or not math.isfinite(weight):
            raise ValueError("uplift weights must be finite and non-negative")
        combined = response_rank + float(weight) * uplift_rank
        order = np.argsort(-combined, kind="stable")
        selected_response = float(response_scores[order[:selected_count]].mean())
        metrics = uplift_policy_metrics(
            combined,
            treatment,
            outcome,
            propensity=propensity,
            fractions=(top_fraction,),
        )
        table.append(
            {
                "uplift_weight": float(weight),
                "response_proxy_selected_mean": selected_response,
                "response_proxy_retention": (
                    selected_response / baseline_response if baseline_response else 0.0
                ),
                "uplift_at_target_fraction": metrics[f"uplift_at_{label}_percent"],
                "policy_incremental_at_target_fraction": metrics[
                    f"policy_incremental_at_{label}_percent"
                ],
                "policy_value_at_target_fraction": metrics[
                    f"policy_value_at_{label}_percent"
                ],
                "incremental_conversions_per_1000": metrics[
                    f"incremental_conversions_per_1000_at_{label}_percent"
                ],
            }
        )
    return table


def evaluate_score_mapping(
    scores: Mapping[str, np.ndarray],
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    propensity: float,
    fractions: Sequence[float],
) -> dict[str, dict[str, float]]:
    return {
        name: uplift_policy_metrics(
            values,
            treatment,
            outcome,
            propensity=propensity,
            fractions=fractions,
        )
        for name, values in scores.items()
    }
