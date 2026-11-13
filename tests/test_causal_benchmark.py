import numpy as np

from regenrec.causal_benchmark import (
    cross_fitted_uplift_scores,
    outcome_calibration_metrics,
    policy_pareto_table,
    stratified_poisson_bootstrap_difference,
    stratified_poisson_bootstrap_metrics,
    uplift_policy_metrics,
)


def test_uplift_policy_metrics_reward_correct_ordering() -> None:
    treatment = np.tile(np.array([1, 0]), 100)
    outcome = np.concatenate(
        (
            np.tile(np.array([1, 0]), 50),
            np.tile(np.array([0, 1]), 50),
        )
    )
    good = np.concatenate((np.arange(100, 0, -1), np.arange(0, -100, -1)))
    bad = -good
    good_metrics = uplift_policy_metrics(
        good, treatment, outcome, propensity=0.5, fractions=(0.5,)
    )
    bad_metrics = uplift_policy_metrics(
        bad, treatment, outcome, propensity=0.5, fractions=(0.5,)
    )
    assert good_metrics["qini_coefficient"] > bad_metrics["qini_coefficient"]
    assert good_metrics["policy_incremental_at_50_percent"] > 0


def test_cross_fitted_learners_return_finite_aligned_scores() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(400, 3)).astype(np.float32)
    treatment = rng.binomial(1, 0.5, size=400).astype(np.int8)
    logits = -2.0 + features[:, 0] + treatment * (0.5 + features[:, 1])
    probability = 1.0 / (1.0 + np.exp(-logits))
    outcome = rng.binomial(1, probability).astype(np.int8)
    result = cross_fitted_uplift_scores(
        features[:320],
        treatment[:320],
        outcome[:320],
        features[320:],
        propensity=float(treatment[:320].mean()),
        folds=2,
        seed=2026,
        max_iter=5,
        alpha=1e-4,
    )
    assert set(result["scores"]) == {
        "random",
        "response_model",
        "s_learner",
        "t_learner",
        "x_learner",
        "dr_learner",
    }
    for scores in result["scores"].values():
        assert scores.shape == (80,)
        assert np.isfinite(scores).all()


def test_calibration_bootstrap_and_pareto_are_reproducible() -> None:
    treatment = np.tile(np.array([1, 0]), 50)
    outcome = np.tile(np.array([1, 0, 0, 0]), 25)
    candidate = np.linspace(1.0, 0.0, 100)
    baseline = -candidate
    calibration = outcome_calibration_metrics(
        np.full(100, 0.1),
        np.full(100, 0.3),
        treatment,
        outcome,
    )
    assert calibration["rows"] == 100
    first = stratified_poisson_bootstrap_difference(
        candidate,
        baseline,
        treatment,
        outcome,
        propensity=0.5,
        samples=20,
        seed=9,
        metric_names=("qini_per_row",),
    )
    second = stratified_poisson_bootstrap_difference(
        candidate,
        baseline,
        treatment,
        outcome,
        propensity=0.5,
        samples=20,
        seed=9,
        metric_names=("qini_per_row",),
    )
    assert first == second
    intervals = stratified_poisson_bootstrap_metrics(
        candidate,
        treatment,
        outcome,
        propensity=0.5,
        samples=10,
        seed=10,
        metric_names=("qini_per_row",),
    )
    assert intervals["metrics"]["qini_per_row"]["ci95_low"] <= intervals["metrics"][
        "qini_per_row"
    ]["ci95_high"]
    table = policy_pareto_table(
        candidate,
        baseline,
        treatment,
        outcome,
        propensity=0.5,
        uplift_weights=(0.0, 0.5),
    )
    assert [row["uplift_weight"] for row in table] == [0.0, 0.5]
