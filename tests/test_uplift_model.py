import pytest
import torch

from regenrec.metrics import uplift_ranking_metrics
from regenrec.models.uplift import (
    TwoHeadOutcomeModel,
    UpliftEffectModel,
    inverse_propensity_outcome_loss,
)


def test_uplift_models_train_and_return_aligned_outputs() -> None:
    torch.manual_seed(5)
    features = torch.randn(8, 4)
    treatment = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
    outcome = torch.tensor([0, 1, 0, 1, 0, 0, 1, 1], dtype=torch.float32)
    propensity = torch.full((8,), 0.5)
    outcome_model = TwoHeadOutcomeModel(4, (8,))
    loss = inverse_propensity_outcome_loss(
        outcome_model,
        features,
        treatment,
        outcome,
        propensity,
    )
    assert torch.isfinite(loss)
    loss.backward()
    mu0, mu1 = outcome_model.potential_outcomes(features)
    assert mu0.shape == mu1.shape == outcome.shape
    effect_model = UpliftEffectModel(4, (8,))
    assert effect_model(features).shape == outcome.shape


def test_uplift_ranking_metrics_reward_good_ordering() -> None:
    treatment = [1, 0, 1, 0, 1, 0, 1, 0]
    outcome = [1, 0, 1, 0, 0, 1, 0, 1]
    good = uplift_ranking_metrics(
        [8, 7, 6, 5, 4, 3, 2, 1],
        treatment,
        outcome,
        top_fraction=0.5,
    )
    bad = uplift_ranking_metrics(
        [1, 2, 3, 4, 5, 6, 7, 8],
        treatment,
        outcome,
        top_fraction=0.5,
    )
    assert good["uplift_at_50_percent"] == pytest.approx(1.0)
    assert good["qini_coefficient"] > bad["qini_coefficient"]
