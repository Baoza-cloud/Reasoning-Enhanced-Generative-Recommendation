from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TwoHeadOutcomeModel(nn.Module):
    """Shared representation with separate control and treatment outcome heads."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (128, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
            current_dim = hidden_dim
        self.representation = nn.Sequential(*layers)
        self.control_head = nn.Linear(current_dim, 1)
        self.treatment_head = nn.Linear(current_dim, 1)

    def potential_outcome_logits(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.representation(features)
        return (
            self.control_head(representation).squeeze(-1),
            self.treatment_head(representation).squeeze(-1),
        )

    def potential_outcomes(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        control, treated = self.potential_outcome_logits(features)
        return torch.sigmoid(control), torch.sigmoid(treated)

    def factual_logits(self, features: torch.Tensor, treatment: torch.Tensor) -> torch.Tensor:
        control, treated = self.potential_outcome_logits(features)
        return torch.where(treatment.bool(), treated, control)


class UpliftEffectModel(nn.Module):
    """Regress a doubly robust individual-effect pseudo-outcome."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (128, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def inverse_propensity_outcome_loss(
    model: TwoHeadOutcomeModel,
    features: torch.Tensor,
    treatment: torch.Tensor,
    outcome: torch.Tensor,
    propensity: torch.Tensor,
    *,
    min_propensity: float = 0.01,
) -> torch.Tensor:
    if not (treatment.shape == outcome.shape == propensity.shape == features.shape[:-1]):
        raise ValueError("batch tensors must align")
    clipped = propensity.clamp(min_propensity, 1 - min_propensity)
    weights = treatment / clipped + (1 - treatment) / (1 - clipped)
    per_example = F.binary_cross_entropy_with_logits(
        model.factual_logits(features, treatment),
        outcome,
        reduction="none",
    )
    return (per_example * weights).sum() / weights.sum().clamp_min(1e-8)
