from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch


CAUSAL_FIELD_ROLES = (
    "unit_id",
    "item_id",
    "treatment",
    "outcome",
    "event_time",
    "pre_treatment_covariates",
)

CANDIDATE_CAUSAL_FIELD_ROLES = (
    "request_id",
    "unit_id",
    "item_id",
    "eligibility",
    "treatment",
    "outcome",
    "assignment_time",
    "conversion_time",
    "position",
    "logging_policy_id",
    "pre_treatment_covariates",
)


def causal_readiness_report(
    observed_fields: Sequence[str],
    *,
    field_mapping: Mapping[str, str | None] | None = None,
    randomized_assignment: bool = False,
    propensity_available: bool = False,
) -> dict[str, object]:
    """Audit whether logged data can identify an incremental treatment effect."""
    observed = set(observed_fields)
    mapping = dict(field_mapping or {})
    resolved = {
        role: source
        for role in CAUSAL_FIELD_ROLES
        if (source := mapping.get(role)) is not None and source in observed
    }
    missing = [role for role in CAUSAL_FIELD_ROLES if role not in resolved]
    reasons: list[str] = []
    if "treatment" in missing:
        reasons.append("no exposure/treatment assignment for both treated and eligible controls")
    if "outcome" in missing:
        reasons.append("no outcome observed in a fixed post-treatment attribution window")
    if "pre_treatment_covariates" in missing:
        reasons.append("no declared pre-treatment confounder feature set")
    if not randomized_assignment and not propensity_available:
        reasons.append("neither randomized assignment nor an auditable propensity is available")
    identifiable = not missing and (randomized_assignment or propensity_available)
    return {
        "status": "ready" if identifiable else "not_identifiable",
        "identifiable": identifiable,
        "observed_fields": sorted(observed),
        "field_mapping": mapping,
        "resolved_roles": resolved,
        "missing_roles": missing,
        "randomized_assignment": randomized_assignment,
        "propensity_available": propensity_available,
        "reasons": reasons,
        "guardrails": [
            "review presence must not be treated as randomized exposure",
            "unreviewed catalog items must not be labeled as non-conversions",
            "post-treatment features must not enter propensity or outcome models",
        ],
    }


def candidate_reranking_readiness_report(
    observed_fields: Sequence[str],
    *,
    field_mapping: Mapping[str, str | None] | None = None,
    randomized_assignment: bool = False,
    propensity_available: bool = False,
    fixed_conversion_window: bool = False,
) -> dict[str, object]:
    """Audit whether logs support candidate-level causal reranking.

    A generic treatment-effect table can be sufficient for an uplift benchmark but
    still be insufficient for recommendation reranking. This stricter gate requires
    request/candidate identity, eligibility, position, logging-policy provenance and
    auditable outcome timing so that non-exposure is never invented from catalog data.
    """
    observed = set(observed_fields)
    mapping = dict(field_mapping or {})
    resolved = {
        role: source
        for role in CANDIDATE_CAUSAL_FIELD_ROLES
        if (source := mapping.get(role)) is not None and source in observed
    }
    missing = [role for role in CANDIDATE_CAUSAL_FIELD_ROLES if role not in resolved]
    reasons: list[str] = []
    if "request_id" in missing:
        reasons.append("no request-level candidate set identifier")
    if "eligibility" in missing:
        reasons.append("no record of candidates eligible for treatment/control")
    if "treatment" in missing:
        reasons.append("no candidate-level treatment/exposure assignment")
    if "position" in missing or "logging_policy_id" in missing:
        reasons.append("position or logging-policy provenance is unavailable")
    if "assignment_time" in missing or "conversion_time" in missing:
        reasons.append("assignment/conversion timestamps cannot support attribution")
    if not fixed_conversion_window:
        reasons.append("no fixed post-assignment conversion window is declared")
    if not randomized_assignment and not propensity_available:
        reasons.append("neither randomization nor auditable propensity is available")
    identifiable = (
        not missing
        and fixed_conversion_window
        and (randomized_assignment or propensity_available)
    )
    return {
        "status": "ready" if identifiable else "not_identifiable",
        "identifiable": identifiable,
        "audit_scope": "candidate_level_causal_reranking",
        "observed_fields": sorted(observed),
        "field_mapping": mapping,
        "resolved_roles": resolved,
        "missing_roles": missing,
        "randomized_assignment": randomized_assignment,
        "propensity_available": propensity_available,
        "fixed_conversion_window": fixed_conversion_window,
        "reasons": reasons,
        "guardrails": [
            "catalog membership is not candidate eligibility",
            "missing exposure is not a control observation",
            "review presence is not randomized treatment",
            "only pre-treatment features may enter propensity and uplift models",
        ],
    }


def propensity_diagnostics(
    treatment: Sequence[int],
    propensity: Sequence[float],
    *,
    min_propensity: float = 0.01,
) -> dict[str, float | int | bool]:
    """Return overlap and inverse-propensity effective-sample-size diagnostics."""
    if len(treatment) != len(propensity) or not treatment:
        raise ValueError("treatment and propensity must be non-empty and aligned")
    if not 0 < min_propensity < 0.5:
        raise ValueError("min_propensity must be in (0, 0.5)")
    normalized_treatment = [int(value) for value in treatment]
    normalized_propensity = [float(value) for value in propensity]
    if any(value not in (0, 1) for value in normalized_treatment):
        raise ValueError("treatment must be binary")
    if len(set(normalized_treatment)) != 2:
        raise ValueError("treated and control observations are both required")
    if any(not math.isfinite(value) for value in normalized_propensity):
        raise ValueError("propensity must be finite")

    violations = sum(
        value < min_propensity or value > 1 - min_propensity
        for value in normalized_propensity
    )
    clipped = [
        min(max(value, min_propensity), 1 - min_propensity)
        for value in normalized_propensity
    ]
    weights = [
        1.0 / score if arm == 1 else 1.0 / (1.0 - score)
        for arm, score in zip(normalized_treatment, clipped, strict=True)
    ]

    def effective_sample_size(values: Sequence[float]) -> float:
        total = sum(values)
        squared_total = sum(value * value for value in values)
        return total * total / squared_total if squared_total else 0.0

    treated_weights = [
        weight
        for weight, arm in zip(weights, normalized_treatment, strict=True)
        if arm == 1
    ]
    control_weights = [
        weight
        for weight, arm in zip(weights, normalized_treatment, strict=True)
        if arm == 0
    ]
    return {
        "rows": len(weights),
        "treated": sum(normalized_treatment),
        "control": len(weights) - sum(normalized_treatment),
        "propensity_min": min(normalized_propensity),
        "propensity_max": max(normalized_propensity),
        "overlap_bound": min_propensity,
        "overlap_violations": violations,
        "overlap_passed": violations == 0,
        "ipw_ess": effective_sample_size(weights),
        "treated_ipw_ess": effective_sample_size(treated_weights),
        "control_ipw_ess": effective_sample_size(control_weights),
    }


def validate_causal_batch(
    treatment: torch.Tensor,
    outcome: torch.Tensor,
    propensity: torch.Tensor,
    *,
    min_propensity: float = 0.01,
) -> None:
    if treatment.shape != outcome.shape or treatment.shape != propensity.shape:
        raise ValueError("treatment, outcome, and propensity must have the same shape")
    if treatment.numel() == 0:
        raise ValueError("causal batch cannot be empty")
    if not torch.all((treatment == 0) | (treatment == 1)):
        raise ValueError("treatment must be binary")
    if not torch.all((outcome == 0) | (outcome == 1)):
        raise ValueError("outcome must be binary")
    if not torch.isfinite(propensity).all():
        raise ValueError("propensity must be finite")
    if not torch.all((propensity >= min_propensity) & (propensity <= 1 - min_propensity)):
        raise ValueError("propensity violates configured overlap bounds")
    if treatment.min() == treatment.max():
        raise ValueError("causal batch must contain treated and control observations")


def doubly_robust_pseudo_outcome(
    outcome: torch.Tensor,
    treatment: torch.Tensor,
    propensity: torch.Tensor,
    outcome_if_control: torch.Tensor,
    outcome_if_treated: torch.Tensor,
    *,
    min_propensity: float = 0.01,
) -> torch.Tensor:
    """Compute the AIPW/DR individual-effect pseudo-outcome."""
    validate_causal_batch(
        treatment,
        outcome,
        propensity,
        min_propensity=min_propensity,
    )
    for prediction in (outcome_if_control, outcome_if_treated):
        if prediction.shape != outcome.shape:
            raise ValueError("outcome model predictions must match outcome shape")
        if not torch.isfinite(prediction).all():
            raise ValueError("outcome model predictions must be finite")
    clipped_propensity = propensity.clamp(min_propensity, 1 - min_propensity)
    return (
        outcome_if_treated
        - outcome_if_control
        + treatment * (outcome - outcome_if_treated) / clipped_propensity
        - (1 - treatment) * (outcome - outcome_if_control) / (1 - clipped_propensity)
    )


def uplift_rerank(
    item_ids: Sequence[str],
    relevance_scores: Sequence[float],
    uplift_scores: Sequence[float],
    *,
    uplift_weight: float,
    top_k: int | None = None,
) -> list[dict[str, float | str]]:
    """Rerank candidates by relevance plus predicted incremental conversion."""
    if not (len(item_ids) == len(relevance_scores) == len(uplift_scores)):
        raise ValueError("item IDs, relevance scores, and uplift scores must align")
    if not item_ids:
        raise ValueError("candidate list cannot be empty")
    if uplift_weight < 0:
        raise ValueError("uplift_weight must be non-negative")
    limit = len(item_ids) if top_k is None else top_k
    if limit <= 0:
        raise ValueError("top_k must be positive")
    rows = [
        {
            "item_id": str(item_id),
            "relevance_score": float(relevance),
            "uplift_score": float(uplift),
            "final_score": float(relevance + uplift_weight * uplift),
        }
        for item_id, relevance, uplift in zip(
            item_ids, relevance_scores, uplift_scores, strict=True
        )
    ]
    rows.sort(key=lambda row: (-float(row["final_score"]), str(row["item_id"])))
    return rows[:limit]
