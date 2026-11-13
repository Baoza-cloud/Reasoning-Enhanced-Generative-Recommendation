import pytest
import torch

from regenrec.causal import (
    candidate_reranking_readiness_report,
    causal_readiness_report,
    doubly_robust_pseudo_outcome,
    propensity_diagnostics,
    uplift_rerank,
)


def test_amazon_style_schema_is_not_causally_identifiable() -> None:
    report = causal_readiness_report(
        ["user_id", "parent_asin", "rating", "timestamp", "history"],
        field_mapping={
            "unit_id": "user_id",
            "item_id": "parent_asin",
            "event_time": "timestamp",
        },
    )
    assert report["status"] == "not_identifiable"
    assert "treatment" in report["missing_roles"]
    assert "outcome" in report["missing_roles"]


def test_doubly_robust_pseudo_outcome_matches_formula() -> None:
    outcome = torch.tensor([1.0, 0.0])
    treatment = torch.tensor([1.0, 0.0])
    propensity = torch.tensor([0.5, 0.5])
    mu0 = torch.tensor([0.2, 0.2])
    mu1 = torch.tensor([0.6, 0.6])
    result = doubly_robust_pseudo_outcome(outcome, treatment, propensity, mu0, mu1)
    assert torch.allclose(result, torch.tensor([1.2, 0.8]))


def test_doubly_robust_rejects_no_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        doubly_robust_pseudo_outcome(
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, 0.5]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.6, 0.6]),
        )


def test_uplift_rerank_combines_scores_deterministically() -> None:
    ranked = uplift_rerank(
        ["a", "b", "c"],
        [0.9, 0.8, 0.7],
        [-0.2, 0.3, 0.1],
        uplift_weight=1.0,
        top_k=2,
    )
    assert [row["item_id"] for row in ranked] == ["b", "c"]
    assert ranked[0]["final_score"] == pytest.approx(1.1)


def test_candidate_reranking_gate_requires_request_and_logging_fields() -> None:
    report = candidate_reranking_readiness_report(
        ["user_id", "parent_asin", "timestamp"],
        field_mapping={
            "unit_id": "user_id",
            "item_id": "parent_asin",
            "assignment_time": "timestamp",
        },
    )
    assert report["status"] == "not_identifiable"
    assert "request_id" in report["missing_roles"]
    assert "logging_policy_id" in report["missing_roles"]
    assert report["fixed_conversion_window"] is False


def test_propensity_diagnostics_reports_effective_sample_size() -> None:
    report = propensity_diagnostics(
        [1, 1, 0, 0],
        [0.5, 0.5, 0.5, 0.5],
    )
    assert report["overlap_passed"] is True
    assert report["ipw_ess"] == pytest.approx(4.0)
    assert report["treated_ipw_ess"] == pytest.approx(2.0)
    assert report["control_ipw_ess"] == pytest.approx(2.0)
