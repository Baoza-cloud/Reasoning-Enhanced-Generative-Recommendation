from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = (
    "README.md",
    "EXPERIMENTS.md",
    "docs/V2_FINAL_REPORT.md",
    "configs/v2_final_default.yaml",
    "configs/v2_final_modes.yaml",
    "data/amazon2023/processed/Musical_Instruments/last_out_v2/data_audit.json",
    "artifacts/candidate_reranking_mild_v1/v2_pareto_summary.json",
    "artifacts/candidate_reranking_mild_v1/acceptance.json",
    "artifacts/v2_final_report/error_analysis.json",
    "artifacts/v2_final_report/provenance.json",
    "artifacts/multicategory_external_validation_v1/summary.json",
    "artifacts/preference_reasoning_ablation_v1/summary.json",
    "artifacts/preference_reasoning_interpretability_v2/summary.json",
    "artifacts/preference_reasoning_mechanism_ablation_v1/summary.json",
    "artifacts/preference_reasoning_retraining_ablation_v1/summary.json",
)

V2_4_FILES = (
    "docs/V2_4_ENHANCEMENT_PROTOCOL.md",
    "configs/v2_4_adaptive_forgetting.yaml",
    "scripts/freeze_v2_evidence.py",
    "scripts/run_v2_4_dynamic_forgetting.py",
    "scripts/run_v2_4_hybrid_scorer_multiseed.py",
    "scripts/run_v2_4_test_confirmation.py",
    "scripts/prepare_v2_4_external_categories.py",
    "scripts/run_unisrec_multiseed.py",
    "artifacts/v2_4_adaptive_forgetting/summary.json",
    "artifacts/v2_4_adaptive_forgetting/state.json",
    "artifacts/v2_4_hybrid_scorer_multiseed/summary.json",
    "artifacts/v2_4_hybrid_scorer_multiseed/state.json",
    "artifacts/v2_4_external_preparation/state.json",
    "artifacts/unisrec_amazon2023_musical_multiseed/state.json",
    "artifacts/v2_4_external_validation/state.json",
    "artifacts/v2_4_test_confirmation/protocol.json",
    "artifacts/v2_4_test_confirmation/state.json",
    "artifacts/v2_4_test_confirmation/summary.json",
    "artifacts/v2_4_test_confirmation/paired_tests.json",
    "artifacts/v2_4_test_confirmation/REPORT.md",
    "artifacts/v2_4_test_confirmation/seed_2026/report.json",
    "artifacts/v2_4_test_confirmation/seed_2026/hit_indicators.jsonl",
    "artifacts/v2_4_test_confirmation/seed_2027/report.json",
    "artifacts/v2_4_test_confirmation/seed_2027/hit_indicators.jsonl",
    "artifacts/v2_4_test_confirmation/seed_2028/report.json",
    "artifacts/v2_4_test_confirmation/seed_2028/hit_indicators.jsonl",
    "data/amazon2023/processed/All_Beauty/last_out_v2/data_audit.json",
    "data/amazon2023/processed/All_Beauty/last_out_v2/item_embeddings_blair.pt",
    "data/amazon2023/processed/Toys_and_Games/last_out_v2/data_audit.json",
    "data/amazon2023/processed/Toys_and_Games/last_out_v2/item_embeddings_blair.pt",
    "data/amazon2023/processed/Sports_and_Outdoors/last_out_v2/data_audit.json",
    "data/amazon2023/processed/Sports_and_Outdoors/last_out_v2/item_embeddings_blair.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the auditable V2 evidence snapshot")
    parser.add_argument("--freeze-id", default="v2.3-pr-retraining-final")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/v2_final_report/v2_evidence_freeze_v2_3.json"),
    )
    parser.add_argument(
        "--include-v2-4",
        action="store_true",
        help="also freeze the concluded V2.4 validation-only enhancement evidence",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    files = DEFAULT_FILES + (V2_4_FILES if args.include_v2_4 else ())
    missing = [relative for relative in files if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze V2; missing evidence files: {missing}")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "frozen",
        "freeze_id": args.freeze_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "V2 main ranking, long-tail, external validation, corrected Preference "
            "Reasoning interpretation, frozen-checkpoint mechanism intervention, "
            "independently retrained fine-grained architecture ablations"
            + (
                ", and the concluded V2.4 adaptive-forgetting and learned-hybrid "
                "experiments, their post-freeze one-shot Test confirmation, plus "
                "audited external-data preparation"
                if args.include_v2_4
                else ""
            )
        ),
        "data_policy": {
            "formal_results_require_real_data": True,
            "synthetic_data_used": False,
            "test_not_used_for_hyperparameter_selection": True,
        },
        "scientific_boundary": (
            "No fine-grained Preference Reasoning component has a stable independent "
            "Recall@10 advantage after three-seed retraining. V2.4 adaptive forgetting "
            "did not stably improve Recall@10 (96.08% Test retention), the learned "
            "hybrid retained only 56.69% of Test Recall@10 and was rejected, UniSRec produced no "
            "formal result after a user-requested resource stop, and Beauty/Toys/Sports "
            "were prepared but not model-evaluated. Original V2 remains the default; "
            "all findings remain offline predictive evidence rather than causal "
            "preference attribution."
            if args.include_v2_4
            else (
                "No fine-grained Preference Reasoning component has a stable independent "
                "Recall@10 advantage after three-seed retraining; all findings remain "
                "offline predictive evidence rather than causal preference attribution."
            )
        ),
        "pending_evidence": [],
        "files": {
            relative: {
                "sha256": sha256_file(ROOT / relative),
                "bytes": (ROOT / relative).stat().st_size,
            }
            for relative in files
        },
    }
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
