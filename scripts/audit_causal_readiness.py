from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

from regenrec.causal import candidate_reranking_readiness_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit V3 causal identifiability")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/amazon2023/processed/Musical_Instruments/last_out_v2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/causal_readiness_amazon2023.json"),
    )
    return parser.parse_args()


def source_fields(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    if path.suffix == ".gz":
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle).fieldnames or ())
    with opener(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle).fieldnames or ())


def main() -> None:
    args = parse_args()
    manifest_path = args.processed_dir / "manifest.json"
    audit_path = args.processed_dir / "data_audit.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("synthetic_data_used") is not False:
        raise ValueError("causal readiness audit requires a passed real-data audit")
    train_source = Path(manifest["source_files"]["train"])
    observed_fields = source_fields(train_source)
    readiness = candidate_reranking_readiness_report(
        observed_fields,
        field_mapping={
            "request_id": None,
            "unit_id": "user_id",
            "item_id": "parent_asin",
            "eligibility": None,
            "treatment": None,
            "outcome": None,
            "assignment_time": "timestamp",
            "conversion_time": None,
            "position": None,
            "logging_policy_id": None,
            "pre_treatment_covariates": None,
        },
        randomized_assignment=False,
        propensity_available=False,
        fixed_conversion_window=False,
    )
    report = {
        "dataset": manifest["source_dataset"],
        "category": manifest["category"],
        "synthetic_data_used": False,
        "source_manifest": str(manifest_path.resolve()),
        "source_data_audit": str(audit_path.resolve()),
        **readiness,
        "decision": (
            "Amazon Reviews 2023 remains valid for V1/V2 ranking, but must not be used "
            "alone to claim causal uplift. V3 training is gated until exposure/control logs "
            "with conversion windows and pre-treatment covariates are supplied."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
