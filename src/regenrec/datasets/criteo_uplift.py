from __future__ import annotations

import csv
import gzip
import hashlib
import math
import random
import shutil
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

CRITEO_UPLIFT_DATASET_ID = "criteo/criteo-uplift"
CRITEO_UPLIFT_FILENAME = "criteo-research-uplift-v2.1.csv.gz"
CRITEO_UPLIFT_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    f"{CRITEO_UPLIFT_FILENAME}?download=true"
)
CRITEO_FEATURES = tuple(f"f{index}" for index in range(12))
CRITEO_LABELS = ("treatment", "conversion", "visit", "exposure")


@dataclass
class CriteoUpliftAuditStats:
    rows: int = 0
    treated: int = 0
    control: int = 0
    conversions: int = 0
    treated_conversions: int = 0
    control_conversions: int = 0
    visits: int = 0
    exposures: int = 0
    invalid_binary_values: int = 0
    non_finite_features: int = 0
    conversion_without_visit: int = 0
    exposure_in_control: int = 0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_criteo_uplift(
    destination: str | Path,
    *,
    timeout: int = 180,
) -> Path:
    output_path = Path(destination)
    if output_path.is_file() and output_path.stat().st_size > 0:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".part")
    request = urllib.request.Request(CRITEO_UPLIFT_URL, headers={"User-Agent": "regenrec/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.replace(output_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return output_path


def iter_criteo_uplift_rows(
    path: str | Path,
    *,
    max_rows: int | None = None,
) -> Iterator[dict[str, float | int]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set((*CRITEO_FEATURES, *CRITEO_LABELS))
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Criteo uplift data is missing columns: {sorted(missing)}")
        for row_index, row in enumerate(reader):
            if max_rows is not None and row_index >= max_rows:
                return
            yield {
                **{feature: float(row[feature]) for feature in CRITEO_FEATURES},
                **{label: int(row[label]) for label in CRITEO_LABELS},
            }


def load_criteo_uplift_tensors(
    path: str | Path,
    *,
    expected_rows: int,
    max_rows: int | None,
    sample_seed: int,
    chunk_size: int = 50_000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load all audited rows or an exact uniform reservoir sample into tensors."""
    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")
    capacity = min(expected_rows, max_rows) if max_rows is not None else expected_rows
    features = torch.empty((capacity, len(CRITEO_FEATURES)), dtype=torch.float32)
    treatment = torch.empty(capacity, dtype=torch.float32)
    outcome = torch.empty(capacity, dtype=torch.float32)
    if max_rows is not None and max_rows < expected_rows:
        generator = random.Random(sample_seed)
        rows_seen = 0
        for row_index, row in enumerate(iter_criteo_uplift_rows(path)):
            if row_index < capacity:
                destination = row_index
            else:
                destination = generator.randrange(row_index + 1)
                if destination >= capacity:
                    rows_seen = row_index + 1
                    continue
            features[destination] = torch.tensor(
                [float(row[name]) for name in CRITEO_FEATURES], dtype=torch.float32
            )
            treatment[destination] = float(row["treatment"])
            outcome[destination] = float(row["conversion"])
            rows_seen = row_index + 1
        if rows_seen != expected_rows:
            raise ValueError(
                f"audited row count changed: expected {expected_rows}, observed {rows_seen}"
            )
        return features, treatment, outcome

    feature_buffer: list[list[float]] = []
    treatment_buffer: list[float] = []
    outcome_buffer: list[float] = []
    written = 0

    def flush() -> None:
        nonlocal written
        if not feature_buffer:
            return
        count = len(feature_buffer)
        end = written + count
        features[written:end] = torch.tensor(feature_buffer, dtype=torch.float32)
        treatment[written:end] = torch.tensor(treatment_buffer, dtype=torch.float32)
        outcome[written:end] = torch.tensor(outcome_buffer, dtype=torch.float32)
        written = end
        feature_buffer.clear()
        treatment_buffer.clear()
        outcome_buffer.clear()

    for row in iter_criteo_uplift_rows(path, max_rows=max_rows):
        feature_buffer.append([float(row[name]) for name in CRITEO_FEATURES])
        treatment_buffer.append(float(row["treatment"]))
        outcome_buffer.append(float(row["conversion"]))
        if len(feature_buffer) == chunk_size:
            flush()
    flush()
    if written == 0:
        raise ValueError("Criteo uplift file yielded no rows")
    return features[:written], treatment[:written], outcome[:written]


def audit_criteo_uplift(
    path: str | Path,
    *,
    max_rows: int | None = None,
) -> dict[str, object]:
    input_path = Path(path)
    stats = CriteoUpliftAuditStats()
    for row in iter_criteo_uplift_rows(input_path, max_rows=max_rows):
        stats.rows += 1
        treatment = int(row["treatment"])
        conversion = int(row["conversion"])
        visit = int(row["visit"])
        exposure = int(row["exposure"])
        binary_values = (treatment, conversion, visit, exposure)
        stats.invalid_binary_values += sum(value not in (0, 1) for value in binary_values)
        stats.treated += treatment == 1
        stats.control += treatment == 0
        stats.conversions += conversion
        stats.treated_conversions += conversion if treatment == 1 else 0
        stats.control_conversions += conversion if treatment == 0 else 0
        stats.visits += visit
        stats.exposures += exposure
        stats.conversion_without_visit += conversion == 1 and visit == 0
        stats.exposure_in_control += exposure == 1 and treatment == 0
        stats.non_finite_features += sum(
            not math.isfinite(float(row[feature])) for feature in CRITEO_FEATURES
        )
    overlap = stats.treated > 0 and stats.control > 0
    passed = (
        stats.rows > 0
        and overlap
        and stats.invalid_binary_values == 0
        and stats.non_finite_features == 0
        and stats.conversion_without_visit == 0
        and stats.exposure_in_control == 0
    )
    propensity = stats.treated / stats.rows if stats.rows else 0.0
    treated_rate = stats.treated_conversions / stats.treated if stats.treated else 0.0
    control_rate = stats.control_conversions / stats.control if stats.control else 0.0
    ipw_ess = (
        4.0 * stats.rows * propensity * (1.0 - propensity)
        if 0.0 < propensity < 1.0
        else 0.0
    )
    return {
        "status": "passed" if passed else "failed",
        "source_dataset": CRITEO_UPLIFT_DATASET_ID,
        "source_url": CRITEO_UPLIFT_URL,
        "source_file": str(input_path.resolve()),
        "source_bytes": input_path.stat().st_size,
        "source_sha256": sha256_file(input_path),
        "synthetic_data_used": False,
        "max_rows": max_rows,
        "features": list(CRITEO_FEATURES),
        "labels": list(CRITEO_LABELS),
        "statistics": asdict(stats),
        "randomized_assignment_diagnostics": {
            "observed_treatment_probability": propensity,
            "minimum_arm_probability": min(propensity, 1.0 - propensity),
            "overlap_passed_at_0_01": 0.01 <= propensity <= 0.99,
            "treated_conversion_rate": treated_rate,
            "control_conversion_rate": control_rate,
            "raw_conversion_rate_difference": treated_rate - control_rate,
            "ipw_effective_sample_size": ipw_ess,
            "ipw_effective_sample_fraction": ipw_ess / stats.rows if stats.rows else 0.0,
            "treated_effective_sample_size": float(stats.treated),
            "control_effective_sample_size": float(stats.control),
        },
        "checks": {
            "treated_and_control_present": overlap,
            "binary_fields_valid": stats.invalid_binary_values == 0,
            "features_finite": stats.non_finite_features == 0,
            "conversion_implies_visit": stats.conversion_without_visit == 0,
            "exposure_only_in_treatment": stats.exposure_in_control == 0,
        },
        "identifiability_note": (
            "Randomized incrementality-test assignment supports uplift benchmarking. "
            "Official non-uniform subsampling prevents recovery of the original absolute "
            "incrementality level; evaluate ranking/policy metrics, not population ATE."
        ),
    }
