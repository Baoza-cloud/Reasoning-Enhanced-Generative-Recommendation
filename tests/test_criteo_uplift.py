import csv
import gzip

from regenrec.datasets.criteo_uplift import (
    CRITEO_FEATURES,
    CRITEO_LABELS,
    audit_criteo_uplift,
    iter_criteo_uplift_rows,
    load_criteo_uplift_tensors,
)


def test_criteo_uplift_parser_and_audit(tmp_path) -> None:
    path = tmp_path / "criteo.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*CRITEO_FEATURES, *CRITEO_LABELS])
        writer.writeheader()
        for treatment, conversion, visit, exposure in ((1, 1, 1, 1), (0, 0, 0, 0)):
            writer.writerow(
                {
                    **{feature: float(index) for index, feature in enumerate(CRITEO_FEATURES)},
                    "treatment": treatment,
                    "conversion": conversion,
                    "visit": visit,
                    "exposure": exposure,
                }
            )
    rows = list(iter_criteo_uplift_rows(path))
    assert len(rows) == 2
    assert rows[0]["f11"] == 11.0
    audit = audit_criteo_uplift(path)
    assert audit["status"] == "passed"
    assert audit["statistics"]["treated"] == 1
    assert audit["statistics"]["control"] == 1
    diagnostics = audit["randomized_assignment_diagnostics"]
    assert diagnostics["observed_treatment_probability"] == 0.5
    assert diagnostics["ipw_effective_sample_size"] == 2.0
    features, treatment, outcome = load_criteo_uplift_tensors(
        path,
        expected_rows=2,
        max_rows=None,
        sample_seed=11,
    )
    assert features.shape == (2, len(CRITEO_FEATURES))
    assert treatment.tolist() == [1.0, 0.0]
    assert outcome.tolist() == [1.0, 0.0]
