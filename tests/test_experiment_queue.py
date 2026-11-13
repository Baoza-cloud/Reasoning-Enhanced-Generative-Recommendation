from scripts.run_remaining_experiments import experiment_steps
from scripts.run_multicategory_validation import prerequisite_paths, validation_steps


def test_experiment_queue_is_serial_and_has_reports() -> None:
    steps = experiment_steps()
    assert [step["name"] for step in steps] == [
        "v2_main",
        "v2_no_intensity_ablation",
        "sasrec_full",
        "criteo_uplift_full",
    ]
    assert steps[0]["wait_only"] is True
    assert all(str(step["report"]).endswith("report.json") for step in steps)


def test_multicategory_validation_queue_is_serial_and_real_data_first() -> None:
    assert len(prerequisite_paths()) == 2
    steps = validation_steps()
    assert [step["name"] for step in steps] == [
        "audit_real_data",
        "blair_embeddings",
        "popularity_markov_baselines",
        "rqvae_v2_beam20",
        "sasrec",
    ]
    assert any(str(part).endswith("audit_amazon2023.py") for part in steps[0]["command"])
    assert all("report" in step and "command" in step for step in steps)
