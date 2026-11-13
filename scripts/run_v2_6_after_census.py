from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V2.6 tests and seed-2026 fit only after census passes"
    )
    parser.add_argument("--census-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    state_path = args.output_dir / "state.json"
    existing = read_json(state_path)
    if existing and existing.get("status") in {"completed", "blocked", "failed"}:
        return 0
    census_state = read_json(args.census_dir / "state.json")
    if census_state is None or census_state.get("status") == "running":
        return 75
    if census_state.get("status") != "completed":
        write_json(
            state_path,
            {
                "status": "blocked",
                "reason": "census dependency did not complete",
                "test_evaluated": False,
                "dependency": census_state,
            },
        )
        return 0
    census_report = read_json(args.census_dir / "report.json")
    if census_report is None or not census_report.get("go_no_go", {}).get("passed"):
        write_json(
            state_path,
            {
                "status": "blocked",
                "reason": "census go/no-go gate failed",
                "test_evaluated": False,
                "census_report": str((args.census_dir / "report.json").resolve()),
            },
        )
        return 0
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = "src:."
    try:
        write_json(
            state_path,
            {
                "status": "running",
                "phase": "post_census_tests",
                "seed": args.seed,
                "test_evaluated": False,
            },
        )
        tests = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_v2_5_residual.py",
                "tests/test_v2_6_census.py",
                "tests/test_v2_6_full_hard_positive.py",
            ],
            check=False,
            env=environment,
        )
        if tests.returncode != 0:
            write_json(
                state_path,
                {
                    "status": "failed",
                    "phase": "post_census_tests",
                    "test_evaluated": False,
                    "returncode": tests.returncode,
                },
            )
            return 0
        command = [
            sys.executable,
            "scripts/train_v2_6_full_hard_positive.py",
            "--artifact-dir",
            str(args.artifact_dir),
            "--census-dir",
            str(args.census_dir),
            "--validation-cache",
            str(args.validation_cache),
            "--output-dir",
            str(args.output_dir),
            "--seed",
            str(args.seed),
            "--torch-threads",
            str(args.torch_threads),
        ]
        print("RUN", " ".join(command), flush=True)
        completed = subprocess.run(command, check=False, env=environment)
        if completed.returncode != 0 and not state_path.is_file():
            write_json(
                state_path,
                {
                    "status": "failed",
                    "phase": "seed_2026_fit",
                    "test_evaluated": False,
                    "returncode": completed.returncode,
                },
            )
        return 0
    except Exception as error:
        write_json(
            state_path,
            {
                "status": "failed",
                "phase": "v2_6_after_census",
                "test_evaluated": False,
                "error": repr(error),
            },
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
