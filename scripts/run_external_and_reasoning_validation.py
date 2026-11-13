from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Preference Reasoning ablation then frozen multi-category validation"
    )
    parser.add_argument(
        "--order",
        choices=("reasoning-first", "external-first"),
        default="reasoning-first",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run the two independent category tracks concurrently",
    )
    return parser.parse_args()


def write_state(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"updated_at": datetime.now(timezone.utc).isoformat(), **payload},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    tasks = {
        "preference_reasoning_ablation": [
            sys.executable,
            "scripts/run_preference_reasoning_ablation.py",
        ],
        "multicategory_external_validation": [
            sys.executable,
            "scripts/run_multicategory_external_validation.py",
        ],
    }
    order = (
        tuple(tasks)
        if args.order == "reasoning-first"
        else tuple(reversed(tuple(tasks)))
    )
    state_path = ROOT / "artifacts/external_and_reasoning_validation_state.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed: list[str] = []
    try:
        if args.parallel:
            write_state(
                state_path,
                status="running",
                current_tasks=list(order),
                parallel=True,
                order=list(order),
                completed=completed,
            )
            processes = {
                task: subprocess.Popen(tasks[task], cwd=ROOT, env=environment)
                for task in order
            }
            failures: dict[str, int] = {}
            for task, process in processes.items():
                return_code = process.wait()
                if return_code == 0:
                    completed.append(task)
                else:
                    failures[task] = return_code
                write_state(
                    state_path,
                    status="running" if len(completed) + len(failures) < len(order) else "finalizing",
                    current_tasks=[
                        name for name in order if name not in completed and name not in failures
                    ],
                    parallel=True,
                    order=list(order),
                    completed=completed,
                    failures=failures,
                )
            if failures:
                raise RuntimeError(f"parallel experiment tracks failed: {failures}")
            write_state(
                state_path,
                status="completed",
                current_tasks=[],
                parallel=True,
                order=list(order),
                completed=completed,
            )
            return
        for task in order:
            write_state(
                state_path,
                status="running",
                current_task=task,
                order=list(order),
                completed=completed,
            )
            subprocess.run(tasks[task], cwd=ROOT, env=environment, check=True)
            completed.append(task)
        write_state(
            state_path,
            status="completed",
            current_task=None,
            order=list(order),
            completed=completed,
        )
    except Exception as error:
        write_state(
            state_path,
            status="failed",
            current_task=locals().get("task"),
            order=list(order),
            completed=completed,
            error=repr(error),
        )
        raise


if __name__ == "__main__":
    main()
