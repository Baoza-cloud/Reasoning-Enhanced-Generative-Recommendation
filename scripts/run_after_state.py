from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one command after a JSON state file reports completion"
    )
    parser.add_argument(
        "--dependency-state", type=Path, action="append", required=True
    )
    parser.add_argument("--completion-state", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def read_state(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a follow-up command is required")

    completion = read_state(args.completion_state)
    if completion and completion.get("status") in {"completed", "failed", "blocked"}:
        return 0

    dependencies = []
    for dependency_path in args.dependency_state:
        dependency = read_state(dependency_path)
        if dependency is None or dependency.get("status") == "running":
            return 75
        dependencies.append(
            {
                "dependency_state": str(dependency_path.resolve()),
                "dependency": dependency,
            }
        )
    incomplete = [
        dependency
        for dependency in dependencies
        if dependency["dependency"].get("status") != "completed"
    ]
    if incomplete:
        write_json(
            args.completion_state,
            {
                "status": "blocked",
                "reason": "one or more dependencies did not complete",
                "dependencies": dependencies,
            },
        )
        return 0

    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
