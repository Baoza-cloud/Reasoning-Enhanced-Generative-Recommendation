from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import torch

from regenrec.config import load_config


GROUP_NAMES = ("head", "torso", "observed_tail", "cold_start")
SOURCE_NAMES = ("generator", "blair_observed_tail_retrieval", "union")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable full-Train V2.6 Beam+BLaIR hard-positive census"
    )
    parser.add_argument("--config", default="configs/v2_final_default.yaml")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Zero scans the complete Train file; positive values are for smoke tests.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--log-every-batches", type=int, default=25)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def nonempty_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def shard_name(start: int, stop: int) -> str:
    return f"shard_{start:09d}_{stop:09d}"


def shard_is_complete(directory: Path, *, start: int, examples: int) -> bool:
    state_path = directory / "state.json"
    report_path = directory / "report.json"
    rows_path = directory / "hard_positive_rows.pt"
    cache_path = directory / "candidate_cache.pt"
    if not all(path.is_file() for path in (state_path, report_path, rows_path, cache_path)):
        return False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return bool(
        state.get("status") == "completed"
        and state.get("test_evaluated") is False
        and report.get("test_evaluated") is False
        and report.get("split") == "train"
        and report.get("selection", {}).get("start_index") == start
        and report.get("examples") == examples
    )


def histogram(values: torch.Tensor, bins: Sequence[tuple[str, int, int | None]]) -> dict[str, int]:
    output: dict[str, int] = {}
    integer_values = values.long()
    for name, lower, upper in bins:
        selected = integer_values >= lower
        if upper is not None:
            selected &= integer_values <= upper
        output[name] = int(selected.sum())
    return output


def empty_oracle() -> dict[str, object]:
    return {
        source: {
            "overall": {"examples": 0, "hits": 0, "recall": 0.0},
            "target_groups": {
                group: {"examples": 0, "hits": 0, "recall": 0.0}
                for group in GROUP_NAMES
            },
        }
        for source in SOURCE_NAMES
    }


def merge_oracle(output: dict[str, object], current: dict[str, object]) -> None:
    for source in SOURCE_NAMES:
        for scope in ("overall",):
            output[source][scope]["examples"] += int(current[source][scope]["examples"])
            output[source][scope]["hits"] += int(current[source][scope]["hits"])
        for group in GROUP_NAMES:
            output[source]["target_groups"][group]["examples"] += int(
                current[source]["target_groups"][group]["examples"]
            )
            output[source]["target_groups"][group]["hits"] += int(
                current[source]["target_groups"][group]["hits"]
            )


def finish_oracle(output: dict[str, object]) -> None:
    for source in SOURCE_NAMES:
        rows = [output[source]["overall"], *output[source]["target_groups"].values()]
        for row in rows:
            row["recall"] = row["hits"] / row["examples"] if row["examples"] else 0.0


def summarize(output_dir: Path, expected_shards: Sequence[tuple[int, int]]) -> dict[str, object]:
    oracle = empty_oracle()
    reports: list[dict[str, object]] = []
    tensor_rows: dict[str, list[torch.Tensor]] = {
        "user_hashes": [],
        "target_catalog_indices": [],
        "groups": [],
        "retrieval_ranks": [],
        "history_lengths": [],
        "target_train_frequencies": [],
    }
    shard_records: list[dict[str, object]] = []
    reference_provenance: dict[str, object] | None = None
    for start, stop in expected_shards:
        directory = output_dir / "shards" / shard_name(start, stop)
        examples = stop - start
        if not shard_is_complete(directory, start=start, examples=examples):
            raise RuntimeError(f"incomplete census shard: {directory}")
        report_path = directory / "report.json"
        rows_path = directory / "hard_positive_rows.pt"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        payload = torch.load(rows_path, map_location="cpu", weights_only=True)
        if payload.get("metadata", {}).get("test_evaluated") is not False:
            raise ValueError(f"Test leakage flag in {rows_path}")
        provenance = report["provenance"]
        if reference_provenance is None:
            reference_provenance = provenance
        elif provenance != reference_provenance:
            raise ValueError(f"input provenance changed at {directory}")
        reports.append(report)
        merge_oracle(oracle, report["candidate_oracle"])
        for name in tensor_rows:
            tensor_rows[name].append(payload[name])
        shard_records.append(
            {
                "start_index": start,
                "stop_index": stop,
                "examples": examples,
                "hard_positive_rows": report["hard_positive_census"]["rows"],
                "report_sha256": sha256_file(report_path),
                "candidate_cache_sha256": report["candidate_cache"]["sha256"],
                "hard_positive_rows_sha256": report["hard_positive_rows"]["sha256"],
                "total_seconds": report["timing"]["total_seconds"],
            }
        )
    finish_oracle(oracle)
    merged = {
        name: torch.cat(rows) if rows else torch.empty(0, dtype=torch.int64)
        for name, rows in tensor_rows.items()
    }
    tail_mask = merged["groups"].long() == GROUP_NAMES.index("observed_tail")
    tail_users = merged["user_hashes"][tail_mask]
    tail_items = merged["target_catalog_indices"][tail_mask]
    hard_positive = {
        "definition": "target in BLaIR Observed-Tail-50 and not in Beam-20",
        "rows": int(len(merged["groups"])),
        "unique_users": int(torch.unique(merged["user_hashes"]).numel()),
        "unique_target_items": int(
            torch.unique(merged["target_catalog_indices"]).numel()
        ),
        "observed_tail_rows": int(tail_mask.sum()),
        "observed_tail_unique_users": int(torch.unique(tail_users).numel()),
        "observed_tail_unique_target_items": int(torch.unique(tail_items).numel()),
        "target_groups": {
            group: int((merged["groups"].long() == index).sum())
            for index, group in enumerate(GROUP_NAMES)
        },
        "retrieval_rank_histogram": histogram(
            merged["retrieval_ranks"],
            (("1-5", 1, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21-50", 21, 50)),
        ),
        "history_length_histogram": histogram(
            merged["history_lengths"],
            (("1-10", 1, 10), ("11-20", 11, 20), ("21-50", 21, 50)),
        ),
        "target_train_frequency_histogram": histogram(
            merged["target_train_frequencies"],
            (("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11+", 11, None)),
        ),
    }
    gates = {
        "observed_tail_rows_ge_500": hard_positive["observed_tail_rows"] >= 500,
        "observed_tail_unique_items_ge_100": hard_positive[
            "observed_tail_unique_target_items"
        ]
        >= 100,
        "observed_tail_unique_users_ge_500": hard_positive[
            "observed_tail_unique_users"
        ]
        >= 500,
    }
    return {
        "status": "completed",
        "protocol": "V2.6 full-Train hard-positive census",
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "category": reports[0]["category"],
        "split": "train",
        "test_evaluated": False,
        "synthetic_data_used": False,
        "seed": reports[0]["seed"],
        "examples": sum(int(report["examples"]) for report in reports),
        "shards": shard_records,
        "candidate_oracle": oracle,
        "hard_positive_census": hard_positive,
        "go_no_go": {"checks": gates, "passed": all(gates.values())},
        "timing": {
            "summed_shard_seconds": sum(
                float(report["timing"]["total_seconds"]) for report in reports
            )
        },
        "provenance": reference_provenance,
    }


def write_markdown(path: Path, report: dict[str, object]) -> None:
    hp = report["hard_positive_census"]
    oracle = report["candidate_oracle"]
    checks = report["go_no_go"]["checks"]
    lines = [
        "# V2.6 全量 Train hard-positive 普查",
        "",
        f"- 状态：{'通过阶段门禁' if report['go_no_go']['passed'] else '未通过阶段门禁'}",
        f"- 扫描真实 Train 样本：{report['examples']:,}",
        f"- hard positives：{hp['rows']:,}",
        f"- 唯一用户：{hp['unique_users']:,}",
        f"- 唯一目标商品：{hp['unique_target_items']:,}",
        "- Test evaluated：false",
        "",
        "## Candidate oracle",
        "",
        "| 来源 | Hits | Recall |",
        "|---|---:|---:|",
    ]
    for source in SOURCE_NAMES:
        row = oracle[source]["overall"]
        lines.append(f"| {source} | {row['hits']:,} | {row['recall']:.6f} |")
    lines.extend(
        [
            "",
            "## Go/no-go",
            "",
            *[f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items()],
            "",
            "只有全部门禁通过，才允许进入 seed 2026 Validation 联合排序训练。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if min(args.shard_size, args.batch_size, args.retrieval_batch_size, args.torch_threads) <= 0:
        raise ValueError("shard, batch and thread settings must be positive")
    if args.max_examples < 0:
        raise ValueError("max-examples must be non-negative")
    config = load_config(Path(args.config))
    train_path = Path(config["data"]["train_path"])
    total_available = nonempty_line_count(train_path)
    requested = min(args.max_examples, total_available) if args.max_examples else total_available
    expected_shards = [
        (start, min(start + args.shard_size, requested))
        for start in range(0, requested, args.shard_size)
    ]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = "src:."
    started = time.time()
    try:
        for shard_index, (start, stop) in enumerate(expected_shards, start=1):
            directory = output_dir / "shards" / shard_name(start, stop)
            examples = stop - start
            if shard_is_complete(directory, start=start, examples=examples):
                continue
            if args.summarize_only:
                raise RuntimeError(f"summarize-only found incomplete shard: {directory}")
            write_json(
                state_path,
                {
                    "status": "running",
                    "phase": "train_census_shards",
                    "seed": args.seed,
                    "test_evaluated": False,
                    "completed_shards": shard_index - 1,
                    "total_shards": len(expected_shards),
                    "current_start_index": start,
                    "current_stop_index": stop,
                },
            )
            command = [
                sys.executable,
                "scripts/analyze_v2_5_teacher_rescoring.py",
                "--config",
                str(args.config),
                "--artifact-dir",
                str(args.artifact_dir),
                "--seed",
                str(args.seed),
                "--split",
                "train",
                "--start-index",
                str(start),
                "--max-examples",
                str(examples),
                "--selection-strategy",
                "prefix",
                "--batch-size",
                str(args.batch_size),
                "--retrieval-batch-size",
                str(args.retrieval_batch_size),
                "--torch-threads",
                str(args.torch_threads),
                "--log-every-batches",
                str(args.log_every_batches),
                "--bootstrap-samples",
                "200",
                "--skip-teacher-scoring",
                "--output-dir",
                str(directory),
            ]
            print("RUN", " ".join(command), flush=True)
            subprocess.run(command, check=True, env=environment)
        write_json(
            state_path,
            {
                "status": "running",
                "phase": "aggregate",
                "seed": args.seed,
                "test_evaluated": False,
                "completed_shards": len(expected_shards),
                "total_shards": len(expected_shards),
            },
        )
        report = summarize(output_dir, expected_shards)
        report["wall_seconds"] = time.time() - started
        write_json(output_dir / "report.json", report)
        write_markdown(output_dir / "REPORT.md", report)
        write_json(
            state_path,
            {
                "status": "completed",
                "phase": None,
                "seed": args.seed,
                "test_evaluated": False,
                "examples": report["examples"],
                "go_no_go_passed": report["go_no_go"]["passed"],
                "report": str((output_dir / "report.json").resolve()),
            },
        )
        print(json.dumps(report["go_no_go"], ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        write_json(
            state_path,
            {
                "status": "failed",
                "phase": "v2_6_hard_positive_census",
                "seed": args.seed,
                "test_evaluated": False,
                "error": repr(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
