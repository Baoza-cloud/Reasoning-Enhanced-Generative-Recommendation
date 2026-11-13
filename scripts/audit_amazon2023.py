from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from regenrec.datasets.amazon2023 import DATA_SPLITS, benchmark_url, metadata_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local Amazon Reviews 2023 provenance")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/amazon2023/processed/Musical_Instruments/last_out"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.processed_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    category = str(manifest["category"])
    split_strategy = str(manifest["split_strategy"])
    if manifest["source_dataset"] != "McAuley-Lab/Amazon-Reviews-2023":
        raise ValueError("unexpected source_dataset in manifest")

    item_ids: set[str] = set()
    item_rows = 0
    with (args.processed_dir / "items.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item_id = str(json.loads(line)["item_id"])
            if item_id in item_ids:
                raise ValueError(f"duplicate item_id in items.jsonl: {item_id}")
            item_ids.add(item_id)
            item_rows += 1

    processed_records: dict[str, object] = {
        "items": {**file_record(args.processed_dir / "items.jsonl"), "rows": item_rows}
    }
    unknown_references = 0
    invalid_behavior_ids = 0
    invalid_intensities = 0
    invalid_time_gap_buckets = 0
    allowed_behaviors = set(int(value) for value in manifest["behavior_mapping"].values())
    num_time_buckets = int(manifest["parameters"]["num_time_buckets"])
    for split in DATA_SPLITS:
        path = args.processed_dir / f"{split}.jsonl"
        rows = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                rows += 1
                referenced = [str(payload["target_item_id"])]
                referenced.extend(str(event["item_id"]) for event in payload["history"])
                unknown_references += sum(item_id not in item_ids for item_id in referenced)
                for event in payload["history"]:
                    invalid_behavior_ids += int(int(event["behavior"]) not in allowed_behaviors)
                    invalid_time_gap_buckets += int(
                        not 0 <= int(event["time_gap_bucket"]) < num_time_buckets
                    )
                    if int(manifest.get("format_version", 1)) >= 2:
                        intensity = float(event.get("behavior_intensity", -1.0))
                        invalid_intensities += int(not 0.0 <= intensity <= 1.0)
        processed_records[split] = {**file_record(path), "rows": rows}

    source_records: dict[str, object] = {}
    for split in DATA_SPLITS:
        source_records[split] = {
            **file_record(Path(manifest["source_files"][split])),
            "official_url": benchmark_url(category, split_strategy, split),
        }
    source_records["metadata"] = {
        **file_record(Path(manifest["source_files"]["metadata"])),
        "official_url": metadata_url(category),
    }

    expected_items = int(manifest["items"]["total"])
    checks_passed = (
        unknown_references == 0
        and item_rows == expected_items
        and invalid_behavior_ids == 0
        and invalid_intensities == 0
        and invalid_time_gap_buckets == 0
    )
    audit = {
        "status": "passed" if checks_passed else "failed",
        "source_dataset": manifest["source_dataset"],
        "data_kind": "observed Amazon review interactions and product metadata",
        "synthetic_data_used": False,
        "category": category,
        "filter": manifest["filter"],
        "split_strategy": split_strategy,
        "source_files": source_records,
        "processed_files": processed_records,
        "checks": {
            "expected_items": expected_items,
            "actual_unique_items": item_rows,
            "unknown_item_references": unknown_references,
            "all_referenced_items_in_catalog": unknown_references == 0,
            "invalid_behavior_ids": invalid_behavior_ids,
            "invalid_behavior_intensities": invalid_intensities,
            "invalid_time_gap_buckets": invalid_time_gap_buckets,
            "observed_feature_integrity": (
                invalid_behavior_ids == 0
                and invalid_intensities == 0
                and invalid_time_gap_buckets == 0
            ),
        },
    }
    output_path = args.output or args.processed_dir / "data_audit.json"
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
