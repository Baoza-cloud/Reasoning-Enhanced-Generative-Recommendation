from __future__ import annotations

import csv
import gzip
import json
import math
import re
import shutil
import urllib.request
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

from regenrec.data import Interaction, SequenceExample, write_sequence_examples_jsonl

AMAZON2023_DATASET_ID = "McAuley-Lab/Amazon-Reviews-2023"
AMAZON2023_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
AMAZON2023_BEHAVIOR = {"padding": 0, "review": 1}
AMAZON2023_RATING_BEHAVIOR = {
    "padding": 0,
    "negative_review_1_2": 1,
    "neutral_review_3": 2,
    "positive_review_4_5": 3,
}
SPLIT_STRATEGIES = frozenset({"last_out", "timestamp"})
DATA_SPLITS = ("train", "valid", "test")


@dataclass
class AmazonPreparationStats:
    rows_seen: int = 0
    rows_without_history: int = 0
    rows_without_target: int = 0
    examples_emitted: int = 0


@dataclass
class AmazonIntensityPreparationStats(AmazonPreparationStats):
    history_events: int = 0
    negative_events: int = 0
    neutral_events: int = 0
    positive_events: int = 0
    missing_history_metadata: int = 0
    future_history_events: int = 0


@dataclass(frozen=True)
class AmazonObservedEvent:
    rating: float
    timestamp_ms: int


def _validate_category(category: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", category):
        raise ValueError("category must contain only letters, numbers, and underscores")
    return category


def benchmark_url(category: str, split_strategy: str, split: str) -> str:
    category = _validate_category(category)
    if split_strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"unsupported split strategy: {split_strategy}")
    if split not in DATA_SPLITS:
        raise ValueError(f"unsupported data split: {split}")
    return f"{AMAZON2023_BASE_URL}/benchmark/5core/{split_strategy}_w_his/{category}.{split}.csv.gz"


def metadata_url(category: str) -> str:
    category = _validate_category(category)
    return f"{AMAZON2023_BASE_URL}/raw/meta_categories/meta_{category}.jsonl.gz"


def _download_file(url: str, destination: Path, *, timeout: int = 120) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "regenrec/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def download_amazon2023(
    data_dir: str | Path,
    *,
    category: str,
    split_strategy: str = "last_out",
    include_metadata: bool = True,
    timeout: int = 120,
) -> dict[str, Path]:
    root = Path(data_dir)
    paths: dict[str, Path] = {}
    for split in DATA_SPLITS:
        destination = root / "5core" / split_strategy / f"{category}.{split}.csv.gz"
        print(f"download {benchmark_url(category, split_strategy, split)} -> {destination}")
        _download_file(
            benchmark_url(category, split_strategy, split),
            destination,
            timeout=timeout,
        )
        paths[split] = destination
    if include_metadata:
        destination = root / "raw" / f"meta_{category}.jsonl.gz"
        print(f"download {metadata_url(category)} -> {destination}")
        _download_file(metadata_url(category), destination, timeout=timeout)
        paths["metadata"] = destination
    return paths


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def iter_benchmark_rows(
    path: str | Path, *, max_rows: int | None = None
) -> Iterator[dict[str, str]]:
    input_path = Path(path)
    with _open_text(input_path) as handle:
        reader = csv.DictReader(handle)
        required = {"user_id", "parent_asin", "rating", "timestamp", "history"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row_index, row in enumerate(reader):
            if max_rows is not None and row_index >= max_rows:
                return
            yield row


def iter_amazon_examples(
    rows: Iterable[Mapping[str, object]],
    *,
    max_history: int = 50,
    num_time_buckets: int = 32,
    stats: AmazonPreparationStats | None = None,
    item_ids: set[str] | None = None,
) -> Iterator[SequenceExample]:
    """Convert the official 5-core *_w_his rows to canonical examples.

    The official history column contains item IDs but not historical timestamps or
    ratings. V1 therefore uses one review behavior and reverse-position recency
    buckets. Raw ratings are reserved for the V2 intensity adapter.
    """
    if max_history <= 0:
        raise ValueError("max_history must be positive")
    if num_time_buckets <= 0:
        raise ValueError("num_time_buckets must be positive")

    stats = stats if stats is not None else AmazonPreparationStats()
    for row in rows:
        stats.rows_seen += 1
        target = str(row.get("parent_asin") or "").strip()
        raw_history = row.get("history")
        history_ids = str(raw_history or "").split()
        if not target:
            stats.rows_without_target += 1
            continue
        if not history_ids:
            stats.rows_without_history += 1
            continue

        history_ids = history_ids[-max_history:]
        history_length = len(history_ids)
        history = tuple(
            Interaction(
                item_id=item_id,
                behavior=AMAZON2023_BEHAVIOR["review"],
                time_gap_bucket=min(
                    num_time_buckets - 1,
                    history_length - history_index - 1,
                ),
            )
            for history_index, item_id in enumerate(history_ids)
        )
        if item_ids is not None:
            item_ids.update(history_ids)
            item_ids.add(target)
        stats.examples_emitted += 1
        yield SequenceExample(
            user_id=str(row["user_id"]),
            history=history,
            target_item_id=target,
        )


def rating_behavior(rating: float) -> int:
    if not 1.0 <= rating <= 5.0:
        raise ValueError(f"Amazon rating must be in [1, 5], got {rating}")
    if rating <= 2.0:
        return AMAZON2023_RATING_BEHAVIOR["negative_review_1_2"]
    if rating < 4.0:
        return AMAZON2023_RATING_BEHAVIOR["neutral_review_3"]
    return AMAZON2023_RATING_BEHAVIOR["positive_review_4_5"]


def elapsed_time_bucket(
    target_timestamp_ms: int,
    history_timestamp_ms: int,
    *,
    num_time_buckets: int,
) -> int:
    if num_time_buckets <= 0:
        raise ValueError("num_time_buckets must be positive")
    elapsed_ms = target_timestamp_ms - history_timestamp_ms
    if elapsed_ms < 0:
        raise ValueError("history event timestamp is later than target timestamp")
    elapsed_days = elapsed_ms / 86_400_000
    return min(num_time_buckets - 1, int(math.log2(1.0 + elapsed_days)))


def build_observed_event_lookup(
    paths: Iterable[str | Path],
) -> dict[tuple[str, str], AmazonObservedEvent]:
    """Index official rows without deriving or imputing any event attributes."""
    lookup: dict[tuple[str, str], AmazonObservedEvent] = {}
    for path in paths:
        for row in iter_benchmark_rows(path):
            key = (str(row["user_id"]), str(row["parent_asin"]))
            event = AmazonObservedEvent(
                rating=float(row["rating"]),
                timestamp_ms=int(row["timestamp"]),
            )
            if key in lookup:
                raise ValueError(f"duplicate Amazon user-item event: {key}")
            rating_behavior(event.rating)
            lookup[key] = event
    return lookup


def iter_amazon_intensity_examples(
    rows: Iterable[Mapping[str, object]],
    event_lookup: Mapping[tuple[str, str], AmazonObservedEvent],
    *,
    max_history: int = 50,
    num_time_buckets: int = 32,
    stats: AmazonIntensityPreparationStats | None = None,
    item_ids: set[str] | None = None,
) -> Iterator[SequenceExample]:
    """Build V2 examples from observed ratings and timestamps in official rows.

    Rating is normalized to ``rating / 5`` for continuous intensity. Its sign is
    represented separately by negative/neutral/positive review behavior IDs.
    Recency uses log2 elapsed-day buckets computed from observed timestamps.
    Missing metadata or temporal leakage is rejected rather than imputed.
    """
    if max_history <= 0:
        raise ValueError("max_history must be positive")
    stats = stats if stats is not None else AmazonIntensityPreparationStats()
    for row in rows:
        stats.rows_seen += 1
        user_id = str(row.get("user_id") or "").strip()
        target = str(row.get("parent_asin") or "").strip()
        history_ids = str(row.get("history") or "").split()
        if not target:
            stats.rows_without_target += 1
            continue
        if not history_ids:
            stats.rows_without_history += 1
            continue

        target_timestamp_ms = int(str(row["timestamp"]))
        interactions: list[Interaction] = []
        for item_id in history_ids[-max_history:]:
            event = event_lookup.get((user_id, item_id))
            if event is None:
                stats.missing_history_metadata += 1
                raise ValueError(f"missing observed metadata for user={user_id} item={item_id}")
            if event.timestamp_ms > target_timestamp_ms:
                stats.future_history_events += 1
                raise ValueError(f"future history event for user={user_id} item={item_id}")
            behavior = rating_behavior(event.rating)
            if behavior == AMAZON2023_RATING_BEHAVIOR["negative_review_1_2"]:
                stats.negative_events += 1
            elif behavior == AMAZON2023_RATING_BEHAVIOR["neutral_review_3"]:
                stats.neutral_events += 1
            else:
                stats.positive_events += 1
            stats.history_events += 1
            interactions.append(
                Interaction(
                    item_id=item_id,
                    behavior=behavior,
                    time_gap_bucket=elapsed_time_bucket(
                        target_timestamp_ms,
                        event.timestamp_ms,
                        num_time_buckets=num_time_buckets,
                    ),
                    behavior_intensity=event.rating / 5.0,
                )
            )
        if item_ids is not None:
            item_ids.update(history_ids[-max_history:])
            item_ids.add(target)
        stats.examples_emitted += 1
        yield SequenceExample(
            user_id=user_id,
            history=tuple(interactions),
            target_item_id=target,
        )


def _flatten_text(value: object) -> Iterator[str]:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text and text.lower() != "none":
            yield text
        return
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _flatten_text(nested)
        return
    if isinstance(value, Sequence):
        for nested in value:
            yield from _flatten_text(nested)


def compose_item_text(record: Mapping[str, object], *, max_chars: int = 4096) -> str:
    fields = (
        record.get("title"),
        record.get("store"),
        record.get("main_category"),
        record.get("categories"),
        record.get("features"),
        record.get("description"),
    )
    parts: list[str] = []
    seen: set[str] = set()
    for field in fields:
        for text in _flatten_text(field):
            if text not in seen:
                parts.append(text)
                seen.add(text)
    return " [SEP] ".join(parts)[:max_chars]


def write_item_texts(
    metadata_path: str | Path,
    item_ids: set[str],
    output_path: str | Path,
    *,
    max_chars: int = 4096,
) -> tuple[int, int]:
    remaining = set(item_ids)
    found = 0
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        _open_text(Path(metadata_path)) as source,
        destination.open("w", encoding="utf-8") as output,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid metadata JSON on line {line_number} of {metadata_path}"
                ) from exc
            item_id = str(record.get("parent_asin") or "")
            if item_id not in remaining:
                continue
            text = compose_item_text(record, max_chars=max_chars) or item_id
            output.write(
                json.dumps(
                    {"item_id": item_id, "text": text, "metadata_found": True},
                    ensure_ascii=False,
                )
                + "\n"
            )
            remaining.remove(item_id)
            found += 1
            if not remaining:
                break
        for item_id in sorted(remaining):
            output.write(
                json.dumps(
                    {"item_id": item_id, "text": item_id, "metadata_found": False},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return found, len(remaining)


def prepare_amazon2023(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    category: str,
    split_strategy: str = "last_out",
    max_rows_per_split: int | None = None,
    max_history: int = 50,
    num_time_buckets: int = 32,
    max_text_chars: int = 4096,
) -> dict[str, object]:
    category = _validate_category(category)
    if split_strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"unsupported split strategy: {split_strategy}")
    source_root = Path(data_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    item_ids: set[str] = set()
    split_statistics: dict[str, dict[str, int]] = {}
    source_files: dict[str, str] = {}
    for split in DATA_SPLITS:
        source_path = source_root / "5core" / split_strategy / f"{category}.{split}.csv.gz"
        if not source_path.is_file():
            raise FileNotFoundError(f"missing Amazon benchmark file: {source_path}")
        source_files[split] = str(source_path.resolve())
        stats = AmazonPreparationStats()
        examples = iter_amazon_examples(
            iter_benchmark_rows(source_path, max_rows=max_rows_per_split),
            max_history=max_history,
            num_time_buckets=num_time_buckets,
            stats=stats,
            item_ids=item_ids,
        )
        write_sequence_examples_jsonl(examples, destination / f"{split}.jsonl")
        split_statistics[split] = asdict(stats)

    metadata_path = source_root / "raw" / f"meta_{category}.jsonl.gz"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing Amazon metadata file: {metadata_path}")
    source_files["metadata"] = str(metadata_path.resolve())
    metadata_found, metadata_missing = write_item_texts(
        metadata_path,
        item_ids,
        destination / "items.jsonl",
        max_chars=max_text_chars,
    )

    manifest: dict[str, object] = {
        "format_version": 1,
        "source_dataset": AMAZON2023_DATASET_ID,
        "category": category,
        "filter": "official 5-core deduplicated benchmark",
        "split_strategy": split_strategy,
        "source_files": source_files,
        "artifacts": {
            "train": "train.jsonl",
            "valid": "valid.jsonl",
            "test": "test.jsonl",
            "items": "items.jsonl",
        },
        "behavior_mapping": AMAZON2023_BEHAVIOR,
        "time_gap_semantics": "reverse chronological rank clipped to num_time_buckets",
        "parameters": {
            "max_rows_per_split": max_rows_per_split,
            "max_history": max_history,
            "num_time_buckets": num_time_buckets,
            "max_text_chars": max_text_chars,
        },
        "items": {
            "total": len(item_ids),
            "metadata_found": metadata_found,
            "metadata_missing": metadata_missing,
        },
        "statistics": split_statistics,
    }
    with (destination / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def prepare_amazon2023_v2(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    category: str,
    split_strategy: str = "last_out",
    max_rows_per_split: int | None = None,
    max_history: int = 50,
    num_time_buckets: int = 32,
    max_text_chars: int = 4096,
) -> dict[str, object]:
    """Prepare leakage-checked rating intensity and elapsed-time V2 data."""
    category = _validate_category(category)
    if split_strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"unsupported split strategy: {split_strategy}")
    source_root = Path(data_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source_paths = {
        split: source_root / "5core" / split_strategy / f"{category}.{split}.csv.gz"
        for split in DATA_SPLITS
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"missing Amazon benchmark file: {path}")

    event_lookup = build_observed_event_lookup(source_paths.values())
    rating_histogram: dict[str, int] = {str(rating): 0 for rating in range(1, 6)}
    for event in event_lookup.values():
        rating_histogram[str(int(event.rating))] += 1

    item_ids: set[str] = set()
    split_statistics: dict[str, dict[str, int]] = {}
    for split, source_path in source_paths.items():
        stats = AmazonIntensityPreparationStats()
        examples = iter_amazon_intensity_examples(
            iter_benchmark_rows(source_path, max_rows=max_rows_per_split),
            event_lookup,
            max_history=max_history,
            num_time_buckets=num_time_buckets,
            stats=stats,
            item_ids=item_ids,
        )
        write_sequence_examples_jsonl(examples, destination / f"{split}.jsonl")
        split_statistics[split] = asdict(stats)

    metadata_path = source_root / "raw" / f"meta_{category}.jsonl.gz"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing Amazon metadata file: {metadata_path}")
    metadata_found, metadata_missing = write_item_texts(
        metadata_path,
        item_ids,
        destination / "items.jsonl",
        max_chars=max_text_chars,
    )
    manifest: dict[str, object] = {
        "format_version": 2,
        "source_dataset": AMAZON2023_DATASET_ID,
        "category": category,
        "filter": "official 5-core deduplicated benchmark",
        "split_strategy": split_strategy,
        "source_files": {
            **{split: str(path.resolve()) for split, path in source_paths.items()},
            "metadata": str(metadata_path.resolve()),
        },
        "artifacts": {
            "train": "train.jsonl",
            "valid": "valid.jsonl",
            "test": "test.jsonl",
            "items": "items.jsonl",
        },
        "behavior_mapping": AMAZON2023_RATING_BEHAVIOR,
        "behavior_intensity": {
            "source": "observed rating",
            "formula": "rating / 5.0",
            "range": [0.2, 1.0],
            "imputation": None,
        },
        "time_gap_semantics": "floor(log2(1 + elapsed_days)) clipped to num_time_buckets",
        "temporal_integrity": {
            "event_lookup_rows": len(event_lookup),
            "duplicate_user_item_events": 0,
            "missing_history_metadata": sum(
                split["missing_history_metadata"] for split in split_statistics.values()
            ),
            "future_history_events": sum(
                split["future_history_events"] for split in split_statistics.values()
            ),
        },
        "observed_rating_histogram": rating_histogram,
        "parameters": {
            "max_rows_per_split": max_rows_per_split,
            "max_history": max_history,
            "num_time_buckets": num_time_buckets,
            "max_text_chars": max_text_chars,
        },
        "items": {
            "total": len(item_ids),
            "metadata_found": metadata_found,
            "metadata_missing": metadata_missing,
        },
        "statistics": split_statistics,
    }
    with (destination / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
