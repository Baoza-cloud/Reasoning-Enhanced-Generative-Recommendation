from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from regenrec.catalog import SemanticIdCatalog
from regenrec.data import Interaction, SequenceExample, write_sequence_examples_jsonl

RECIF_DATASET_ID = "OpenOneRec/OpenOneRec-RecIF"
RECIF_PRODUCT_BEHAVIOR = {"padding": 0, "goods_browse": 1}

MAIN_FILE_NAMES = ("onerec_bench_release.parquet",)
PRODUCT_SID_FILE_NAMES = ("product_pid2sid.parquet", "goods_pid2sid.parquet")


@dataclass
class RecIFPreparationStats:
    rows_seen: int = 0
    rows_filtered_by_split: int = 0
    rows_without_history: int = 0
    rows_without_target: int = 0
    unknown_history_items: int = 0
    unknown_target_items: int = 0
    examples_emitted: int = 0


def _normalise_item_ids(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item_id) for item_id in value if item_id is not None]
    raise ValueError(f"expected a sequence of item IDs, got {type(value).__name__}")


def build_catalog_from_sid_rows(
    rows: Iterable[Mapping[str, object]], *, codebook_size: int | None = None
) -> SemanticIdCatalog:
    item_ids: list[str] = []
    sid_rows: list[list[int]] = []
    seen: set[str] = set()
    sid_length: int | None = None

    for row in rows:
        item_id = str(row["pid"])
        if item_id in seen:
            continue
        raw_sid = row["sid"]
        if isinstance(raw_sid, torch.Tensor):
            raw_sid = raw_sid.tolist()
        if not isinstance(raw_sid, Sequence) or isinstance(raw_sid, (str, bytes)):
            raise ValueError(f"sid for product {item_id} must be a sequence")
        sid = [int(code) for code in raw_sid]
        if not sid or any(code < 0 for code in sid):
            raise ValueError(f"sid for product {item_id} must contain non-negative integers")
        if sid_length is None:
            sid_length = len(sid)
        elif len(sid) != sid_length:
            raise ValueError("all RecIF semantic IDs must have the same length")
        seen.add(item_id)
        item_ids.append(item_id)
        sid_rows.append(sid)

    if not item_ids:
        raise ValueError("the RecIF Semantic ID file is empty")
    codes = torch.tensor(sid_rows, dtype=torch.long)
    inferred_codebook_size = int(codes.max().item()) + 1
    if codebook_size is None:
        codebook_size = inferred_codebook_size
    if codebook_size < inferred_codebook_size:
        raise ValueError(
            f"codebook_size={codebook_size} is smaller than the largest SID code + 1 "
            f"({inferred_codebook_size})"
        )
    return SemanticIdCatalog.from_codes(item_ids, codes, codebook_size=codebook_size)


def iter_product_examples(
    rows: Iterable[Mapping[str, object]],
    catalog: SemanticIdCatalog,
    *,
    max_history: int = 100,
    num_time_buckets: int = 32,
    max_targets_per_user: int | None = 1,
    allowed_splits: set[int] | None = None,
    stats: RecIFPreparationStats | None = None,
) -> Iterator[SequenceExample]:
    """Convert RecIF product rows to the project's canonical sequence format.

    RecIF product histories are chronological but do not expose timestamps. The
    time-gap bucket therefore records recency rank: 0 is the most recent event,
    1 is the previous event, and older events are clipped to the final bucket.
    """
    if max_history <= 0:
        raise ValueError("max_history must be positive")
    if num_time_buckets <= 0:
        raise ValueError("num_time_buckets must be positive")
    if max_targets_per_user is not None and max_targets_per_user <= 0:
        raise ValueError("max_targets_per_user must be positive or None")

    stats = stats if stats is not None else RecIFPreparationStats()
    known_items = catalog.item_to_index
    for row in rows:
        stats.rows_seen += 1
        if allowed_splits is not None:
            raw_split = row.get("split")
            if raw_split is None or int(raw_split) not in allowed_splits:
                stats.rows_filtered_by_split += 1
                continue

        history_ids = _normalise_item_ids(row.get("hist_goods_pid"))
        target_ids = _normalise_item_ids(row.get("target_goods_pid"))
        if not history_ids:
            stats.rows_without_history += 1
            continue
        if not target_ids:
            stats.rows_without_target += 1
            continue

        known_history: list[str] = []
        for item_id in history_ids:
            if item_id in known_items:
                known_history.append(item_id)
            else:
                stats.unknown_history_items += 1
        known_history = known_history[-max_history:]
        if not known_history:
            stats.rows_without_history += 1
            continue

        history_length = len(known_history)
        history = tuple(
            Interaction(
                item_id=item_id,
                behavior=RECIF_PRODUCT_BEHAVIOR["goods_browse"],
                time_gap_bucket=min(
                    num_time_buckets - 1,
                    history_length - history_index - 1,
                ),
            )
            for history_index, item_id in enumerate(known_history)
        )

        emitted_for_user = 0
        for target_id in target_ids:
            if target_id not in known_items:
                stats.unknown_target_items += 1
                continue
            yield SequenceExample(
                user_id=str(row["uid"]),
                history=history,
                target_item_id=target_id,
            )
            stats.examples_emitted += 1
            emitted_for_user += 1
            if max_targets_per_user is not None and emitted_for_user >= max_targets_per_user:
                break


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "RecIF Parquet support requires the data dependencies. "
            "Install them with: pip install -e '.[data]'"
        ) from exc
    return parquet, parquet.ParquetFile


def _find_file(data_dir: Path, candidates: Sequence[str]) -> Path:
    for candidate in candidates:
        direct = data_dir / candidate
        if direct.is_file():
            return direct
    for candidate in candidates:
        matches = sorted(data_dir.rglob(candidate))
        if matches:
            return matches[0]
    names = ", ".join(candidates)
    raise FileNotFoundError(f"none of [{names}] was found below {data_dir}")


def iter_parquet_rows(
    path: str | Path,
    *,
    columns: Sequence[str],
    batch_size: int = 4096,
    max_rows: int | None = None,
) -> Iterator[dict[str, object]]:
    _, parquet_file_type = _require_pyarrow()
    parquet_file = parquet_file_type(Path(path))
    available = set(parquet_file.schema_arrow.names)
    missing = set(columns) - available
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    emitted = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=list(columns)):
        for row in batch.to_pylist():
            yield row
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
                return


def load_product_catalog(
    sid_path: str | Path, *, codebook_size: int | None = None
) -> SemanticIdCatalog:
    rows = iter_parquet_rows(sid_path, columns=("pid", "sid"))
    return build_catalog_from_sid_rows(rows, codebook_size=codebook_size)


def prepare_recif_product(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    max_rows: int | None = 100_000,
    max_history: int = 100,
    num_time_buckets: int = 32,
    max_targets_per_user: int | None = 1,
    allowed_splits: set[int] | None = None,
    codebook_size: int | None = None,
) -> dict[str, object]:
    source_dir = Path(data_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    main_path = _find_file(source_dir, MAIN_FILE_NAMES)
    sid_path = _find_file(source_dir, PRODUCT_SID_FILE_NAMES)

    catalog = load_product_catalog(sid_path, codebook_size=codebook_size)
    stats = RecIFPreparationStats()
    columns = ["uid", "split", "hist_goods_pid", "target_goods_pid"]
    rows = iter_parquet_rows(main_path, columns=columns, max_rows=max_rows)
    examples = iter_product_examples(
        rows,
        catalog,
        max_history=max_history,
        num_time_buckets=num_time_buckets,
        max_targets_per_user=max_targets_per_user,
        allowed_splits=allowed_splits,
        stats=stats,
    )

    catalog_path = destination / "catalog.json"
    examples_path = destination / "examples.jsonl"
    catalog.save(catalog_path)
    written = write_sequence_examples_jsonl(examples, examples_path)
    if written == 0:
        raise ValueError("RecIF conversion produced no product recommendation examples")

    manifest: dict[str, object] = {
        "format_version": 1,
        "source_dataset": RECIF_DATASET_ID,
        "domain": "product",
        "source_files": {
            "interactions": str(main_path.resolve()),
            "semantic_ids": str(sid_path.resolve()),
        },
        "artifacts": {
            "catalog": catalog_path.name,
            "examples": examples_path.name,
        },
        "behavior_mapping": RECIF_PRODUCT_BEHAVIOR,
        "time_gap_semantics": "reverse chronological rank clipped to num_time_buckets",
        "parameters": {
            "max_rows": max_rows,
            "max_history": max_history,
            "num_time_buckets": num_time_buckets,
            "max_targets_per_user": max_targets_per_user,
            "allowed_splits": sorted(allowed_splits) if allowed_splits else None,
            "codebook_size": catalog.codebook_size,
        },
        "catalog": {
            "items": len(catalog.item_ids),
            "semantic_id_length": catalog.sequence_length,
            "vocab_size": catalog.vocab_size,
        },
        "statistics": asdict(stats),
    }
    with (destination / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
