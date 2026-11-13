from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from regenrec.data import iter_sequence_examples_jsonl
from regenrec.embeddings import load_embedding_bundle


SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export audited Amazon Reviews 2023 data to official UniSRec format"
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--embedding-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def private_user_token(user_id: str) -> str:
    return hashlib.sha256(user_id.encode()).hexdigest()[:24]


def export_unisrec_dataset(
    processed_dir: Path,
    embedding_path: Path,
    output_root: Path,
    dataset_name: str,
) -> dict[str, object]:
    audit_path = processed_dir / "data_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("synthetic_data_used") is not False:
        raise ValueError("UniSRec export requires a passed real-data audit")
    item_ids, embeddings, metadata = load_embedding_bundle(embedding_path)
    if metadata.get("backend") != "blair":
        raise ValueError("UniSRec export requires BLaIR item vectors")
    if embeddings.ndim != 2 or embeddings.size(0) != len(item_ids):
        raise ValueError("embedding bundle is misaligned")
    expected_items = int(audit["checks"]["actual_unique_items"])
    if len(item_ids) != expected_items:
        raise ValueError("embedding count does not match the real-data audit")

    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
    split_counts: dict[str, int] = {}
    source_hashes: dict[str, str] = {}
    for split in SPLITS:
        source_path = processed_dir / f"{split}.jsonl"
        destination = dataset_dir / f"{dataset_name}.{split}.inter"
        rows = 0
        with destination.open("w", encoding="utf-8") as handle:
            handle.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
            for example in iter_sequence_examples_jsonl(source_path):
                history = [str(item_to_index[event.item_id]) for event in example.history]
                target = str(item_to_index[example.target_item_id])
                handle.write(
                    f"{private_user_token(example.user_id)}\t{' '.join(history)}\t{target}\n"
                )
                rows += 1
        split_counts[split] = rows
        source_hashes[split] = sha256(source_path)

    feature_path = dataset_dir / f"{dataset_name}.feat1CLS"
    np.asarray(embeddings.cpu(), dtype=np.float32).tofile(feature_path)
    mapping_path = dataset_dir / f"{dataset_name}.item2index"
    with mapping_path.open("w", encoding="utf-8") as handle:
        for item_id, index in item_to_index.items():
            handle.write(f"{item_id}\t{index}\n")

    manifest = {
        "format": "RecBole benchmark atomic files for official RUCAIBox/UniSRec",
        "dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "category": audit["category"],
        "dataset_name": dataset_name,
        "synthetic_data_used": False,
        "split_strategy": audit["split_strategy"],
        "split_counts": split_counts,
        "items": len(item_ids),
        "embedding_dimension": embeddings.size(1),
        "embedding_metadata": metadata,
        "source_sha256": source_hashes,
        "audit_sha256": sha256(audit_path),
        "embedding_sha256": sha256(embedding_path),
        "user_identifier": "sha256(raw_user_id)[:24]",
        "test_labels_used_for_training": False,
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = export_unisrec_dataset(
        args.processed_dir,
        args.embedding_path,
        args.output_root,
        args.dataset_name,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
