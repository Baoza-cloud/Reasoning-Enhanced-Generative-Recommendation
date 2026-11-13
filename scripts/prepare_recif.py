from __future__ import annotations

import argparse
import json
from pathlib import Path

from regenrec.datasets.recif import (
    MAIN_FILE_NAMES,
    PRODUCT_SID_FILE_NAMES,
    RECIF_DATASET_ID,
    prepare_recif_product,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OpenOneRec RecIF product data")
    parser.add_argument("--data-dir", type=Path, default=Path("data/recif/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/recif/processed"))
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the gated dataset from Hugging Face before conversion",
    )
    parser.add_argument("--max-rows", type=int, default=100_000, help="Use 0 for all rows")
    parser.add_argument("--max-history", type=int, default=100)
    parser.add_argument("--num-time-buckets", type=int, default=32)
    parser.add_argument("--max-targets-per-user", type=int, default=1, help="Use 0 for all")
    parser.add_argument(
        "--split",
        dest="splits",
        type=int,
        action="append",
        help="Keep a numeric RecIF split; repeat the option for multiple splits",
    )
    parser.add_argument("--codebook-size", type=int)
    return parser.parse_args()


def download_recif(data_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Downloading RecIF requires huggingface_hub. Install it with: pip install -e '.[data]'"
        ) from exc

    data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=RECIF_DATASET_ID,
        repo_type="dataset",
        local_dir=data_dir,
        allow_patterns=[*MAIN_FILE_NAMES, *PRODUCT_SID_FILE_NAMES, "README.md"],
    )


def main() -> None:
    args = parse_args()
    if args.download:
        download_recif(args.data_dir)
    manifest = prepare_recif_product(
        args.data_dir,
        args.output_dir,
        max_rows=None if args.max_rows == 0 else args.max_rows,
        max_history=args.max_history,
        num_time_buckets=args.num_time_buckets,
        max_targets_per_user=(
            None if args.max_targets_per_user == 0 else args.max_targets_per_user
        ),
        allowed_splits=set(args.splits) if args.splits else None,
        codebook_size=args.codebook_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
