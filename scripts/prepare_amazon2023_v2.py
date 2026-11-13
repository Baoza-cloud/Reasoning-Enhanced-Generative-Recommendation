from __future__ import annotations

import argparse
import json
from pathlib import Path

from regenrec.datasets.amazon2023 import (
    SPLIT_STRATEGIES,
    download_amazon2023,
    prepare_amazon2023_v2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Amazon Reviews 2023 V2 intensity data")
    parser.add_argument("--category", default="Musical_Instruments")
    parser.add_argument("--split-strategy", choices=sorted(SPLIT_STRATEGIES), default="last_out")
    parser.add_argument("--data-dir", type=Path, default=Path("data/amazon2023/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to data/amazon2023/processed/<category>/<split-strategy>_v2",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-timeout", type=int, default=120)
    parser.add_argument("--max-rows-per-split", type=int, default=0, help="0 means all rows")
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--num-time-buckets", type=int, default=32)
    parser.add_argument("--max-text-chars", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        Path("data/amazon2023/processed") / args.category / f"{args.split_strategy}_v2"
    )
    if args.download:
        download_amazon2023(
            args.data_dir,
            category=args.category,
            split_strategy=args.split_strategy,
            timeout=args.download_timeout,
        )
    manifest = prepare_amazon2023_v2(
        args.data_dir,
        output_dir,
        category=args.category,
        split_strategy=args.split_strategy,
        max_rows_per_split=(None if args.max_rows_per_split == 0 else args.max_rows_per_split),
        max_history=args.max_history,
        num_time_buckets=args.num_time_buckets,
        max_text_chars=args.max_text_chars,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
