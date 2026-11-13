from __future__ import annotations

import argparse
import json
from pathlib import Path

from regenrec.datasets.criteo_uplift import (
    CRITEO_UPLIFT_FILENAME,
    audit_criteo_uplift,
    download_criteo_uplift,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and audit the real Criteo uplift RCT")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/criteo_uplift/raw") / CRITEO_UPLIFT_FILENAME,
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/criteo_uplift/criteo_uplift_audit.json"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-timeout", type=int, default=180)
    parser.add_argument("--max-rows", type=int, default=0, help="0 audits every row")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download:
        download_criteo_uplift(args.data_path, timeout=args.download_timeout)
    if not args.data_path.is_file():
        raise FileNotFoundError(f"missing Criteo uplift file: {args.data_path}; pass --download")
    audit = audit_criteo_uplift(
        args.data_path,
        max_rows=None if args.max_rows == 0 else args.max_rows,
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
