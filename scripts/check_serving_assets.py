from __future__ import annotations

import argparse
import json

from regenrec.serving import ServingSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate online inference asset paths")
    parser.add_argument(
        "--without-long-tail",
        action="store_true",
        help="Do not require the BLaIR embedding bundle",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ServingSettings.from_env()
    settings.validate(require_long_tail=not args.without_long_tail)
    print(
        json.dumps(
            {
                "status": "passed",
                "model_files": ["catalog.json", "generator.pt"],
                "long_tail_assets_checked": not args.without_long_tail,
                "device": settings.device,
                "max_beam_size": settings.max_beam_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
