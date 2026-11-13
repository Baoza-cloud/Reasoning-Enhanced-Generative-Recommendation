from __future__ import annotations

import argparse
import json
from pathlib import Path

from regenrec.data import iter_sequence_examples_jsonl
from regenrec.serving import RecommendationEngine, ServingSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one real-sequence inference demo")
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument(
        "--long-tail-mode", choices=("off", "mild", "aggressive"), default="off"
    )
    parser.add_argument("--input", type=Path, help="Optional sequence JSONL input")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = RecommendationEngine.load(ServingSettings.from_env())
    input_path = args.input or Path(engine.config["data"]["test_path"])
    example = next(iter_sequence_examples_jsonl(input_path))
    result = engine.recommend(
        [example.history],
        beam_size=args.beam_size,
        long_tail_mode=args.long_tail_mode,
    )[0]
    payload = {
        "input": {
            "history": [
                {
                    "item_id": event.item_id,
                    "title": engine.item_metadata[event.item_id]["title"],
                    "behavior": event.behavior,
                    "behavior_intensity": event.behavior_intensity,
                    "time_gap_bucket": event.time_gap_bucket,
                }
                for event in example.history
            ],
            "target_hidden": True,
        },
        "preference_basis": result["preference_basis"],
        "beam_candidates": result["beam_candidates"],
        "recommendations": result["recommendations"],
        "latency_ms": result["latency_ms"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
