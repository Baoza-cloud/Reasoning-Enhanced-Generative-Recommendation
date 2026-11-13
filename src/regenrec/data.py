from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Iterator
from collections.abc import Mapping
from typing import Sequence

import torch
from torch.utils.data import Dataset

from regenrec.catalog import SemanticIdCatalog


@dataclass(frozen=True)
class Interaction:
    item_id: str
    behavior: int
    time_gap_bucket: int
    behavior_intensity: float = 1.0


@dataclass(frozen=True)
class SequenceExample:
    user_id: str
    history: tuple[Interaction, ...]
    target_item_id: str


class RecommendationDataset(Dataset[SequenceExample]):
    def __init__(self, examples: Sequence[SequenceExample]) -> None:
        if not examples:
            raise ValueError("examples cannot be empty")
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SequenceExample:
        return self.examples[index]


class SequenceCollator:
    def __init__(
        self,
        catalog: SemanticIdCatalog,
        *,
        item_loss_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.catalog = catalog
        self.item_loss_weights = item_loss_weights

    def __call__(self, examples: Sequence[SequenceExample]) -> dict[str, torch.Tensor]:
        batch_size = len(examples)
        max_history = max(len(example.history) for example in examples)
        semantic_length = self.catalog.sequence_length
        pad = self.catalog.special_tokens.pad

        history_tokens = torch.full(
            (batch_size, max_history, semantic_length), pad, dtype=torch.long
        )
        behavior_types = torch.zeros((batch_size, max_history), dtype=torch.long)
        behavior_intensities = torch.zeros((batch_size, max_history), dtype=torch.float32)
        time_gap_buckets = torch.zeros((batch_size, max_history), dtype=torch.long)
        history_mask = torch.zeros((batch_size, max_history), dtype=torch.bool)

        target_input = torch.full((batch_size, semantic_length + 1), pad, dtype=torch.long)
        target_labels = torch.full_like(target_input, pad)
        sample_weights = torch.ones(batch_size, dtype=torch.float32)

        for batch_index, example in enumerate(examples):
            for history_index, event in enumerate(example.history):
                history_tokens[batch_index, history_index] = self.catalog.tokens_for_item(
                    event.item_id
                )
                behavior_types[batch_index, history_index] = event.behavior
                behavior_intensities[batch_index, history_index] = event.behavior_intensity
                time_gap_buckets[batch_index, history_index] = event.time_gap_bucket
                history_mask[batch_index, history_index] = True

            target_tokens = self.catalog.tokens_for_item(example.target_item_id)
            target_input[batch_index, 0] = self.catalog.special_tokens.bos
            target_input[batch_index, 1:] = target_tokens
            target_labels[batch_index, :-1] = target_tokens
            target_labels[batch_index, -1] = self.catalog.special_tokens.eos
            if self.item_loss_weights is not None:
                sample_weights[batch_index] = float(
                    self.item_loss_weights.get(example.target_item_id, 1.0)
                )

        return {
            "history_tokens": history_tokens,
            "behavior_types": behavior_types,
            "behavior_intensities": behavior_intensities,
            "time_gap_buckets": time_gap_buckets,
            "history_mask": history_mask,
            "target_input": target_input,
            "target_labels": target_labels,
            "sample_weights": sample_weights,
        }


def sequence_example_to_dict(example: SequenceExample) -> dict[str, object]:
    return {
        "user_id": example.user_id,
        "history": [
            {
                "item_id": event.item_id,
                "behavior": event.behavior,
                "behavior_intensity": event.behavior_intensity,
                "time_gap_bucket": event.time_gap_bucket,
            }
            for event in example.history
        ],
        "target_item_id": example.target_item_id,
    }


def sequence_example_from_dict(payload: dict[str, object]) -> SequenceExample:
    raw_history = payload.get("history")
    if not isinstance(raw_history, list):
        raise ValueError("history must be a list")
    history: list[Interaction] = []
    for raw_event in raw_history:
        if not isinstance(raw_event, dict):
            raise ValueError("each history event must be a mapping")
        history.append(
            Interaction(
                item_id=str(raw_event["item_id"]),
                behavior=int(raw_event["behavior"]),
                time_gap_bucket=int(raw_event["time_gap_bucket"]),
                behavior_intensity=float(raw_event.get("behavior_intensity", 1.0)),
            )
        )
    return SequenceExample(
        user_id=str(payload["user_id"]),
        history=tuple(history),
        target_item_id=str(payload["target_item_id"]),
    )


def write_sequence_examples_jsonl(examples: Iterable[SequenceExample], path: str | Path) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(sequence_example_to_dict(example), ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def iter_sequence_examples_jsonl(path: str | Path) -> Iterator[SequenceExample]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} of {path} must contain a JSON object")
            yield sequence_example_from_dict(payload)


def load_sequence_examples_jsonl(
    path: str | Path,
    *,
    limit: int | None = None,
    selection_strategy: str = "prefix",
    seed: int = 0,
) -> list[SequenceExample]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    if selection_strategy not in {"prefix", "reservoir"}:
        raise ValueError("selection_strategy must be 'prefix' or 'reservoir'")
    if limit is None or selection_strategy == "prefix":
        examples: list[SequenceExample] = []
        for example in iter_sequence_examples_jsonl(path):
            examples.append(example)
            if limit is not None and len(examples) >= limit:
                break
    else:
        rng = random.Random(seed)
        sampled: list[tuple[int, SequenceExample]] = []
        for index, example in enumerate(iter_sequence_examples_jsonl(path)):
            if index < limit:
                sampled.append((index, example))
                continue
            replacement = rng.randrange(index + 1)
            if replacement < limit:
                sampled[replacement] = (index, example)
        sampled.sort(key=lambda row: row[0])
        examples = [example for _, example in sampled]
    if not examples:
        raise ValueError(f"no sequence examples found in {path}")
    return examples
