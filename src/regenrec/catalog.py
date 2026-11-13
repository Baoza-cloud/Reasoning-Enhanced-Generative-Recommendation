from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch


class _TrieNode:
    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}
        self.item_id: str | None = None


@dataclass(frozen=True)
class CatalogSpecialTokens:
    pad: int
    bos: int
    eos: int


class SemanticIdCatalog:
    """Maps RQ codes to collision-free token sequences and constrains decoding.

    Each RQ level owns a disjoint token range. A final collision-rank token is
    appended to every item, which makes the sequence length fixed while keeping
    duplicate RQ codes resolvable.
    """

    def __init__(
        self,
        item_ids: Sequence[str],
        semantic_tokens: torch.Tensor,
        *,
        codebook_size: int,
        num_levels: int,
        max_collision_size: int,
    ) -> None:
        if semantic_tokens.ndim != 2:
            raise ValueError("semantic_tokens must have shape [num_items, sequence_length]")
        if len(item_ids) != semantic_tokens.size(0):
            raise ValueError("item_ids and semantic_tokens must contain the same number of items")

        self.item_ids = [str(item_id) for item_id in item_ids]
        self.semantic_tokens = semantic_tokens.to(dtype=torch.long, device="cpu")
        self.codebook_size = int(codebook_size)
        self.num_levels = int(num_levels)
        self.max_collision_size = int(max_collision_size)
        base_vocab_size = self.num_levels * self.codebook_size + self.max_collision_size
        self.special_tokens = CatalogSpecialTokens(
            pad=base_vocab_size,
            bos=base_vocab_size + 1,
            eos=base_vocab_size + 2,
        )
        self.vocab_size = base_vocab_size + 3
        self.item_to_index = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        if len(self.item_to_index) != len(self.item_ids):
            raise ValueError("item_ids must be unique")

        self._root = _TrieNode()
        self._sequence_to_item: dict[tuple[int, ...], str] = {}
        for item_id, sequence in zip(self.item_ids, self.semantic_tokens.tolist(), strict=True):
            key = tuple(sequence)
            if key in self._sequence_to_item:
                raise ValueError("Collision disambiguation produced duplicate token sequences")
            self._sequence_to_item[key] = item_id
            self._insert(sequence, item_id)

    @classmethod
    def from_codes(
        cls,
        item_ids: Sequence[str],
        codes: torch.Tensor,
        *,
        codebook_size: int,
    ) -> "SemanticIdCatalog":
        if codes.ndim != 2:
            raise ValueError("codes must have shape [num_items, num_levels]")
        if len(item_ids) != codes.size(0):
            raise ValueError("item_ids and codes must contain the same number of items")
        if torch.any(codes < 0) or torch.any(codes >= codebook_size):
            raise ValueError("codes contain an index outside the configured codebook")

        num_levels = codes.size(1)
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for row, code in enumerate(codes.tolist()):
            groups[tuple(code)].append(row)
        max_collision_size = max((len(rows) for rows in groups.values()), default=1)

        semantic_tokens = torch.empty(
            (codes.size(0), num_levels + 1), dtype=torch.long
        )
        level_offsets = torch.arange(num_levels, dtype=torch.long) * codebook_size
        semantic_tokens[:, :num_levels] = codes.to(dtype=torch.long, device="cpu") + level_offsets
        collision_base = num_levels * codebook_size

        for rows in groups.values():
            rows = sorted(rows, key=lambda row: str(item_ids[row]))
            for collision_rank, row in enumerate(rows):
                semantic_tokens[row, -1] = collision_base + collision_rank

        return cls(
            item_ids=item_ids,
            semantic_tokens=semantic_tokens,
            codebook_size=codebook_size,
            num_levels=num_levels,
            max_collision_size=max_collision_size,
        )

    def _insert(self, sequence: Iterable[int], item_id: str) -> None:
        node = self._root
        for token in [*sequence, self.special_tokens.eos]:
            node = node.children.setdefault(int(token), _TrieNode())
        node.item_id = item_id

    @property
    def sequence_length(self) -> int:
        return self.semantic_tokens.size(1)

    def tokens_for_item(self, item_id: str) -> torch.Tensor:
        return self.semantic_tokens[self.item_to_index[str(item_id)]].clone()

    def allowed_next(self, prefix: Sequence[int]) -> list[int]:
        """Return legal next tokens after BOS and the supplied generated prefix."""
        normalized = list(prefix)
        if normalized and normalized[0] == self.special_tokens.bos:
            normalized = normalized[1:]
        node = self._root
        for token in normalized:
            child = node.children.get(int(token))
            if child is None:
                return []
            node = child
        return sorted(node.children)

    def resolve(self, sequence: Sequence[int]) -> str | None:
        normalized = list(sequence)
        if normalized and normalized[0] == self.special_tokens.bos:
            normalized = normalized[1:]
        if normalized and normalized[-1] == self.special_tokens.eos:
            normalized = normalized[:-1]
        return self._sequence_to_item.get(tuple(int(token) for token in normalized))

    def to_dict(self) -> dict[str, object]:
        return {
            "item_ids": self.item_ids,
            "semantic_tokens": self.semantic_tokens.tolist(),
            "codebook_size": self.codebook_size,
            "num_levels": self.num_levels,
            "max_collision_size": self.max_collision_size,
        }

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SemanticIdCatalog":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            item_ids=payload["item_ids"],
            semantic_tokens=torch.tensor(payload["semantic_tokens"], dtype=torch.long),
            codebook_size=payload["codebook_size"],
            num_levels=payload["num_levels"],
            max_collision_size=payload["max_collision_size"],
        )
