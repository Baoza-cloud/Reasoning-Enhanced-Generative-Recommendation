from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


def load_item_texts(path: str | Path) -> tuple[list[str], list[str]]:
    item_ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            item_id = str(payload["item_id"])
            if item_id in seen:
                raise ValueError(f"duplicate item_id {item_id} on line {line_number}")
            seen.add(item_id)
            item_ids.append(item_id)
            texts.append(str(payload["text"]))
    if not item_ids:
        raise ValueError(f"no item texts found in {path}")
    return item_ids, texts


def hashing_text_embeddings(texts: list[str], *, dimension: int = 384) -> torch.Tensor:
    """Create deterministic CPU-only smoke embeddings without model downloads."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    embeddings = torch.zeros((len(texts), dimension), dtype=torch.float32)
    for row, text in enumerate(texts):
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            column = int.from_bytes(digest[:8], "little") % dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            embeddings[row, column] += sign
    return F.normalize(embeddings, p=2, dim=1)


@torch.inference_mode()
def blair_text_embeddings(
    texts: list[str],
    *,
    model_name: str = "hyp1231/blair-roberta-base",
    device: str = "cpu",
    batch_size: int = 64,
    max_length: int = 512,
) -> torch.Tensor:
    """Encode item text with BLaIR's documented CLS-pooling recipe."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for BLaIR embeddings. "
            "Install it with: pip install -e '.[embeddings]'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    batches: list[torch.Tensor] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
        encoded: dict[str, Any] = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        hidden = model(**encoded, return_dict=True).last_hidden_state
        batches.append(F.normalize(hidden[:, 0].float(), p=2, dim=1).cpu())
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == total_batches:
            completed = min(start + batch_size, len(texts))
            print(
                f"blair encoding batch={batch_index}/{total_batches} "
                f"items={completed}/{len(texts)}",
                flush=True,
            )
    if not batches:
        raise ValueError("texts cannot be empty")
    return torch.cat(batches, dim=0)


def save_embedding_bundle(
    path: str | Path,
    *,
    item_ids: list[str],
    embeddings: torch.Tensor,
    metadata: dict[str, object],
) -> None:
    if embeddings.ndim != 2 or embeddings.size(0) != len(item_ids):
        raise ValueError("embeddings must have shape [len(item_ids), embedding_dim]")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "item_ids": item_ids,
            "embeddings": embeddings.detach().cpu().to(torch.float32),
            "metadata": metadata,
        },
        destination,
    )


def load_embedding_bundle(
    path: str | Path,
) -> tuple[list[str], torch.Tensor, dict[str, object]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    item_ids = [str(item_id) for item_id in payload["item_ids"]]
    embeddings = payload["embeddings"].to(torch.float32)
    metadata = dict(payload.get("metadata", {}))
    if embeddings.ndim != 2 or embeddings.size(0) != len(item_ids):
        raise ValueError("invalid item embedding bundle")
    return item_ids, embeddings, metadata
