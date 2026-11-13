from __future__ import annotations

import random

import torch

from regenrec.data import Interaction, SequenceExample


def make_synthetic_catalog(
    *, num_items: int, embedding_dim: int, seed: int
) -> tuple[list[str], torch.Tensor, list[int]]:
    """Create clustered product embeddings with category signal."""
    generator = torch.Generator().manual_seed(seed)
    num_categories = max(4, min(8, num_items // 4))
    category_centers = torch.randn(num_categories, embedding_dim, generator=generator)
    categories = [item_index % num_categories for item_index in range(num_items)]
    item_embeddings = torch.stack(
        [
            category_centers[category]
            + 0.15 * torch.randn(embedding_dim, generator=generator)
            for category in categories
        ]
    )
    item_ids = [f"item_{item_index:04d}" for item_index in range(num_items)]
    return item_ids, item_embeddings, categories


def make_synthetic_sequences(
    *,
    item_ids: list[str],
    categories: list[int],
    num_users: int,
    min_history: int,
    max_history: int,
    num_behaviors: int,
    num_time_buckets: int,
    seed: int,
) -> list[SequenceExample]:
    """Generate preference-consistent sequences for pipeline verification."""
    rng = random.Random(seed)
    by_category: dict[int, list[str]] = {}
    for item_id, category in zip(item_ids, categories, strict=True):
        by_category.setdefault(category, []).append(item_id)

    category_ids = sorted(by_category)
    examples: list[SequenceExample] = []
    for user_index in range(num_users):
        preferred = category_ids[user_index % len(category_ids)]
        history_size = rng.randint(min_history, max_history)
        history: list[Interaction] = []
        for position in range(history_size):
            use_preference = rng.random() < 0.8
            category = preferred if use_preference else rng.choice(category_ids)
            behavior = min(num_behaviors - 1, 1 + int(use_preference) + position % 2)
            history.append(
                Interaction(
                    item_id=rng.choice(by_category[category]),
                    behavior=behavior,
                    time_gap_bucket=min(num_time_buckets - 1, history_size - position - 1),
                )
            )
        target = rng.choice(by_category[preferred])
        examples.append(
            SequenceExample(
                user_id=f"user_{user_index:04d}",
                history=tuple(history),
                target_item_id=target,
            )
        )
    return examples
