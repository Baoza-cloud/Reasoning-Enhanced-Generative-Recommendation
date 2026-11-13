from __future__ import annotations

import torch


def semantic_id_diagnostics(
    codes: torch.Tensor,
    *,
    codebook_size: int,
) -> dict[str, object]:
    """Summarize RQ code usage and collisions before rank disambiguation."""
    if codes.ndim != 2 or codes.size(0) == 0:
        raise ValueError("codes must have shape [num_items, num_levels] and be non-empty")
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    if torch.any(codes < 0) or torch.any(codes >= codebook_size):
        raise ValueError("codes contain an index outside the configured codebook")

    codes = codes.to(dtype=torch.long, device="cpu")
    levels: list[dict[str, object]] = []
    for level in range(codes.size(1)):
        counts = torch.bincount(codes[:, level], minlength=codebook_size)
        probabilities = counts.float() / codes.size(0)
        active = counts > 0
        entropy = -(probabilities[active] * probabilities[active].log()).sum()
        perplexity = float(entropy.exp())
        most_frequent_count, most_frequent_code = counts.max(dim=0)
        levels.append(
            {
                "level": level,
                "used_codes": int(active.sum()),
                "dead_codes": int((~active).sum()),
                "usage_ratio": float(active.float().mean()),
                "perplexity": perplexity,
                "effective_code_ratio": perplexity / codebook_size,
                "most_frequent_code": int(most_frequent_code),
                "most_frequent_count": int(most_frequent_count),
                "top_code_share": float(most_frequent_count / codes.size(0)),
                "counts": counts.tolist(),
            }
        )

    _, group_sizes = torch.unique(codes, dim=0, return_counts=True)
    collided = group_sizes > 1
    num_items = codes.size(0)
    unique_tuples = group_sizes.numel()
    collided_items = int(group_sizes[collided].sum())
    return {
        "num_items": num_items,
        "num_levels": codes.size(1),
        "codebook_size": codebook_size,
        "levels": levels,
        "code_tuples": {
            "unique": unique_tuples,
            "unique_ratio": unique_tuples / num_items,
            "duplicate_assignments": num_items - unique_tuples,
            "collision_groups": int(collided.sum()),
            "collided_items": collided_items,
            "collision_item_ratio": collided_items / num_items,
            "max_collision_size": int(group_sizes.max()),
            "mean_items_per_code": float(group_sizes.float().mean()),
        },
    }
