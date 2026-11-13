from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SASRec(nn.Module):
    """Causal self-attention sequential recommendation baseline."""

    def __init__(
        self,
        *,
        num_items: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
        max_history_length: int,
    ) -> None:
        super().__init__()
        if num_items <= 0:
            raise ValueError("num_items must be positive")
        self.num_items = num_items
        self.padding_idx = 0
        self.item_embedding = nn.Embedding(num_items + 1, model_dim, padding_idx=self.padding_idx)
        self.position_embedding = nn.Embedding(max_history_length, model_dim)
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(model_dim)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[self.padding_idx].zero_()

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def sequence_state(
        self, history_items: torch.Tensor, history_mask: torch.Tensor
    ) -> torch.Tensor:
        if history_items.shape != history_mask.shape:
            raise ValueError("history_items and history_mask must have the same shape")
        positions = torch.arange(history_items.size(1), device=history_items.device)
        values = self.item_embedding(history_items) + self.position_embedding(positions).unsqueeze(
            0
        )
        encoded = self.encoder(
            self.input_dropout(values),
            mask=self._causal_mask(history_items.size(1), history_items.device),
            src_key_padding_mask=~history_mask,
        )
        last_positions = history_mask.sum(dim=1).clamp_min(1) - 1
        batch_indices = torch.arange(history_items.size(0), device=history_items.device)
        return self.output_norm(encoded[batch_indices, last_positions])

    def forward(self, history_items: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        state = self.sequence_state(history_items, history_mask)
        return state @ self.item_embedding.weight[1:].transpose(0, 1)

    @staticmethod
    def loss(logits: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target_indices)

    @torch.no_grad()
    def recommend(
        self,
        history_items: torch.Tensor,
        history_mask: torch.Tensor,
        *,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 < k <= self.num_items:
            raise ValueError("k must be in [1, num_items]")
        self.eval()
        return self(history_items, history_mask).topk(k, dim=-1)
