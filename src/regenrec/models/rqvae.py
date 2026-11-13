from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RQVAEOutput:
    reconstruction: torch.Tensor
    codes: torch.Tensor
    quantized: torch.Tensor
    reconstruction_loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    loss: torch.Tensor


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_levels: int,
        codebook_size: int,
        code_dim: int,
        *,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        dead_code_threshold: float = 2.0,
    ) -> None:
        super().__init__()
        if min(num_levels, codebook_size, code_dim) <= 0:
            raise ValueError("quantizer dimensions must be positive")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if ema_epsilon <= 0:
            raise ValueError("ema_epsilon must be positive")
        if dead_code_threshold < 0:
            raise ValueError("dead_code_threshold cannot be negative")
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        self.dead_code_threshold = dead_code_threshold
        self.codebooks = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(codebook_size, code_dim),
                    requires_grad=not use_ema,
                )
                for _ in range(num_levels)
            ]
        )
        for level in range(num_levels):
            self.register_buffer(f"ema_cluster_size_{level}", torch.zeros(codebook_size))
            self.register_buffer(f"ema_embed_avg_{level}", torch.zeros(codebook_size, code_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for codebook in self.codebooks:
            nn.init.uniform_(codebook, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    @staticmethod
    def _nearest_codes(inputs: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        distances = (
            inputs.pow(2).sum(dim=-1, keepdim=True)
            - 2 * inputs @ codebook.t()
            + codebook.pow(2).sum(dim=-1).unsqueeze(0)
        )
        return distances.argmin(dim=-1)

    @torch.no_grad()
    def _kmeans(
        self,
        inputs: torch.Tensor,
        *,
        num_iters: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.size(0) < self.codebook_size:
            raise ValueError("K-means needs at least codebook_size input vectors")
        initial_indices = torch.randperm(inputs.size(0), generator=generator, device=inputs.device)[
            : self.codebook_size
        ]
        centroids = inputs[initial_indices].clone()
        counts = torch.zeros(self.codebook_size, dtype=inputs.dtype, device=inputs.device)

        for _ in range(num_iters):
            assignments = self._nearest_codes(inputs, centroids)
            counts = torch.bincount(assignments, minlength=self.codebook_size).to(inputs.dtype)
            sums = torch.zeros_like(centroids)
            sums.index_add_(0, assignments, inputs)
            nonempty = counts > 0
            updated = sums / counts.clamp_min(1.0).unsqueeze(1)
            if torch.any(~nonempty):
                nearest = (inputs - centroids[assignments]).pow(2).sum(dim=1)
                replacements = nearest.topk(int((~nonempty).sum())).indices
                updated[~nonempty] = inputs[replacements]
                counts[~nonempty] = 1.0
            if torch.allclose(updated, centroids, rtol=1e-4, atol=1e-5):
                centroids = updated
                break
            centroids = updated
        assignments = self._nearest_codes(inputs, centroids)
        counts = torch.bincount(assignments, minlength=self.codebook_size).to(inputs.dtype)
        return centroids, counts

    @torch.no_grad()
    def initialize_from_data(
        self,
        latents: torch.Tensor,
        *,
        num_iters: int = 20,
        seed: int = 2026,
        ema_count_scale: float = 1.0,
    ) -> list[dict[str, float]]:
        """Initialize every residual level with sequential K-means."""
        if latents.ndim != 2 or latents.size(1) != self.code_dim:
            raise ValueError("latents must have shape [num_samples, code_dim]")
        if num_iters <= 0:
            raise ValueError("num_iters must be positive")
        if ema_count_scale <= 0:
            raise ValueError("ema_count_scale must be positive")
        generator = torch.Generator(device=latents.device).manual_seed(seed)
        residual = latents.detach().to(self.codebooks[0].device)
        initialization: list[dict[str, float]] = []
        for level, codebook in enumerate(self.codebooks):
            centroids, counts = self._kmeans(
                residual,
                num_iters=num_iters,
                generator=generator,
            )
            codebook.copy_(centroids)
            cluster_size = getattr(self, f"ema_cluster_size_{level}")
            embed_avg = getattr(self, f"ema_embed_avg_{level}")
            ema_counts = counts * ema_count_scale
            cluster_size.copy_(ema_counts)
            embed_avg.copy_(centroids * ema_counts.unsqueeze(1))
            assignments = self._nearest_codes(residual, centroids)
            residual = residual - F.embedding(assignments, centroids)
            initialization.append(
                {
                    "level": float(level),
                    "used_codes": float((counts > 0).sum()),
                    "min_cluster_size": float(counts.min()),
                    "max_cluster_size": float(counts.max()),
                    "residual_mse": float(residual.pow(2).mean()),
                }
            )
        return initialization

    @torch.no_grad()
    def _ema_update(
        self,
        *,
        level: int,
        residual: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        codebook = self.codebooks[level]
        cluster_size = getattr(self, f"ema_cluster_size_{level}")
        embed_avg = getattr(self, f"ema_embed_avg_{level}")
        batch_counts = torch.bincount(indices, minlength=self.codebook_size).to(residual.dtype)
        batch_sums = torch.zeros_like(codebook)
        batch_sums.index_add_(0, indices, residual)
        cluster_size.mul_(self.ema_decay).add_(batch_counts, alpha=1.0 - self.ema_decay)
        embed_avg.mul_(self.ema_decay).add_(batch_sums, alpha=1.0 - self.ema_decay)

        total = cluster_size.sum()
        smoothed = (
            (cluster_size + self.ema_epsilon)
            / (total + self.codebook_size * self.ema_epsilon)
            * total
        )
        codebook.copy_(embed_avg / smoothed.clamp_min(self.ema_epsilon).unsqueeze(1))

        if self.dead_code_threshold <= 0 or residual.size(0) == 0:
            return
        dead = cluster_size < self.dead_code_threshold
        if not torch.any(dead):
            return
        count = int(dead.sum())
        sample_indices = torch.randint(residual.size(0), (count,), device=residual.device)
        replacements = residual[sample_indices]
        codebook[dead] = replacements
        cluster_size[dead] = self.dead_code_threshold
        embed_avg[dead] = replacements * self.dead_code_threshold

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z
        quantized_sum = torch.zeros_like(z)
        code_indices: list[torch.Tensor] = []
        codebook_losses: list[torch.Tensor] = []
        commitment_losses: list[torch.Tensor] = []

        for level, codebook in enumerate(self.codebooks):
            indices = self._nearest_codes(residual, codebook)
            quantized = F.embedding(indices, codebook)
            codebook_losses.append((quantized - residual.detach()).pow(2).sum(dim=-1))
            commitment_losses.append((residual - quantized.detach()).pow(2).sum(dim=-1))
            quantized_sum = quantized_sum + quantized
            if self.training and self.use_ema:
                self._ema_update(level=level, residual=residual.detach(), indices=indices)
            residual = residual - quantized.detach()
            code_indices.append(indices)

        codes = torch.stack(code_indices, dim=-1)
        codebook_loss = torch.stack(codebook_losses, dim=0).sum(dim=0).mean()
        commitment_loss = torch.stack(commitment_losses, dim=0).sum(dim=0).mean()
        return quantized_sum, codes, codebook_loss, commitment_loss


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        code_dim: int,
        num_levels: int,
        codebook_size: int,
        commitment_weight: float = 0.25,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        dead_code_threshold: float = 2.0,
        normalize_reconstruction: bool = False,
    ) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight
        self.normalize_reconstruction = normalize_reconstruction
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, code_dim, bias=False),
        )
        self.quantizer = ResidualVectorQuantizer(
            num_levels,
            codebook_size,
            code_dim,
            use_ema=use_ema,
            ema_decay=ema_decay,
            ema_epsilon=ema_epsilon,
            dead_code_threshold=dead_code_threshold,
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim, bias=False),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    @torch.no_grad()
    def initialize_codebooks(
        self,
        inputs: torch.Tensor,
        *,
        sample_size: int = 20000,
        batch_size: int = 2048,
        num_iters: int = 20,
        seed: int = 2026,
    ) -> list[dict[str, float]]:
        """Encode a reproducible sample and K-means initialize residual codebooks."""
        if inputs.ndim != 2 or inputs.size(0) == 0:
            raise ValueError("inputs must have shape [num_items, input_dim] and be non-empty")
        if min(sample_size, batch_size) <= 0:
            raise ValueError("sample_size and batch_size must be positive")
        cpu_generator = torch.Generator().manual_seed(seed)
        sample_count = min(sample_size, inputs.size(0))
        sample_indices = torch.randperm(inputs.size(0), generator=cpu_generator)[:sample_count]
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        encoded: list[torch.Tensor] = []
        for start in range(0, sample_count, batch_size):
            batch_indices = sample_indices[start : start + batch_size]
            encoded.append(self.encoder(inputs[batch_indices].to(device)))
        latents = torch.cat(encoded, dim=0)
        statistics = self.quantizer.initialize_from_data(
            latents,
            num_iters=num_iters,
            seed=seed,
            ema_count_scale=batch_size / sample_count,
        )
        self.train(was_training)
        return statistics

    @torch.no_grad()
    def encode_codes(self, inputs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(inputs)
        _, codes, _, _ = self.quantizer(z)
        return codes

    def forward(self, inputs: torch.Tensor) -> RQVAEOutput:
        z = self.encoder(inputs)
        quantized, codes, codebook_loss, commitment_loss = self.quantizer(z)
        straight_through = z + (quantized - z).detach()
        reconstruction = self.decoder(straight_through)
        if self.normalize_reconstruction:
            reconstruction = F.normalize(reconstruction, p=2, dim=-1)

        reconstruction_loss = (reconstruction - inputs).pow(2).sum(dim=-1).mean()
        if self.quantizer.use_ema:
            codebook_loss = torch.zeros((), dtype=z.dtype, device=z.device)
        loss = reconstruction_loss + codebook_loss + self.commitment_weight * commitment_loss
        return RQVAEOutput(
            reconstruction=reconstruction,
            codes=codes,
            quantized=quantized,
            reconstruction_loss=reconstruction_loss,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            loss=loss,
        )

    @torch.no_grad()
    def codebook_statistics(self, inputs: torch.Tensor) -> list[dict[str, float]]:
        codes = self.encode_codes(inputs)
        statistics: list[dict[str, float]] = []
        for level in range(codes.size(1)):
            counts = torch.bincount(codes[:, level], minlength=self.quantizer.codebook_size).float()
            probabilities = counts / counts.sum().clamp_min(1.0)
            nonzero = probabilities > 0
            entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
            statistics.append(
                {
                    "level": float(level),
                    "usage_ratio": float(nonzero.float().mean()),
                    "perplexity": float(entropy.exp()),
                }
            )
        return statistics
