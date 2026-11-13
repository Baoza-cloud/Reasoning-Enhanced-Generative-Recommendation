from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from regenrec.catalog import SemanticIdCatalog


@dataclass
class PreferenceState:
    vector: torch.Tensor
    long_term: torch.Tensor
    short_term: torch.Tensor
    short_term_weight: torch.Tensor
    event_weights: torch.Tensor | None = None
    decay_rate: torch.Tensor | None = None
    history_retention: torch.Tensor | None = None
    forgetting_rate: torch.Tensor | None = None


class PreferenceReasoner(nn.Module):
    """Build a preference summary from the encoded interaction history.

    ``full`` is the proposed explicit reasoner: it separates long- and short-term
    preference, learns a dimension-wise gate, and transforms the fused state.
    ``mean_pool`` is the pre-registered coarse ablation: it uses the same history encoder
    and observed behavior/recency strengths, but removes the learned event
    attention, long/short gate, and output MLP from the path used by the decoder.
    Keeping the surrounding architecture fixed makes the comparison attributable
    to explicit preference reasoning rather than to a different sequence encoder.

    The remaining modes support both frozen-checkpoint mechanism interventions
    and controlled from-scratch retraining ablations:

    - ``long_term_only`` and ``short_term_only`` test whether either source alone
      is sufficient, bypassing the gate and fusion MLP.
    - ``fixed_0p5_gate`` retains event attention and the fusion MLP but replaces
      the learned gate with an equal mixture.
    - ``no_event_attention`` retains the learned gate and fusion MLP but removes
      only the learned event-attention multiplier.
    - ``no_fusion_mlp`` retains event attention and the learned gate but feeds the
      gated mixture directly to the decoder.
    """

    def __init__(
        self,
        model_dim: int,
        short_term_window: int,
        *,
        use_behavior_intensity: bool = False,
        reasoning_mode: str = "full",
    ) -> None:
        super().__init__()
        valid_modes = {
            "full",
            "mean_pool",
            "long_term_only",
            "short_term_only",
            "fixed_0p5_gate",
            "no_event_attention",
            "no_fusion_mlp",
        }
        if reasoning_mode not in valid_modes:
            raise ValueError(f"reasoning_mode must be one of {sorted(valid_modes)}")
        self.short_term_window = short_term_window
        self.use_behavior_intensity = use_behavior_intensity
        self.reasoning_mode = reasoning_mode
        self.gate = nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.Sigmoid())
        self.output = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        if use_behavior_intensity:
            self.intensity_attention = nn.Linear(model_dim, 1)

    def forward(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        event_strengths: torch.Tensor | None = None,
        history_retention: torch.Tensor | None = None,
    ) -> PreferenceState:
        if event_strengths is None:
            scalar_weights = mask.to(encoded.dtype)
        else:
            if not self.use_behavior_intensity:
                raise ValueError("event strengths require an intensity-aware reasoner")
            scalar_weights = mask.to(encoded.dtype) * event_strengths
            if self.reasoning_mode not in {"mean_pool", "no_event_attention"}:
                attention = torch.sigmoid(self.intensity_attention(encoded)).squeeze(-1)
                scalar_weights = scalar_weights * attention
        if history_retention is not None:
            if history_retention.shape != mask.shape:
                raise ValueError("history_retention must match history mask shape")
            scalar_weights = scalar_weights * history_retention.to(encoded.dtype)
        weights = scalar_weights.unsqueeze(-1)
        long_term = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        batch_size = encoded.size(0)
        recent_vectors: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            valid = encoded[batch_index][mask[batch_index]]
            if valid.numel() == 0:
                recent_vectors.append(torch.zeros_like(encoded[batch_index, 0]))
            else:
                recent = valid[-self.short_term_window :]
                if event_strengths is None:
                    recent_vectors.append(recent.mean(dim=0))
                else:
                    recent_weights = scalar_weights[batch_index][mask[batch_index]][
                        -self.short_term_window :
                    ].unsqueeze(-1)
                    recent_vectors.append(
                        (recent * recent_weights).sum(dim=0) / recent_weights.sum().clamp_min(1e-6)
                    )
        short_term = torch.stack(recent_vectors)

        if self.reasoning_mode == "mean_pool":
            short_term_weight = torch.zeros_like(long_term)
            vector = long_term
        elif self.reasoning_mode == "long_term_only":
            short_term_weight = torch.zeros_like(long_term)
            vector = long_term
        elif self.reasoning_mode == "short_term_only":
            short_term_weight = torch.ones_like(long_term)
            vector = short_term
        else:
            if self.reasoning_mode == "fixed_0p5_gate":
                short_term_weight = torch.full_like(long_term, 0.5)
            else:
                short_term_weight = self.gate(torch.cat([long_term, short_term], dim=-1))
            mixed = short_term_weight * short_term + (1.0 - short_term_weight) * long_term
            if self.reasoning_mode == "no_fusion_mlp":
                vector = mixed
            else:
                vector = self.output(torch.cat([mixed, long_term], dim=-1))
        return PreferenceState(
            vector=vector,
            long_term=long_term,
            short_term=short_term,
            short_term_weight=short_term_weight,
            event_weights=scalar_weights if event_strengths is not None else None,
        )


class GenerativeRecommender(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        pad_token_id: int,
        num_behaviors: int,
        num_time_buckets: int,
        model_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        feedforward_dim: int,
        dropout: float,
        short_term_window: int,
        max_history_length: int = 256,
        max_target_length: int = 16,
        use_behavior_intensity: bool = False,
        preference_reasoning_mode: str = "full",
        history_forgetting_mode: str = "none",
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.num_time_buckets = num_time_buckets
        self.use_behavior_intensity = use_behavior_intensity
        self.preference_reasoning_mode = preference_reasoning_mode
        valid_forgetting_modes = {"none", "learned_exponential"}
        if history_forgetting_mode not in valid_forgetting_modes:
            raise ValueError(
                "history_forgetting_mode must be one of "
                f"{sorted(valid_forgetting_modes)}"
            )
        self.history_forgetting_mode = history_forgetting_mode
        self.token_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_token_id)
        self.behavior_embedding = nn.Embedding(num_behaviors, model_dim)
        self.time_embedding = nn.Embedding(num_time_buckets, model_dim)
        self.history_position = nn.Embedding(max_history_length, model_dim)
        self.target_position = nn.Embedding(max_target_length, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False
        )
        self.reasoner = PreferenceReasoner(
            model_dim,
            short_term_window,
            use_behavior_intensity=use_behavior_intensity,
            reasoning_mode=preference_reasoning_mode,
        )
        if use_behavior_intensity:
            self.intensity_projection = nn.Sequential(
                nn.Linear(1, model_dim),
                nn.SiLU(),
                nn.Linear(model_dim, model_dim),
            )
            self.personalized_decay = nn.Linear(model_dim, 1)
        if history_forgetting_mode == "learned_exponential":
            self.history_forgetting = nn.Linear(model_dim, 1)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_projection = nn.Linear(model_dim, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def encode_history(
        self,
        history_tokens: torch.Tensor,
        behavior_types: torch.Tensor,
        time_gap_buckets: torch.Tensor,
        history_mask: torch.Tensor,
        behavior_intensities: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PreferenceState]:
        token_mask = history_tokens.ne(self.pad_token_id).unsqueeze(-1)
        token_values = self.token_embedding(history_tokens)
        item_embedding = (token_values * token_mask).sum(dim=2) / token_mask.sum(dim=2).clamp_min(1)
        positions = torch.arange(history_tokens.size(1), device=history_tokens.device)
        base_event_embedding = (
            item_embedding
            + self.behavior_embedding(behavior_types)
            + self.time_embedding(time_gap_buckets)
            + self.history_position(positions).unsqueeze(0)
        )
        event_strengths: torch.Tensor | None = None
        decay_rate: torch.Tensor | None = None
        forgetting_rate: torch.Tensor | None = None
        history_retention: torch.Tensor | None = None
        needs_user_context = (
            self.use_behavior_intensity
            or self.history_forgetting_mode == "learned_exponential"
        )
        user_context: torch.Tensor | None = None
        if needs_user_context:
            context_values = item_embedding + self.behavior_embedding(behavior_types)
            context_weights = history_mask.unsqueeze(-1).to(item_embedding.dtype)
            user_context = (context_values * context_weights).sum(dim=1) / context_weights.sum(
                dim=1
            ).clamp_min(1.0)
        if self.use_behavior_intensity:
            if behavior_intensities is None:
                behavior_intensities = torch.ones_like(time_gap_buckets, dtype=item_embedding.dtype)
            if behavior_intensities.shape != history_mask.shape:
                raise ValueError("behavior_intensities must match history_mask shape")
            behavior_intensities = behavior_intensities.to(item_embedding.dtype).clamp(0.0, 1.0)
            assert user_context is not None
            decay_rate = F.softplus(self.personalized_decay(user_context)).squeeze(-1) + 1e-4
            normalized_gap = time_gap_buckets.to(item_embedding.dtype) / max(
                1, self.num_time_buckets - 1
            )
            event_strengths = behavior_intensities * torch.exp(
                -decay_rate.unsqueeze(-1) * normalized_gap
            )
            event_embedding = base_event_embedding + self.intensity_projection(
                behavior_intensities.unsqueeze(-1)
            ) * event_strengths.unsqueeze(-1)
        else:
            event_embedding = base_event_embedding
        if self.history_forgetting_mode == "learned_exponential":
            assert user_context is not None
            forgetting_rate = (
                F.softplus(self.history_forgetting(user_context)).squeeze(-1) + 1e-4
            )
            sequence_positions = torch.arange(
                history_tokens.size(1), device=history_tokens.device
            ).unsqueeze(0)
            history_lengths = history_mask.sum(dim=1, keepdim=True)
            age = (history_lengths - 1 - sequence_positions).clamp_min(0).to(item_embedding.dtype)
            normalized_age = age / (history_lengths - 1).clamp_min(1).to(item_embedding.dtype)
            history_retention = torch.exp(-forgetting_rate.unsqueeze(-1) * normalized_age)
            history_retention = history_retention * history_mask.to(item_embedding.dtype)
        encoded = self.history_encoder(
            event_embedding,
            mask=self._causal_mask(history_tokens.size(1), history_tokens.device),
            src_key_padding_mask=~history_mask,
        )
        preference = self.reasoner(
            encoded,
            history_mask,
            event_strengths,
            history_retention,
        )
        preference.decay_rate = decay_rate
        preference.history_retention = history_retention
        preference.forgetting_rate = forgetting_rate
        memory_events = (
            encoded * history_retention.unsqueeze(-1)
            if history_retention is not None
            else encoded
        )
        memory = torch.cat([preference.vector.unsqueeze(1), memory_events], dim=1)
        return memory, preference

    def forward(
        self,
        history_tokens: torch.Tensor,
        behavior_types: torch.Tensor,
        time_gap_buckets: torch.Tensor,
        history_mask: torch.Tensor,
        target_input: torch.Tensor,
        behavior_intensities: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PreferenceState]:
        memory, preference = self.encode_history(
            history_tokens,
            behavior_types,
            time_gap_buckets,
            history_mask,
            behavior_intensities,
        )
        logits = self.decode(memory, history_mask, target_input)
        return logits, preference

    def decode(
        self,
        memory: torch.Tensor,
        history_mask: torch.Tensor,
        target_input: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a target prefix against precomputed user-history memory."""
        positions = torch.arange(target_input.size(1), device=target_input.device)
        target = self.token_embedding(target_input) + self.target_position(positions).unsqueeze(0)
        memory_padding_mask = torch.cat(
            [
                torch.zeros(
                    (history_mask.size(0), 1), dtype=torch.bool, device=history_mask.device
                ),
                ~history_mask,
            ],
            dim=1,
        )
        decoded = self.decoder(
            target,
            memory,
            tgt_mask=self._causal_mask(target_input.size(1), target_input.device),
            tgt_key_padding_mask=target_input.eq(self.pad_token_id),
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.output_projection(decoded)

    def loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        sample_weights: torch.Tensor | None = None,
        token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.shape[:2] != labels.shape:
            raise ValueError("logits and labels must have aligned batch/sequence axes")
        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.pad_token_id,
            reduction="none",
        ).reshape_as(labels)
        active = labels.ne(self.pad_token_id)
        weights = active.to(logits.dtype)
        if sample_weights is not None:
            if sample_weights.ndim != 1 or sample_weights.numel() != labels.size(0):
                raise ValueError("sample_weights must contain one value per batch row")
            sample_weights = sample_weights.to(device=logits.device, dtype=logits.dtype)
            if torch.any(sample_weights < 0) or not torch.any(sample_weights > 0):
                raise ValueError("sample_weights must be non-negative with positive mass")
            weights = weights * sample_weights.unsqueeze(-1)
        if token_weights is not None:
            if token_weights.shape != (labels.size(1), logits.size(-1)):
                raise ValueError(
                    "token_weights must have shape [target_length, vocabulary_size]"
                )
            token_weights = token_weights.to(device=logits.device, dtype=logits.dtype)
            if torch.any(token_weights < 0):
                raise ValueError("token_weights must be non-negative")
            positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
            weights = weights * token_weights[positions, labels]
        return (losses * weights).sum() / weights.sum().clamp_min(1e-12)

    def trie_constrained_loss(
        self,
        logits: torch.Tensor,
        target_input: torch.Tensor,
        labels: torch.Tensor,
        catalog: SemanticIdCatalog,
        *,
        position_weights: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
        token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-entropy over legal trie continuations instead of the global vocabulary.

        Deterministic trie transitions do not contribute a gradient. This keeps training
        focused on branches where the model actually has to rank multiple catalog paths
        and matches the constrained distribution used during generation.
        """
        if logits.shape[:2] != labels.shape or target_input.shape != labels.shape:
            raise ValueError("logits, target_input, and labels must have aligned batch/sequence axes")
        if logits.size(-1) != self.output_projection.out_features:
            raise ValueError("logits vocabulary size does not match the model")
        if position_weights is not None:
            if position_weights.ndim != 1 or position_weights.numel() != labels.size(1):
                raise ValueError("position_weights must have one value per target position")
            position_weights = position_weights.to(device=logits.device, dtype=logits.dtype)
        if sample_weights is not None:
            if sample_weights.ndim != 1 or sample_weights.numel() != labels.size(0):
                raise ValueError("sample_weights must contain one value per batch row")
            sample_weights = sample_weights.to(device=logits.device, dtype=logits.dtype)
        if token_weights is not None:
            if token_weights.shape != (labels.size(1), logits.size(-1)):
                raise ValueError(
                    "token_weights must have shape [target_length, vocabulary_size]"
                )
            token_weights = token_weights.to(device=logits.device, dtype=logits.dtype)

        losses: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for batch_index in range(labels.size(0)):
            for position in range(labels.size(1)):
                target = int(labels[batch_index, position])
                if target == self.pad_token_id:
                    continue
                prefix = target_input[batch_index, : position + 1].tolist()
                allowed = catalog.allowed_next(prefix)
                if target not in allowed:
                    raise ValueError("target label is not a legal continuation of its trie prefix")
                if len(allowed) <= 1:
                    continue
                allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=logits.device)
                branch_logits = logits[batch_index, position, allowed_tensor]
                target_offset = allowed.index(target)
                losses.append(torch.logsumexp(branch_logits, dim=0) - branch_logits[target_offset])
                if position_weights is None:
                    weight = logits.new_ones(())
                else:
                    weight = position_weights[position]
                if sample_weights is not None:
                    weight = weight * sample_weights[batch_index]
                if token_weights is not None:
                    weight = weight * token_weights[position, target]
                weights.append(weight)

        if not losses:
            return logits.sum() * 0.0
        stacked_losses = torch.stack(losses)
        stacked_weights = torch.stack(weights)
        if torch.any(stacked_weights < 0) or not torch.any(stacked_weights > 0):
            raise ValueError("position_weights must be non-negative with positive active weight")
        return (stacked_losses * stacked_weights).sum() / stacked_weights.sum()

    @torch.no_grad()
    def generate(
        self,
        *,
        history_tokens: torch.Tensor,
        behavior_types: torch.Tensor,
        time_gap_buckets: torch.Tensor,
        history_mask: torch.Tensor,
        catalog: SemanticIdCatalog,
        behavior_intensities: torch.Tensor | None = None,
    ) -> tuple[list[str | None], torch.Tensor]:
        """Greedy trie-constrained generation for a batch of user histories."""
        self.eval()
        batch_size = history_tokens.size(0)
        memory, _ = self.encode_history(
            history_tokens,
            behavior_types,
            time_gap_buckets,
            history_mask,
            behavior_intensities,
        )
        generated = torch.full(
            (batch_size, 1),
            catalog.special_tokens.bos,
            dtype=torch.long,
            device=history_tokens.device,
        )

        for _ in range(catalog.sequence_length + 1):
            logits = self.decode(memory, history_mask, generated)
            next_logits = logits[:, -1].clone()
            for batch_index in range(batch_size):
                prefix = generated[batch_index].tolist()
                allowed = catalog.allowed_next(prefix)
                mask = torch.ones_like(next_logits[batch_index], dtype=torch.bool)
                if allowed:
                    mask[torch.tensor(allowed, device=mask.device)] = False
                next_logits[batch_index].masked_fill_(mask, float("-inf"))
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

        item_ids = [catalog.resolve(row.tolist()) for row in generated]
        return item_ids, generated

    @torch.no_grad()
    def generate_beam(
        self,
        *,
        history_tokens: torch.Tensor,
        behavior_types: torch.Tensor,
        time_gap_buckets: torch.Tensor,
        history_mask: torch.Tensor,
        catalog: SemanticIdCatalog,
        beam_size: int = 10,
        num_return_sequences: int | None = None,
        renormalize_allowed_tokens: bool = False,
        behavior_intensities: torch.Tensor | None = None,
    ) -> tuple[list[list[str]], list[list[float]]]:
        """Return trie-constrained ranked item lists with vectorized beam search."""
        if beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if num_return_sequences is None:
            num_return_sequences = beam_size
        if num_return_sequences <= 0 or num_return_sequences > beam_size:
            raise ValueError("num_return_sequences must be in [1, beam_size]")
        self.eval()
        memory, _ = self.encode_history(
            history_tokens,
            behavior_types,
            time_gap_buckets,
            history_mask,
            behavior_intensities,
        )
        return self.generate_beam_from_memory(
            memory=memory,
            history_mask=history_mask,
            catalog=catalog,
            beam_size=beam_size,
            num_return_sequences=num_return_sequences,
            renormalize_allowed_tokens=renormalize_allowed_tokens,
        )

    @torch.no_grad()
    def generate_beam_with_preference(
        self,
        *,
        history_tokens: torch.Tensor,
        behavior_types: torch.Tensor,
        time_gap_buckets: torch.Tensor,
        history_mask: torch.Tensor,
        catalog: SemanticIdCatalog,
        beam_size: int = 10,
        num_return_sequences: int | None = None,
        renormalize_allowed_tokens: bool = False,
        behavior_intensities: torch.Tensor | None = None,
    ) -> tuple[list[list[str]], list[list[float]], PreferenceState]:
        """Beam search plus the exact preference state used for decoding.

        Serving uses this path so preference diagnostics do not require a second
        history-encoder pass. The ranked items and scores are identical to
        :meth:`generate_beam` for the same inputs.
        """
        if beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if num_return_sequences is None:
            num_return_sequences = beam_size
        if num_return_sequences <= 0 or num_return_sequences > beam_size:
            raise ValueError("num_return_sequences must be in [1, beam_size]")
        self.eval()
        memory, preference = self.encode_history(
            history_tokens,
            behavior_types,
            time_gap_buckets,
            history_mask,
            behavior_intensities,
        )
        items, scores = self.generate_beam_from_memory(
            memory=memory,
            history_mask=history_mask,
            catalog=catalog,
            beam_size=beam_size,
            num_return_sequences=num_return_sequences,
            renormalize_allowed_tokens=renormalize_allowed_tokens,
        )
        return items, scores, preference

    @torch.no_grad()
    def generate_beam_from_memory(
        self,
        *,
        memory: torch.Tensor,
        history_mask: torch.Tensor,
        catalog: SemanticIdCatalog,
        beam_size: int = 10,
        num_return_sequences: int | None = None,
        renormalize_allowed_tokens: bool = False,
    ) -> tuple[list[list[str]], list[list[float]]]:
        """Run trie-constrained beam search from precomputed history memory."""
        if beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if num_return_sequences is None:
            num_return_sequences = beam_size
        if num_return_sequences <= 0 or num_return_sequences > beam_size:
            raise ValueError("num_return_sequences must be in [1, beam_size]")
        if memory.ndim != 3 or history_mask.ndim != 2:
            raise ValueError("memory and history_mask must be batched tensors")
        if memory.size(0) != history_mask.size(0):
            raise ValueError("memory and history_mask batch sizes must match")
        self.eval()
        batch_size = history_mask.size(0)
        device = memory.device
        vocab_size = self.output_projection.out_features
        sequences = torch.full(
            (batch_size, 1, 1),
            catalog.special_tokens.bos,
            dtype=torch.long,
            device=device,
        )
        scores = torch.zeros((batch_size, 1), dtype=memory.dtype, device=device)

        for _ in range(catalog.sequence_length + 1):
            current_beam = sequences.size(1)
            flat_sequences = sequences.reshape(batch_size * current_beam, -1)
            expanded_memory = memory.repeat_interleave(current_beam, dim=0)
            expanded_history_mask = history_mask.repeat_interleave(current_beam, dim=0)
            logits = self.decode(
                expanded_memory,
                expanded_history_mask,
                flat_sequences,
            )[:, -1]
            logits = logits.reshape(batch_size, current_beam, vocab_size)
            if renormalize_allowed_tokens:
                constrained = torch.full_like(logits, float("-inf"))
            else:
                log_probabilities = F.log_softmax(logits, dim=-1)
                constrained = torch.full_like(log_probabilities, float("-inf"))
            for batch_index in range(batch_size):
                for beam_index in range(current_beam):
                    allowed = catalog.allowed_next(sequences[batch_index, beam_index].tolist())
                    if allowed:
                        allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
                        if renormalize_allowed_tokens:
                            constrained[batch_index, beam_index, allowed_tensor] = F.log_softmax(
                                logits[batch_index, beam_index, allowed_tensor], dim=-1
                            )
                        else:
                            constrained[batch_index, beam_index, allowed_tensor] = (
                                log_probabilities[batch_index, beam_index, allowed_tensor]
                            )

            candidate_scores = constrained + scores.unsqueeze(-1)
            finite_per_example = torch.isfinite(candidate_scores).sum(dim=(1, 2))
            next_beam = min(beam_size, int(finite_per_example.min()))
            if next_beam == 0:
                raise RuntimeError("catalog trie produced no valid beam continuation")
            scores, flat_indices = candidate_scores.flatten(1).topk(next_beam, dim=-1)
            parent_indices = torch.div(flat_indices, vocab_size, rounding_mode="floor")
            next_tokens = flat_indices.remainder(vocab_size)
            gather_indices = parent_indices.unsqueeze(-1).expand(-1, -1, sequences.size(-1))
            parent_sequences = sequences.gather(1, gather_indices)
            sequences = torch.cat([parent_sequences, next_tokens.unsqueeze(-1)], dim=-1)

        ranked_items: list[list[str]] = []
        ranked_scores: list[list[float]] = []
        for batch_index in range(batch_size):
            items: list[str] = []
            item_scores: list[float] = []
            for sequence, score in zip(
                sequences[batch_index, :num_return_sequences].tolist(),
                scores[batch_index, :num_return_sequences].tolist(),
                strict=True,
            ):
                item_id = catalog.resolve(sequence)
                if item_id is not None and item_id not in items:
                    items.append(item_id)
                    item_scores.append(float(score))
            ranked_items.append(items)
            ranked_scores.append(item_scores)
        return ranked_items, ranked_scores

    @torch.no_grad()
    def score_semantic_candidates(
        self,
        *,
        history_tokens: torch.Tensor,
        behavior_types: torch.Tensor,
        time_gap_buckets: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_mask: torch.Tensor,
        behavior_intensities: torch.Tensor | None = None,
        pair_batch_size: int = 1024,
    ) -> torch.Tensor:
        """Score every supplied Semantic-ID sequence with teacher forcing.

        Scores are sums of global-vocabulary token log probabilities, matching
        ``generate_beam(..., renormalize_allowed_tokens=False)``. Unlike the old
        candidate fusion path, this gives retrieval-only items a genuine score
        from the frozen generator instead of an artificial zero.
        """
        if candidate_tokens.ndim != 3:
            raise ValueError(
                "candidate_tokens must have shape [batch, candidates, semantic_length]"
            )
        if candidate_mask.shape != candidate_tokens.shape[:2]:
            raise ValueError("candidate_mask must match candidate batch/candidate axes")
        if candidate_tokens.size(0) != history_tokens.size(0):
            raise ValueError("history and candidate batch sizes must match")
        if pair_batch_size <= 0:
            raise ValueError("pair_batch_size must be positive")
        if torch.any(candidate_tokens[candidate_mask] == self.pad_token_id):
            raise ValueError("active candidate Semantic IDs cannot contain padding")
        self.eval()
        memory, _ = self.encode_history(
            history_tokens,
            behavior_types,
            time_gap_buckets,
            history_mask,
            behavior_intensities,
        )
        active = candidate_mask.nonzero(as_tuple=False)
        output = torch.full(
            candidate_mask.shape,
            float("-inf"),
            dtype=memory.dtype,
            device=memory.device,
        )
        if active.numel() == 0:
            return output
        semantic_length = candidate_tokens.size(2)
        bos = self.output_projection.out_features - 2
        eos = self.output_projection.out_features - 1
        # Catalog construction always places PAD/BOS/EOS at vocab-3/vocab-2/vocab-1.
        if bos == self.pad_token_id:
            raise ValueError("invalid catalog special-token layout")
        for start in range(0, active.size(0), pair_batch_size):
            positions = active[start : start + pair_batch_size]
            user_indices = positions[:, 0]
            row_tokens = candidate_tokens[positions[:, 0], positions[:, 1]]
            target_input = torch.full(
                (len(positions), semantic_length + 1),
                bos,
                dtype=torch.long,
                device=memory.device,
            )
            target_input[:, 1:] = row_tokens
            labels = torch.full_like(target_input, eos)
            labels[:, :-1] = row_tokens
            logits = self.decode(
                memory[user_indices],
                history_mask[user_indices],
                target_input,
            )
            log_probabilities = F.log_softmax(logits, dim=-1)
            scores = log_probabilities.gather(-1, labels.unsqueeze(-1)).squeeze(-1).sum(dim=1)
            output[positions[:, 0], positions[:, 1]] = scores
        return output
