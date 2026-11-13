from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.nn import functional as F

from regenrec.catalog import SemanticIdCatalog
from regenrec.config import load_config
from regenrec.data import Interaction, SequenceExample, iter_sequence_examples_jsonl
from regenrec.embeddings import load_embedding_bundle
from regenrec.metrics import popularity_strata
from regenrec.models.generative import GenerativeRecommender, PreferenceState
from regenrec.reranking import (
    build_blair_user_profiles,
    build_candidate_features,
    exact_blair_retrieve,
    fused_soft_quota_rerank_with_scores,
    score_item_rows,
)


LONG_TAIL_PARAMETERS: dict[str, dict[str, object]] = {
    "mild": {
        "retrieval_weight": 0.25,
        "frequency_alpha": 0.10,
        "quota_shares": {"head": 0.60, "torso": 0.20, "observed_tail": 0.20},
        "quota_strength": 0.01,
    },
    "aggressive": {
        "retrieval_weight": 0.25,
        "frequency_alpha": 0.20,
        "quota_shares": {"head": 0.60, "torso": 0.25, "observed_tail": 0.15},
        "quota_strength": 0.05,
    },
}


@dataclass(frozen=True)
class ServingSettings:
    config_path: Path
    artifact_dir: Path
    items_path: Path
    embedding_path: Path
    device: str = "cpu"
    max_beam_size: int = 20
    max_batch_size: int = 32
    retrieval_count: int = 50
    torch_threads: int = 0

    @classmethod
    def from_env(cls) -> ServingSettings:
        return cls(
            config_path=Path(
                os.getenv("REGENREC_CONFIG", "configs/v2_final_default.yaml")
            ),
            artifact_dir=Path(
                os.getenv(
                    "REGENREC_MODEL_DIR",
                    "artifacts/amazon2023_musical_instruments_long_tail_training_v1/"
                    "formal/token_loss_010/seed_2026",
                )
            ),
            items_path=Path(
                os.getenv(
                    "REGENREC_ITEMS_PATH",
                    "data/amazon2023/processed/Musical_Instruments/last_out_v2/items.jsonl",
                )
            ),
            embedding_path=Path(
                os.getenv(
                    "REGENREC_EMBEDDINGS_PATH",
                    "data/amazon2023/processed/Musical_Instruments/last_out/"
                    "item_embeddings_blair.pt",
                )
            ),
            device=os.getenv("REGENREC_DEVICE", "cpu"),
            max_beam_size=int(os.getenv("REGENREC_MAX_BEAM_SIZE", "20")),
            max_batch_size=int(os.getenv("REGENREC_MAX_BATCH_SIZE", "32")),
            retrieval_count=int(os.getenv("REGENREC_RETRIEVAL_COUNT", "50")),
            torch_threads=int(os.getenv("REGENREC_TORCH_THREADS", "0")),
        )

    def validate(self, *, require_long_tail: bool = True) -> None:
        required = {
            "config": self.config_path,
            "catalog": self.artifact_dir / "catalog.json",
            "generator": self.artifact_dir / "generator.pt",
            "items": self.items_path,
        }
        if require_long_tail:
            required["BLaIR embeddings"] = self.embedding_path
        missing = [f"{name}={path}" for name, path in required.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing serving assets: " + ", ".join(missing))
        if self.max_beam_size <= 0 or self.max_batch_size <= 0 or self.retrieval_count <= 0:
            raise ValueError("beam, batch, and retrieval limits must be positive")


def _load_item_metadata(path: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            item_id = str(payload["item_id"])
            if item_id in metadata:
                raise ValueError(f"duplicate item metadata on line {line_number}: {item_id}")
            text = str(payload.get("text", ""))
            title = text.split(" [SEP] ", maxsplit=1)[0].strip() or item_id
            metadata[item_id] = {"title": title, "text": text}
    if not metadata:
        raise ValueError(f"no item metadata found in {path}")
    return metadata


def _build_model(
    config: dict[str, object],
    catalog: SemanticIdCatalog,
    model_path: Path,
    device: torch.device,
) -> GenerativeRecommender:
    data_config = config["data"]
    generator_config = config["generator"]
    model = GenerativeRecommender(
        vocab_size=catalog.vocab_size,
        pad_token_id=catalog.special_tokens.pad,
        num_behaviors=int(data_config["num_behaviors"]),
        num_time_buckets=int(data_config["num_time_buckets"]),
        model_dim=int(generator_config["model_dim"]),
        num_heads=int(generator_config["num_heads"]),
        num_encoder_layers=int(generator_config["num_encoder_layers"]),
        num_decoder_layers=int(generator_config["num_decoder_layers"]),
        feedforward_dim=int(generator_config["feedforward_dim"]),
        dropout=float(generator_config["dropout"]),
        short_term_window=int(generator_config["short_term_window"]),
        max_history_length=int(data_config["max_history"]),
        max_target_length=catalog.sequence_length + 2,
        use_behavior_intensity=bool(
            generator_config.get("use_behavior_intensity", False)
        ),
        preference_reasoning_mode=str(
            generator_config.get("preference_reasoning_mode", "full")
        ),
        history_forgetting_mode=str(
            generator_config.get("history_forgetting_mode", "none")
        ),
    ).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


class RecommendationEngine:
    """Loads the frozen recommender and serves catalog-valid ranked products."""

    def __init__(
        self,
        *,
        settings: ServingSettings,
        config: dict[str, object],
        catalog: SemanticIdCatalog,
        model: GenerativeRecommender,
        item_metadata: dict[str, dict[str, str]],
        item_embeddings: torch.Tensor,
        embedding_ids: list[str],
        item_frequencies: Counter[str],
    ) -> None:
        self.settings = settings
        self.config = config
        self.catalog = catalog
        self.model = model
        self.item_metadata = item_metadata
        self.device = torch.device(settings.device)
        self.max_history = int(config["data"]["max_history"])
        self.num_behaviors = int(config["data"]["num_behaviors"])
        self.num_time_buckets = int(config["data"]["num_time_buckets"])
        self.short_term_window = int(config["generator"]["short_term_window"])
        self.item_embeddings = F.normalize(item_embeddings.to(torch.float32), p=2, dim=1)
        self.item_to_embedding = {
            item_id: index for index, item_id in enumerate(embedding_ids)
        }
        self.item_frequencies = item_frequencies
        self.strata = popularity_strata(catalog.item_ids, item_frequencies)
        self.item_groups = {
            item_id: group for group, items in self.strata.items() for item_id in items
        }
        self.observed_tail_ids = [
            item_id
            for item_id in catalog.item_ids
            if item_id in self.strata["observed_tail"]
        ]
        self._inference_lock = threading.Lock()

    @classmethod
    def load(cls, settings: ServingSettings | None = None) -> RecommendationEngine:
        settings = settings or ServingSettings.from_env()
        settings.validate(require_long_tail=True)
        if settings.torch_threads > 0:
            torch.set_num_threads(settings.torch_threads)
        config = load_config(settings.config_path)
        catalog = SemanticIdCatalog.load(settings.artifact_dir / "catalog.json")
        item_metadata = _load_item_metadata(settings.items_path)
        missing_metadata = sorted(set(catalog.item_ids) - set(item_metadata))
        if missing_metadata:
            raise ValueError(
                f"item metadata misses {len(missing_metadata)} catalog products"
            )
        device = torch.device(settings.device)
        model = _build_model(
            config,
            catalog,
            settings.artifact_dir / "generator.pt",
            device,
        )
        embedding_ids, item_embeddings, embedding_metadata = load_embedding_bundle(
            settings.embedding_path
        )
        if embedding_metadata.get("backend") != "blair":
            raise ValueError("long-tail serving requires a verified BLaIR embedding bundle")
        if set(embedding_ids) != set(catalog.item_ids):
            raise ValueError("embedding products and Semantic-ID catalog must match")
        train_path = Path(config["data"]["train_path"])
        if not train_path.is_file():
            raise FileNotFoundError(f"training split required for tail strata: {train_path}")
        frequencies: Counter[str] = Counter()
        for example in iter_sequence_examples_jsonl(train_path):
            frequencies[example.target_item_id] += 1
        return cls(
            settings=settings,
            config=config,
            catalog=catalog,
            model=model,
            item_metadata=item_metadata,
            item_embeddings=item_embeddings,
            embedding_ids=embedding_ids,
            item_frequencies=frequencies,
        )

    def _validate_histories(
        self, histories: Sequence[Sequence[Interaction]]
    ) -> list[tuple[Interaction, ...]]:
        if not histories:
            raise ValueError("at least one history is required")
        if len(histories) > self.settings.max_batch_size:
            raise ValueError(
                f"batch size {len(histories)} exceeds {self.settings.max_batch_size}"
            )
        output: list[tuple[Interaction, ...]] = []
        for row_index, history in enumerate(histories):
            if not history:
                raise ValueError(f"history {row_index} cannot be empty")
            clipped = tuple(history[-self.max_history :])
            for event in clipped:
                if event.item_id not in self.catalog.item_to_index:
                    raise ValueError(f"unknown catalog item: {event.item_id}")
                if not 0 <= event.behavior < self.num_behaviors:
                    raise ValueError(
                        f"behavior must be in [0, {self.num_behaviors - 1}]"
                    )
                if not 0 <= event.time_gap_bucket < self.num_time_buckets:
                    raise ValueError(
                        f"time_gap_bucket must be in [0, {self.num_time_buckets - 1}]"
                    )
                if not 0.0 <= event.behavior_intensity <= 1.0:
                    raise ValueError("behavior_intensity must be in [0, 1]")
            output.append(clipped)
        return output

    def _collate_histories(
        self, histories: Sequence[Sequence[Interaction]]
    ) -> dict[str, torch.Tensor]:
        maximum = max(len(history) for history in histories)
        batch_size = len(histories)
        semantic_length = self.catalog.sequence_length
        history_tokens = torch.full(
            (batch_size, maximum, semantic_length),
            self.catalog.special_tokens.pad,
            dtype=torch.long,
        )
        behavior_types = torch.zeros((batch_size, maximum), dtype=torch.long)
        behavior_intensities = torch.zeros((batch_size, maximum), dtype=torch.float32)
        time_gap_buckets = torch.zeros((batch_size, maximum), dtype=torch.long)
        history_mask = torch.zeros((batch_size, maximum), dtype=torch.bool)
        for row_index, history in enumerate(histories):
            for column, event in enumerate(history):
                history_tokens[row_index, column] = self.catalog.tokens_for_item(
                    event.item_id
                )
                behavior_types[row_index, column] = event.behavior
                behavior_intensities[row_index, column] = event.behavior_intensity
                time_gap_buckets[row_index, column] = event.time_gap_bucket
                history_mask[row_index, column] = True
        return {
            "history_tokens": history_tokens.to(self.device),
            "behavior_types": behavior_types.to(self.device),
            "behavior_intensities": behavior_intensities.to(self.device),
            "time_gap_buckets": time_gap_buckets.to(self.device),
            "history_mask": history_mask.to(self.device),
        }

    def _preference_basis(
        self,
        preference: PreferenceState,
        histories: Sequence[Sequence[Interaction]],
        row_index: int,
    ) -> dict[str, object]:
        history = histories[row_index]
        gate = float(preference.short_term_weight[row_index].mean().cpu())
        cosine = float(
            F.cosine_similarity(
                preference.long_term[row_index].unsqueeze(0),
                preference.short_term[row_index].unsqueeze(0),
            ).cpu()
        )
        if preference.event_weights is not None:
            weights = preference.event_weights[row_index, : len(history)].detach().cpu()
            evidence_indices = torch.argsort(weights, descending=True)[:3].tolist()
        else:
            evidence_indices = list(range(max(0, len(history) - 3), len(history)))[::-1]
        evidence = [
            {
                "item_id": history[index].item_id,
                "title": self.item_metadata[history[index].item_id]["title"],
            }
            for index in evidence_indices
        ]
        emphasis = "近期行为" if gate >= 0.5 else "长期兴趣"
        return {
            "history_length": len(history),
            "recent_window": min(len(history), self.short_term_window),
            "short_term_weight": gate,
            "long_short_cosine": cosine,
            "evidence_items": evidence,
            "summary": (
                f"偏好融合更侧重{emphasis}（近期权重 {gate:.1%}）；"
                f"长期与近期方向相似度 {cosine:.3f}。"
            ),
        }

    def _semantic_trace(self, item_id: str) -> list[dict[str, int | str]]:
        tokens = self.catalog.tokens_for_item(item_id).tolist()
        trace: list[dict[str, int | str]] = []
        for level, token in enumerate(tokens[: self.catalog.num_levels]):
            trace.append(
                {
                    "step": level + 1,
                    "component": f"rq_level_{level + 1}",
                    "token": token,
                    "code": token - level * self.catalog.codebook_size,
                }
            )
        collision_token = tokens[-1]
        trace.append(
            {
                "step": self.catalog.num_levels + 1,
                "component": "collision_rank",
                "token": collision_token,
                "code": collision_token
                - self.catalog.num_levels * self.catalog.codebook_size,
            }
        )
        return trace

    def _long_tail_rank(
        self,
        histories: Sequence[Sequence[Interaction]],
        generator_items: list[list[str]],
        generator_scores: list[list[float]],
        mode: str,
    ) -> list[list[tuple[str, float]]]:
        examples = [
            SequenceExample(
                user_id=f"serving-{index}",
                history=tuple(history),
                target_item_id=self.catalog.item_ids[0],
            )
            for index, history in enumerate(histories)
        ]
        profiles = build_blair_user_profiles(
            examples, self.item_embeddings, self.item_to_embedding
        )
        retrieval_count = min(
            self.settings.retrieval_count, len(self.observed_tail_ids)
        )
        retrieval_items, retrieval_scores = exact_blair_retrieve(
            profiles,
            self.observed_tail_ids,
            self.item_embeddings,
            self.item_to_embedding,
            [[event.item_id for event in history] for history in histories],
            top_k=retrieval_count,
            batch_size=max(1, len(histories)),
        )
        generator_semantic_scores = score_item_rows(
            profiles,
            generator_items,
            self.item_embeddings,
            self.item_to_embedding,
            batch_size=max(1, len(histories)),
        )
        features = build_candidate_features(
            generator_items,
            generator_scores,
            generator_semantic_scores,
            retrieval_items,
            retrieval_scores,
            self.item_frequencies,
            self.item_groups,
        )
        parameters = LONG_TAIL_PARAMETERS[mode]
        return fused_soft_quota_rerank_with_scores(
            features,
            retrieval_weight=float(parameters["retrieval_weight"]),
            frequency_alpha=float(parameters["frequency_alpha"]),
            quota_shares=parameters["quota_shares"],
            quota_strength=float(parameters["quota_strength"]),
            top_k=10,
        )

    @torch.inference_mode()
    def recommend(
        self,
        histories: Sequence[Sequence[Interaction]],
        *,
        beam_size: int = 20,
        long_tail_mode: str = "off",
    ) -> list[dict[str, object]]:
        histories = self._validate_histories(histories)
        if not 1 <= beam_size <= self.settings.max_beam_size:
            raise ValueError(
                f"beam_size must be in [1, {self.settings.max_beam_size}]"
            )
        if long_tail_mode not in {"off", *LONG_TAIL_PARAMETERS}:
            raise ValueError("long_tail_mode must be off, mild, or aggressive")
        batch = self._collate_histories(histories)
        started = time.perf_counter()
        with self._inference_lock:
            generator_items, generator_scores, preference = (
                self.model.generate_beam_with_preference(
                    **batch,
                    catalog=self.catalog,
                    beam_size=beam_size,
                    num_return_sequences=beam_size,
                    renormalize_allowed_tokens=False,
                )
            )
            if long_tail_mode == "off":
                ranked = [
                    list(zip(items[:10], scores[:10], strict=True))
                    for items, scores in zip(
                        generator_items, generator_scores, strict=True
                    )
                ]
                score_type = "generator_log_probability"
            else:
                ranked = self._long_tail_rank(
                    histories,
                    generator_items,
                    generator_scores,
                    long_tail_mode,
                )
                score_type = "normalized_fusion_objective"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        responses: list[dict[str, object]] = []
        for row_index, row in enumerate(ranked):
            generator_set = set(generator_items[row_index])
            recommendations = []
            for rank, (item_id, score) in enumerate(row, start=1):
                recommendations.append(
                    {
                        "rank": rank,
                        "item_id": item_id,
                        "title": self.item_metadata[item_id]["title"],
                        "semantic_id": self.catalog.tokens_for_item(item_id).tolist(),
                        "final_score": float(score),
                        "score_type": score_type,
                        "popularity_group": self.item_groups[item_id],
                        "candidate_source": (
                            "generator" if item_id in generator_set else "blair_tail_retrieval"
                        ),
                    }
                )
            responses.append(
                {
                    "recommendations": recommendations,
                    "returned_count": len(recommendations),
                    "beam_size": beam_size,
                    "long_tail_mode": long_tail_mode,
                    "latency_ms": elapsed_ms,
                    "preference_basis": self._preference_basis(
                        preference, histories, row_index
                    ),
                    "beam_candidates": [
                        {
                            "item_id": item_id,
                            "title": self.item_metadata[item_id]["title"],
                            "semantic_id": self.catalog.tokens_for_item(item_id).tolist(),
                            "semantic_trace": self._semantic_trace(item_id),
                            "generator_score": float(score),
                        }
                        for item_id, score in zip(
                            generator_items[row_index],
                            generator_scores[row_index],
                            strict=True,
                        )
                    ],
                }
            )
        return responses
