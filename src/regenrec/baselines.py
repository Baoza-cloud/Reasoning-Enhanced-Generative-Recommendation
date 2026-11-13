from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import chain

from regenrec.data import SequenceExample


def _rank_counts(counts: Counter[str]) -> list[str]:
    """Return a deterministic descending frequency ranking."""
    return [item_id for item_id, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]


class PopularityRecommender:
    """Non-personalized next-item baseline fitted on training targets only."""

    def __init__(self) -> None:
        self.item_counts: Counter[str] = Counter()
        self.ranking: list[str] = []

    def fit(self, examples: Iterable[SequenceExample]) -> "PopularityRecommender":
        self.item_counts.clear()
        for example in examples:
            self.item_counts[example.target_item_id] += 1
        if not self.item_counts:
            raise ValueError("cannot fit popularity baseline on an empty dataset")
        self.ranking = _rank_counts(self.item_counts)
        return self

    def recommend(
        self,
        example: SequenceExample,
        *,
        k: int,
        exclude_history: bool = False,
    ) -> list[str]:
        if k <= 0:
            raise ValueError("k must be positive")
        excluded = {event.item_id for event in example.history} if exclude_history else set()
        recommendations: list[str] = []
        for item_id in self.ranking:
            if item_id in excluded:
                continue
            recommendations.append(item_id)
            if len(recommendations) == k:
                break
        return recommendations


class FirstOrderMarkovRecommender:
    """Rank successors of the most recent item, then back off to popularity."""

    def __init__(self) -> None:
        self.popularity = PopularityRecommender()
        self.transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.transition_rankings: dict[str, list[str]] = {}

    def fit(self, examples: Iterable[SequenceExample]) -> "FirstOrderMarkovRecommender":
        cached = list(examples)
        if not cached:
            raise ValueError("cannot fit Markov baseline on an empty dataset")
        self.popularity.fit(cached)
        self.transition_counts.clear()
        for example in cached:
            if example.history:
                source = example.history[-1].item_id
                self.transition_counts[source][example.target_item_id] += 1
        self.transition_rankings = {
            source: _rank_counts(counts) for source, counts in self.transition_counts.items()
        }
        return self

    def recommend(
        self,
        example: SequenceExample,
        *,
        k: int,
        exclude_history: bool = False,
    ) -> list[str]:
        if k <= 0:
            raise ValueError("k must be positive")
        excluded = {event.item_id for event in example.history} if exclude_history else set()
        candidates: Sequence[str] = ()
        if example.history:
            candidates = self.transition_rankings.get(example.history[-1].item_id, ())
        recommendations: list[str] = []
        seen: set[str] = set()
        for item_id in chain(candidates, self.popularity.ranking):
            if item_id in seen or item_id in excluded:
                continue
            recommendations.append(item_id)
            seen.add(item_id)
            if len(recommendations) == k:
                break
        return recommendations
