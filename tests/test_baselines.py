from regenrec.baselines import FirstOrderMarkovRecommender, PopularityRecommender
from regenrec.data import Interaction, SequenceExample


def _example(user: str, history: list[str], target: str) -> SequenceExample:
    return SequenceExample(
        user_id=user,
        history=tuple(Interaction(item_id=item, behavior=1, time_gap_bucket=0) for item in history),
        target_item_id=target,
    )


def test_popularity_is_deterministic_and_uses_targets_only() -> None:
    train = [
        _example("u1", ["history_only"], "b"),
        _example("u2", ["x"], "a"),
        _example("u3", ["x"], "b"),
    ]
    model = PopularityRecommender().fit(train)
    assert model.ranking == ["b", "a"]
    assert model.recommend(train[0], k=2) == ["b", "a"]
    assert "history_only" not in model.ranking


def test_markov_uses_last_item_and_backs_off_without_duplicates() -> None:
    train = [
        _example("u1", ["x"], "b"),
        _example("u2", ["x"], "a"),
        _example("u3", ["x"], "b"),
        _example("u4", ["y"], "c"),
    ]
    model = FirstOrderMarkovRecommender().fit(train)
    query = _example("test", ["x"], "z")
    assert model.recommend(query, k=3) == ["b", "a", "c"]
    assert model.recommend(query, k=3, exclude_history=True) == ["b", "a", "c"]
