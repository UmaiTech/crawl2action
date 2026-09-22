import pytest

from c2a.graders.ranking import dedupe_preserving_order, ndcg_at_k, recall_at_k


def test_ndcg_perfect_and_reversed():
    rel = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], rel, 2) == pytest.approx(1.0)
    assert 0 < ndcg_at_k(["b", "a"], rel, 2) < 1
    assert ndcg_at_k(["x"], {}, 5) == 0.0


def test_recall_and_dedupe():
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5
    assert dedupe_preserving_order(["a", "b", "a", "c"]) == ["a", "b", "c"]
