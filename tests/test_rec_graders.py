import json

import pytest

from c2a.graders.base import composite
from c2a.graders.rec import RecGrader


def out(*ids):
    return json.dumps({"items": [{"product_id": i} for i in ids]})


def test_perfect_ranking_scores_high(rec_example):
    res = RecGrader(k=3).grade(out("a", "b"), rec_example)
    assert res.passed and res.components["ndcg"] == 1.0 and res.components["recall"] == 1.0
    assert res.score == 1.0


def test_hard_failures(rec_example):
    assert not RecGrader().grade("not json", rec_example).passed
    res = RecGrader().grade(out("a", "zzz"), rec_example)
    assert not res.passed and res.score == 0 and "zzz" in res.reasons[0]


def test_constraint_violations_reduce_score(rec_example):
    res = RecGrader(k=3).grade(out("a", "c", "d"), rec_example)  # c too pricey, d out of stock
    assert res.passed and res.components["constraints"] == pytest.approx(1 / 3)
    assert any("over max price" in r for r in res.reasons)


def test_composite_weights():
    assert composite({"x": 1.0, "y": 0.0}, {"x": 3, "y": 1}) == 0.75
