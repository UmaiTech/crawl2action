import pytest

from c2a.decide.calibration import accuracy, brier, ece
from c2a.schemas import DecisionResult


def r(label, p):
    other = "b" if label == "a" else "a"
    return DecisionResult(label=label, probs={label: p, other: 1 - p})


def test_perfect_confident_predictions():
    res = [r("a", 1.0), r("b", 1.0)]
    assert (
        brier(res, ["a", "b"]) == 0.0
        and ece(res, ["a", "b"]) == 0.0
        and accuracy(res, ["a", "b"]) == 1
    )


def test_overconfident_wrong_is_penalized():
    res = [r("a", 0.9), r("a", 0.9)]
    assert ece(res, ["a", "b"]) == pytest.approx(0.4)
    assert brier(res, ["a", "b"]) == pytest.approx((0.02 + 1.62) / 2)
