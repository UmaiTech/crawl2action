from __future__ import annotations

import math
from typing import Protocol

from c2a.schemas import DecisionRequest, DecisionResult


class Decider(Protocol):
    def decide(self, request: DecisionRequest) -> DecisionResult: ...


def normalize_probs(
    scores: dict[str, float], labels: list[str], temperature: float = 1.0
) -> DecisionResult:
    """Softmax over raw scores (log-probs or logits); every label gets a probability."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    raw = [scores.get(label, -1e9) / temperature for label in labels]
    m = max(raw)
    exps = [math.exp(r - m) for r in raw]
    z = sum(exps)
    probs = {label: e / z for label, e in zip(labels, exps, strict=True)}
    # absorb float error so probabilities sum to exactly 1
    top = max(probs, key=probs.get)
    probs[top] += 1.0 - sum(probs.values())
    return DecisionResult(label=top, probs=probs)
