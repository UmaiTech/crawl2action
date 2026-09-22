from __future__ import annotations

import math
from typing import Protocol

from c2a.decide.systemone import Choice, ChoiceAnswer, SystemOneRequest, SystemOneResponse
from c2a.schemas import DecisionResult


class Decider(Protocol):
    """Anything that answers System One requests (hosted Jev, our decider, test backends)."""

    def ask(self, request: SystemOneRequest) -> SystemOneResponse: ...


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


def decide(
    decider: Decider, evidence: str, question: str, labels: list[str], model: str | None = None
) -> DecisionResult:
    """Single-choice convenience wrapper over the System One contract."""
    req = SystemOneRequest(
        state=evidence,
        questions={"q": Choice(instructions=question, criteria=dict.fromkeys(labels))},
        **({"model": model} if model else {}),
    )
    answer = decider.ask(req).answers["q"]
    if not isinstance(answer, ChoiceAnswer):
        raise TypeError(f"expected a choice answer, got {answer.type}")
    probs = {label: float(answer.probabilities.get(label, 0.0)) for label in labels}
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("decider returned no probability mass for the requested labels")
    probs = {k: v / total for k, v in probs.items()}
    top = max(probs, key=probs.get)
    probs[top] += 1.0 - sum(probs.values())
    return DecisionResult(label=top, probs=probs)
