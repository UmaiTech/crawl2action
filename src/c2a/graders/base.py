"""Grader protocol and weighted composite scoring."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class GradeResult(BaseModel):
    score: float  # in [0, 1] unless a hard failure sets it to 0
    passed: bool = True  # False = hard failure (invalid output, hallucination, ...)
    components: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class Grader(Protocol):
    name: str

    def grade(self, output: str, example: dict[str, Any]) -> GradeResult: ...


def composite(components: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean over the components present in `weights`."""
    missing = weights.keys() - components.keys()
    if missing:
        raise KeyError(f"missing components: {sorted(missing)}")
    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("weights must sum to > 0")
    return sum(components[k] * w for k, w in weights.items()) / total_w
