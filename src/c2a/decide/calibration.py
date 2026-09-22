"""Calibration metrics for deciders: multi-class Brier score and expected calibration error."""

from __future__ import annotations

from collections.abc import Sequence

from c2a.schemas import DecisionResult


def brier(results: Sequence[DecisionResult], truths: Sequence[str]) -> float:
    if len(results) != len(truths) or not results:
        raise ValueError("results and truths must be non-empty and equal length")
    total = 0.0
    for r, t in zip(results, truths, strict=True):
        total += sum((p - (1.0 if label == t else 0.0)) ** 2 for label, p in r.probs.items())
    return total / len(results)


def ece(results: Sequence[DecisionResult], truths: Sequence[str], n_bins: int = 10) -> float:
    """Top-label ECE with equal-width confidence bins."""
    if len(results) != len(truths) or not results:
        raise ValueError("results and truths must be non-empty and equal length")
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for r, t in zip(results, truths, strict=True):
        conf = r.probs[r.label]
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, r.label == t))
    n = len(results)
    return sum(
        len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(ok for _, ok in b) / len(b))
        for b in bins
        if b
    )


def accuracy(results: Sequence[DecisionResult], truths: Sequence[str]) -> float:
    return sum(r.label == t for r, t in zip(results, truths, strict=True)) / len(results)
