"""Ranking metrics (binary or graded relevance)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def dcg_at_k(ranked: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    return sum(
        (2 ** relevance.get(pid, 0.0) - 1) / math.log2(i + 2) for i, pid in enumerate(ranked[:k])
    )


def ndcg_at_k(ranked: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg_at_k(ranked, relevance, k) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def dedupe_preserving_order(ranked: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in ranked if not (x in seen or seen.add(x))]
