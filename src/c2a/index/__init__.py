"""Candidate retrieval. Production: multilingual embeddings in LanceDB on a Modal Volume (M1).
M0 ships the protocol plus a tiny in-memory retriever for tests and the dev gateway."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from c2a.schemas import Candidate, Product


class Retriever(Protocol):
    def search(self, query: str, k: int = 50) -> list[Candidate]: ...


def _tokens(text: str) -> Counter[str]:
    return Counter(re.findall(r"\w+", text.lower()))


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    dot = sum(a[t] * b[t] for t in a.keys() & b.keys())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def to_candidate(p: Product) -> Candidate:
    return Candidate(
        product_id=p.id,
        title=p.title,
        price=p.min_price,
        in_stock=p.in_stock,
        attributes=p.attributes,
    )


class InMemoryRetriever:
    """Bag-of-words cosine over title/tags/category. Dev/test only."""

    def __init__(self, products: list[Product]) -> None:
        self._items = [
            (p, _tokens(" ".join([p.title, p.category or "", " ".join(p.tags)]))) for p in products
        ]

    def search(self, query: str, k: int = 50) -> list[Candidate]:
        q = _tokens(query)
        scored = sorted(self._items, key=lambda it: _cosine(q, it[1]), reverse=True)
        return [to_candidate(p) for p, vec in scored[:k] if _cosine(q, vec) > 0]
