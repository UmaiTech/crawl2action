"""Candidate product pairs for pair_v1 labeling (exact / substitute / complement / irrelevant).

Within one store: the k most similar products per anchor (likely substitutes) plus a few
products from other categories (likely complements or irrelevant). Deterministic and
deduplicated as unordered pairs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from c2a.index import InMemoryRetriever
from c2a.labeling.records import LabelItem
from c2a.labeling.state import pair_state
from c2a.schemas import Product


def _stable_key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def candidate_pairs(
    products: Sequence[Product], k: int = 3, other: int = 1
) -> list[tuple[Product, Product]]:
    products = sorted(products, key=lambda p: p.id)  # ties resolve the same for any input order
    by_id = {p.id: p for p in products}
    retriever = InMemoryRetriever(products)
    seen: set[frozenset[str]] = set()
    pairs: list[tuple[Product, Product]] = []

    def add(a: Product, b: Product) -> None:
        key = frozenset((a.id, b.id))
        if a.id != b.id and key not in seen:
            seen.add(key)
            pairs.append((a, b))

    for anchor in products:
        query = " ".join([anchor.title, anchor.category or "", " ".join(anchor.tags)])
        similar = [c for c in retriever.search(query, k=k + 1) if c.product_id != anchor.id][:k]
        for c in similar:
            add(anchor, by_id[c.product_id])
        others = sorted(
            (p for p in products if p.id != anchor.id and p.category != anchor.category),
            key=lambda p: _stable_key(anchor.id, p.id),
        )
        for p in others[:other]:
            add(anchor, p)
    return pairs


def pair_items(pairs: Sequence[tuple[Product, Product]]) -> list[LabelItem]:
    return [LabelItem(id=f"pair:{a.id}|{b.id}", state=pair_state(a, b)) for a, b in pairs]
