"""Loaders for open behavioural datasets (licenses to verify before use). M1.

- Amazon Reviews 2023 (user histories, co-purchase)
- Amazon ESCI / SQID (query-product relevance labels)
- Amazon-M2 (multilingual sessions, KDD Cup 2023)
- H&M Personalized Fashion, RetailRocket, Diginetica
"""

from __future__ import annotations

from c2a import NotYetImplemented

DATASETS = (
    "amazon_reviews_2023",
    "esci",
    "sqid",
    "amazon_m2",
    "hm_fashion",
    "retailrocket",
    "diginetica",
)


def load(name: str, split: str = "train"):
    if name not in DATASETS:
        raise KeyError(name)
    raise NotYetImplemented(f"open dataset loader '{name}'", "M1")
