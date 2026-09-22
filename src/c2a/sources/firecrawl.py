"""Firecrawl /map + /crawl + /extract ingestion (tier B). Planned for M1.

Uses `schemas.Product` JSON schema for /extract, a credit budget per run, and must go
through `compliance.can_crawl` + `RateLimiter` for every URL.
"""

from __future__ import annotations

from c2a import NotYetImplemented
from c2a.schemas import Product, Store


def product_extract_schema() -> dict:
    """JSON schema passed to Firecrawl /extract."""
    return Product.model_json_schema()


def crawl_store(store: Store, max_pages: int, credit_budget: int) -> list[Product]:
    raise NotYetImplemented("Firecrawl crawl", "M1")
