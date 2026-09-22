"""Store discovery: Firecrawl /search expansion + platform fingerprinting + UCP probe. M1."""

from __future__ import annotations

from c2a import NotYetImplemented
from c2a.schemas import Store


def fingerprint(domain: str) -> str:
    """Return shopify | woocommerce | ucp | custom | unknown."""
    raise NotYetImplemented("platform fingerprinting", "M1")


def expand(country: str, category: str, limit: int = 20) -> list[Store]:
    raise NotYetImplemented("store discovery via Firecrawl search", "M1")
