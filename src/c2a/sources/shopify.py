"""Shopify storefront `/products.json` ingestion (tier A: structured, no LLM extraction)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx

from c2a.normalize import to_minor_units
from c2a.schemas import Money, Product, Store, Variant


def parse_product(raw: dict[str, Any], store: Store, currency: str) -> Product:
    variants = [
        Variant(
            id=str(v["id"]),
            title=v.get("title") or "",
            sku=v.get("sku") or None,
            options={
                opt["name"]: v.get(f"option{i + 1}")
                for i, opt in enumerate(raw.get("options", []))
                if v.get(f"option{i + 1}") is not None
            },
            price=Money(
                amount=to_minor_units(Decimal(str(v["price"])), currency), currency=currency
            ),
            compare_at_price=(
                Money(
                    amount=to_minor_units(Decimal(str(v["compare_at_price"])), currency),
                    currency=currency,
                )
                if v.get("compare_at_price")
                else None
            ),
            available=bool(v.get("available", True)),
        )
        for v in raw.get("variants", [])
    ]
    tags = raw.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    return Product(
        id=f"{store.id}:{raw['id']}",
        store_id=store.id,
        native_id=str(raw["id"]),
        title=raw["title"],
        description=raw.get("body_html") or "",
        brand=raw.get("vendor") or None,
        category=raw.get("product_type") or None,
        tags=tags,
        url=f"https://{store.domain}/products/{raw['handle']}" if raw.get("handle") else None,
        image_urls=[img["src"] for img in raw.get("images", []) if img.get("src")],
        variants=variants,
        locale=store.locale,
        source="shopify",
        crawled_at=datetime.now(UTC),
    )


def parse_products_json(
    payload: dict[str, Any], store: Store, currency: str
) -> tuple[list[Product], list[str]]:
    """Return (products, errors). Products without variants are skipped with an error."""
    products, errors = [], []
    for raw in payload.get("products", []):
        try:
            products.append(parse_product(raw, store, currency))
        except Exception as exc:  # collect and continue
            errors.append(f"{raw.get('id')}: {exc}")
    return products, errors


def iter_pages(
    store: Store, client: httpx.Client, limit: int = 250, max_pages: int = 100
) -> Iterator[dict[str, Any]]:
    """Yield raw `/products.json` pages. Caller must check compliance and rate limit first."""
    for page in range(1, max_pages + 1):
        resp = client.get(
            f"https://{store.domain}/products.json", params={"limit": limit, "page": page}
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("products"):
            return
        yield payload
