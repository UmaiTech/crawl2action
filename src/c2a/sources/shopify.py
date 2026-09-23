"""Shopify storefront `/products.json` ingestion (tier A: structured, no LLM extraction)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from c2a.normalize import to_minor_units
from c2a.schemas import Money, Product, Store, Variant
from c2a.sources.compliance import CrawlSession


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


def shop_currency(session: CrawlSession) -> str:
    """Store config wins; otherwise Shopify's public /meta.json."""
    if session.store.currency:
        return session.store.currency
    resp = session.get(f"https://{session.store.domain}/meta.json")
    resp.raise_for_status()
    currency = resp.json().get("currency")
    if not currency:
        raise ValueError(f"{session.store.id}: currency unknown; set `currency` in the registry")
    return currency


def crawl(
    session: CrawlSession, max_products: int | None = None, limit: int = 250, max_pages: int = 100
) -> tuple[list[Product], list[str]]:
    """Crawl a Shopify storefront through the compliance session."""
    currency = shop_currency(session)
    products: list[Product] = []
    errors: list[str] = []
    for page in range(1, max_pages + 1):
        resp = session.get(
            f"https://{session.store.domain}/products.json",
            params={"limit": limit, "page": page},
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("products"):
            break
        got, errs = parse_products_json(payload, session.store, currency)
        products += got
        errors += errs
        if max_products is not None and len(products) >= max_products:
            return products[:max_products], errors
    return products, errors
