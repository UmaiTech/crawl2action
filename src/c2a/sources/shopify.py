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


COUNTRY_CURRENCY = {"SE": "SEK", "GB": "GBP", "ES": "EUR", "US": "USD", "CA": "CAD"}


def shop_currency(session: CrawlSession) -> str:
    """Registry currency, else Shopify's public /meta.json, else the country's currency
    (recorded as a warning because multi-currency shops may differ)."""
    if session.store.currency:
        return session.store.currency
    meta_url = f"https://{session.store.domain}/meta.json"
    if session.allowed(meta_url):
        resp = session.get(meta_url)
        if resp.status_code == 200:
            try:
                currency = resp.json().get("currency")
            except ValueError:
                currency = None
            if currency:
                return currency
    fallback = COUNTRY_CURRENCY.get(session.store.country)
    if not fallback:
        raise ValueError(f"{session.store.id}: currency unknown; set `currency` in the registry")
    session.warnings.append(f"currency not found; assumed {fallback} from country")
    return fallback


def crawl(
    session: CrawlSession, max_products: int | None = None, limit: int = 250, max_pages: int = 100
) -> tuple[list[Product], list[str]]:
    """Crawl a Shopify storefront through the compliance session.

    Stops on an empty page, a short page (fewer than `limit` products) or a page identical
    to the previous one (some themes ignore `page`).
    """
    currency = shop_currency(session)
    products: list[Product] = []
    errors: list[str] = []
    prev_ids: list | None = None
    for page in range(1, max_pages + 1):
        resp = session.get(
            f"https://{session.store.domain}/products.json",
            params={"limit": limit, "page": page},
        )
        resp.raise_for_status()
        raw = resp.json().get("products") or []
        ids = [p.get("id") for p in raw]
        if not raw or ids == prev_ids:
            break
        prev_ids = ids
        got, errs = parse_products_json({"products": raw}, session.store, currency)
        products += got
        errors += errs
        if max_products is not None and len(products) >= max_products:
            return products[:max_products], errors
        if len(raw) < limit:
            break
    return products, errors
