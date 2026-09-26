"""UCP (Universal Commerce Protocol) catalog client, REST binding (spec version 2026-01-01).

Discovery: GET https://<domain>/.well-known/ucp -> profile advertising the shopping service
endpoint and catalog capabilities. Catalog: POST {endpoint}/catalog/search (cursor
pagination) and POST {endpoint}/catalog/lookup. Prices are already integer minor units.
Spec: https://ucp.dev/latest/specification/shopping/catalog/
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from c2a.schemas import Money, Product, Store, Variant
from c2a.sources.compliance import CrawlSession

WELL_KNOWN_PATH = "/.well-known/ucp"
SHOPPING_SERVICE = "dev.ucp.shopping"
CAP_SEARCH = "dev.ucp.shopping.catalog.search"
CAP_LOOKUP = "dev.ucp.shopping.catalog.lookup"


class UcpProfile(BaseModel):
    version: str
    endpoint: str  # REST base URL of the shopping service
    capabilities: list[str]

    @property
    def can_search(self) -> bool:
        return CAP_SEARCH in self.capabilities

    @property
    def can_lookup(self) -> bool:
        return CAP_LOOKUP in self.capabilities


def parse_profile(doc: dict[str, Any]) -> UcpProfile | None:
    """Return the REST shopping profile, or None if the store offers no REST catalog."""
    ucp = doc.get("ucp") or {}
    services = (ucp.get("services") or {}).get(SHOPPING_SERVICE) or []
    rest = next((s for s in services if s.get("transport") == "rest" and s.get("endpoint")), None)
    if rest is None:
        return None
    caps = sorted((ucp.get("capabilities") or {}).keys())
    return UcpProfile(
        version=str(ucp.get("version", "")),
        endpoint=rest["endpoint"].rstrip("/"),
        capabilities=caps,
    )


def discover(session: CrawlSession) -> UcpProfile | None:
    resp = session.get(f"https://{session.store.domain}{WELL_KNOWN_PATH}")
    if resp.status_code != 200:
        return None
    profile = parse_profile(resp.json())
    if profile is not None:
        session.allow_api_host(profile.endpoint)
    return profile


def _money(raw: dict[str, Any] | None) -> Money | None:
    if not raw or "amount" not in raw or "currency" not in raw:
        return None
    return Money(amount=int(raw["amount"]), currency=raw["currency"])


def parse_ucp_product(raw: dict[str, Any], store: Store) -> Product:
    variants = []
    for v in raw.get("variants") or []:
        price = _money(v.get("price"))
        if price is None:
            continue
        variants.append(
            Variant(
                id=str(v["id"]),
                title=v.get("title") or "",
                sku=v.get("sku") or None,
                options={o["name"]: o["label"] for o in v.get("options") or [] if "label" in o},
                price=price,
                compare_at_price=_money(v.get("list_price")),
                available=bool((v.get("availability") or {}).get("available", True)),
            )
        )
    if not variants:  # search results may carry only a price range
        price = _money((raw.get("price_range") or {}).get("min"))
        if price is None:
            raise ValueError(f"product {raw.get('id')} has no priced variant")
        variants = [Variant(id=str(raw["id"]), price=price)]
    categories = raw.get("categories") or []
    merchant_cat = next((c["value"] for c in categories if c.get("taxonomy") == "merchant"), None)
    attributes = {
        f"category_{c['taxonomy']}": str(c["value"])
        for c in categories
        if c.get("taxonomy") and c.get("value") and c.get("taxonomy") != "merchant"
    }
    tags = sorted({t for v in raw.get("variants") or [] for t in v.get("tags") or []})
    return Product(
        id=f"{store.id}:{raw['id']}",
        store_id=store.id,
        native_id=str(raw["id"]),
        title=raw["title"],
        description=(raw.get("description") or {}).get("plain") or "",
        category=merchant_cat,
        tags=tags,
        attributes=attributes,
        url=raw.get("url"),
        image_urls=[m["url"] for m in raw.get("media") or [] if m.get("type") == "image"],
        variants=variants,
        locale=store.locale,
        source="ucp",
        crawled_at=datetime.now(UTC),
    )


def search_catalog(
    session: CrawlSession, profile: UcpProfile, query: str, limit: int = 50, max_pages: int = 20
) -> Iterator[dict[str, Any]]:
    """Yield raw products for one query, following cursor pagination."""
    cursor = None
    for _ in range(max_pages):
        body: dict[str, Any] = {"query": query, "pagination": {"limit": limit}}
        if cursor:
            body["pagination"]["cursor"] = cursor
        resp = session.post(f"{profile.endpoint}/catalog/search", json=body)
        resp.raise_for_status()
        data = resp.json()
        yield from data.get("products") or []
        page = data.get("pagination") or {}
        cursor = page.get("cursor")
        if not page.get("has_next_page") or not cursor:
            return


def iter_catalog(
    session: CrawlSession,
    profile: UcpProfile,
    seed_queries: Iterable[str],
    max_products: int | None = None,
) -> tuple[list[Product], list[str]]:
    """Enumerate a catalog via seed queries (UCP search needs a query), dedupe by id."""
    if not profile.can_search:
        raise ValueError(f"{session.store.id}: UCP profile has no catalog.search capability")
    seen: dict[str, Product] = {}
    errors: list[str] = []
    for q in seed_queries:
        for raw in search_catalog(session, profile, q):
            pid = str(raw.get("id"))
            if pid in seen:
                continue
            try:
                seen[pid] = parse_ucp_product(raw, session.store)
            except (KeyError, ValueError) as exc:
                errors.append(f"{pid}: {exc}")
            if max_products is not None and len(seen) >= max_products:
                return list(seen.values()), errors
    return list(seen.values()), errors
