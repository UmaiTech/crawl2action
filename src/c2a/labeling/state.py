"""Render scraped items into compact System One `state` documents."""

from __future__ import annotations

from typing import Any

from c2a.normalize import currency_exponent, strip_html
from c2a.schemas import Money, Product

MAX_DESCRIPTION_CHARS = 2000
MAX_PAGE_CHARS = 8000


def _money(m: Money) -> str:
    exp = currency_exponent(m.currency)
    return f"{m.amount / 10**exp:.{exp}f} {m.currency}"


def product_state(p: Product) -> dict[str, Any]:
    options: dict[str, set[str]] = {}
    for v in p.variants:
        for k, val in v.options.items():
            options.setdefault(k, set()).add(val)
    state: dict[str, Any] = {
        "title": p.title,
        "brand": p.brand,
        "category": p.category,
        "tags": p.tags,
        "attributes": p.attributes,
        "description": strip_html(p.description)[:MAX_DESCRIPTION_CHARS],
        "price": _money(p.min_price),
        "in_stock": p.in_stock,
        "options": {k: sorted(v) for k, v in options.items()},
        "locale": p.locale,
    }
    return {k: v for k, v in state.items() if v not in (None, "", [], {})}


def page_state(url: str, markdown: str) -> dict[str, Any]:
    return {"url": url, "content": markdown[:MAX_PAGE_CHARS]}


def pair_state(anchor: str | Product, candidate: Product) -> dict[str, Any]:
    anchor_doc = {"query": anchor} if isinstance(anchor, str) else product_state(anchor)
    return {"anchor": anchor_doc, "candidate": product_state(candidate)}
