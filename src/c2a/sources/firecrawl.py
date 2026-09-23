"""Firecrawl v2 ingestion (tier B): /map -> compliance filter -> URL heuristic ->
Jev page gate (markdown scrape) -> JSON extract for product pages only.

Uses the REST API directly (httpx) so it is easy to mock and has no SDK dependency.
Every URL goes through `CrawlSession.allowed` (ToS + robots + domain) before Firecrawl
fetches it; Firecrawl's own request rate is bounded by the session's rate limiter.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
from pydantic import BaseModel, Field

from c2a.decide.base import Decider
from c2a.labeling.gating import Thresholds, load_thresholds
from c2a.labeling.pipeline import label
from c2a.labeling.questions import page_v1
from c2a.labeling.records import LabelItem
from c2a.labeling.state import page_state
from c2a.normalize import to_minor_units
from c2a.schemas import Money, Product, Store, Variant
from c2a.sources.compliance import CrawlSession

FIRECRAWL_BASE_URL = "https://api.firecrawl.dev"
RETRY_STATUS = {429, 500, 502, 503, 504}
# Approximate credit costs (Firecrawl pricing); used for budgeting, not billing.
CREDITS = {"map": 1, "scrape_markdown": 1, "scrape_json": 5, "search": 2}


class BudgetExceeded(RuntimeError):
    pass


class FirecrawlClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = FIRECRAWL_BASE_URL,
        credit_budget: int = 5000,
        timeout: float = 60.0,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        api_key = api_key if api_key is not None else os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            raise RuntimeError("FIRECRAWL_API_KEY is not set")
        self.credit_budget = credit_budget
        self.credits_spent = 0
        self.max_retries = max_retries
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def _spend(self, op: str) -> None:
        cost = CREDITS[op]
        if self.credits_spent + cost > self.credit_budget:
            raise BudgetExceeded(f"Firecrawl credit budget {self.credit_budget} exhausted")
        self.credits_spent += cost

    def _post(self, path: str, body: dict[str, Any], op: str) -> dict[str, Any]:
        self._spend(op)
        for attempt in range(self.max_retries + 1):
            resp = self._client.post(path, json=body)
            if resp.status_code not in RETRY_STATUS or attempt == self.max_retries:
                resp.raise_for_status()
                data = resp.json()
                if data.get("success") is False:
                    raise RuntimeError(f"firecrawl {path} failed: {data.get('error')}")
                return data
            self._sleep(2**attempt)
        raise AssertionError("unreachable")

    def map(self, url: str, limit: int = 5000) -> list[str]:
        data = self._post(
            "/v2/map",
            {"url": url, "limit": limit, "sitemap": "include", "ignoreQueryParameters": True},
            "map",
        )
        links = data.get("links") or []
        return [link["url"] if isinstance(link, dict) else link for link in links]

    def scrape_markdown(self, url: str) -> str:
        data = self._post(
            "/v2/scrape",
            {"url": url, "formats": ["markdown"], "onlyMainContent": True},
            "scrape_markdown",
        )
        return (data.get("data") or {}).get("markdown") or ""

    def scrape_json(self, url: str, schema: dict[str, Any]) -> dict[str, Any] | None:
        data = self._post(
            "/v2/scrape",
            {"url": url, "formats": [{"type": "json", "schema": schema}], "onlyMainContent": True},
            "scrape_json",
        )
        return (data.get("data") or {}).get("json")

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Web search results as [{url, title, description}]. Accepts both documented shapes
        (data.web[] and data[]) since the exact v2 shape is unconfirmed here."""
        data = self._post("/v2/search", {"query": query, "limit": limit}, "search")
        payload = data.get("data")
        if isinstance(payload, dict):
            payload = payload.get("web") or []
        return [r for r in payload or [] if isinstance(r, dict) and r.get("url")]


# ---------- extraction schema ----------


class ExtractedVariant(BaseModel):
    title: str = ""
    sku: str | None = None
    price: str | float | None = None
    available: bool | None = None
    options: dict[str, str] = Field(default_factory=dict)


class ExtractedProduct(BaseModel):
    """What we ask Firecrawl JSON mode to extract from a product page."""

    title: str
    description: str = ""
    brand: str | None = None
    currency: str | None = Field(default=None, description="ISO 4217 code, e.g. SEK, GBP")
    price: str | float | None = Field(default=None, description="Current price as shown")
    images: list[str] = Field(default_factory=list)
    variants: list[ExtractedVariant] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)
    breadcrumbs: list[str] = Field(default_factory=list)
    sku: str | None = None


_NUM_RE = re.compile(r"\d[\d\s.,]*")


def parse_price(text: str | float | None) -> Decimal | None:
    """'1 299,00 kr' -> 1299.00 ; '£19.99' -> 19.99 ; '1,299.50' -> 1299.50."""
    if text is None:
        return None
    if isinstance(text, int | float):
        return Decimal(str(text))
    m = _NUM_RE.search(text)
    if not m:
        return None
    s = m.group().replace(" ", "").replace(" ", "").rstrip(".,")
    if "," in s and "." in s:
        s = (
            s.replace(",", "")
            if s.rfind(".") > s.rfind(",")
            else s.replace(".", "").replace(",", ".")
        )
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = f"{head.replace(',', '')}.{tail}" if len(tail) in (1, 2) else s.replace(",", "")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def to_product(ex: ExtractedProduct, url: str, store: Store) -> Product:
    currency = (ex.currency or store.currency or "").upper()
    if len(currency) != 3:
        raise ValueError("currency unknown")
    variants = []
    for i, v in enumerate(
        ex.variants or [ExtractedVariant(title=ex.title, sku=ex.sku, price=ex.price)]
    ):
        amount = parse_price(v.price if v.price is not None else ex.price)
        if amount is None:
            continue
        variants.append(
            Variant(
                id=v.sku or f"v{i}",
                title=v.title,
                sku=v.sku,
                options=v.options,
                price=Money(amount=to_minor_units(amount, currency), currency=currency),
                available=True if v.available is None else v.available,
            )
        )
    if not variants:
        raise ValueError("no parsable price")
    native_id = ex.sku or re.sub(r"[^\w-]+", "-", url.split("//", 1)[-1].split("/", 1)[-1]).strip(
        "-"
    )
    return Product(
        id=f"{store.id}:{native_id}",
        store_id=store.id,
        native_id=native_id,
        title=ex.title,
        description=ex.description,
        brand=ex.brand,
        category=" > ".join(ex.breadcrumbs) or None,
        attributes=ex.attributes,
        url=url,
        image_urls=ex.images,
        variants=variants,
        locale=store.locale,
        source="firecrawl",
        crawled_at=datetime.now(UTC),
    )


# ---------- URL triage ----------

PRODUCT_URL_RE = re.compile(r"/products?/|/p/|/dp/|/item/|-p-\d|/produkt/|/producto/", re.I)
SKIP_URL_RE = re.compile(
    r"/(blog|news|journal|cart|checkout|account|login|register|search|help|faq|pages|policies|"
    r"collections|category|categories|kategori|stores?|careers|about|contact|kontakt)(/|$|\?)"
    r"|\.(jpg|jpeg|png|gif|webp|svg|pdf|xml|css|js)$",
    re.I,
)


def triage_url(url: str) -> str:
    """product | skip | ambiguous (ambiguous pages go through the Jev page gate)."""
    if PRODUCT_URL_RE.search(url):
        return "product"
    if SKIP_URL_RE.search(url):
        return "skip"
    return "ambiguous"


@dataclass
class FirecrawlCrawlResult:
    products: list[Product] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    urls_mapped: int = 0
    urls_denied: int = 0
    triage: dict[str, int] = field(default_factory=dict)
    gated_out: int = 0
    jev_requests: int = 0
    credits_spent: int = 0
    budget_exhausted: bool = False


def crawl_store(
    session: CrawlSession,
    client: FirecrawlClient,
    decider: Decider | None,
    max_products: int | None = None,
    max_urls: int = 5000,
    thresholds: Thresholds | None = None,
) -> FirecrawlCrawlResult:
    """Without a decider, ambiguous URLs are skipped (never extracted blind)."""
    res = FirecrawlCrawlResult()
    start_credits = client.credits_spent
    schema = ExtractedProduct.model_json_schema()
    thresholds = thresholds or load_thresholds()
    try:
        urls = client.map(f"https://{session.store.domain}", limit=max_urls)
        res.urls_mapped = len(urls)
        allowed = [u for u in dict.fromkeys(urls) if session.allowed(u)]
        res.urls_denied = res.urls_mapped - len(allowed)
        buckets: dict[str, list[str]] = {"product": [], "skip": [], "ambiguous": []}
        for u in allowed:
            buckets[triage_url(u)].append(u)
        res.triage = {k: len(v) for k, v in buckets.items()}

        to_extract = list(buckets["product"])
        if decider is not None and buckets["ambiguous"]:
            items = []
            for u in buckets["ambiguous"]:
                session.limiter.wait(session.store.domain)
                items.append(LabelItem(id=u, state=page_state(u, client.scrape_markdown(u))))
            run = label(items, decider, page_v1(), thresholds)
            res.jev_requests = run.requests_made
            product_pages = {
                r.item_id
                for r in run.records
                if r.question_id == "page_type"
                and r.value == "product_detail"
                and r.status == "accepted"
            }
            res.gated_out = len(items) - len(product_pages)
            to_extract += [u for u in buckets["ambiguous"] if u in product_pages]

        for u in to_extract:
            if max_products is not None and len(res.products) >= max_products:
                break
            session.limiter.wait(session.store.domain)
            raw = client.scrape_json(u, schema)
            if not raw:
                res.errors.append(f"{u}: empty extraction")
                continue
            try:
                res.products.append(
                    to_product(ExtractedProduct.model_validate(raw), u, session.store)
                )
            except ValueError as exc:
                res.errors.append(f"{u}: {exc}")
    except BudgetExceeded as exc:
        res.budget_exhausted = True
        res.errors.append(str(exc))
    res.credits_spent = client.credits_spent - start_credits
    return res
