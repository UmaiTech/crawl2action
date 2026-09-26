"""Crawl orchestration over the registry: default-deny, per-method dispatch, incremental
output under data/raw/{store_id}/ (products.jsonl, hashes.json, manifest.json)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from c2a.decide.base import Decider
from c2a.schemas import Product, Store
from c2a.sources import firecrawl as fc
from c2a.sources import shopify, ucp
from c2a.sources.compliance import CrawlSession, RateLimiter, can_crawl
from c2a.sources.registry import Registry
from c2a.train.data import read_jsonl, write_jsonl

SessionFactory = Callable[[Store], CrawlSession]


def content_hash(p: Product) -> str:
    """Hash of the fields that matter for training; ignores crawl timestamps."""
    data = p.model_dump(mode="json", exclude={"crawled_at"})
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@dataclass
class StoreManifest:
    store_id: str
    method: str
    started_at: str
    finished_at: str = ""
    products_total: int = 0
    products_new_or_changed: int = 0
    products_unchanged: int = 0
    errors: list[str] = field(default_factory=list)
    firecrawl_credits: int = 0
    jev_requests: int = 0
    denied_urls: int = 0
    retries: int = 0
    warnings: list[str] = field(default_factory=list)
    budget_exhausted: bool = False


def plan(registry: Registry, store_ids: Sequence[str] | None = None) -> list[tuple[Store, str]]:
    """(store, reason) for every selected store; reason 'ok' means it will be crawled."""
    stores = [registry.by_id(s) for s in store_ids] if store_ids else registry.stores
    out = []
    for s in stores:
        ok, why = can_crawl(s, f"https://{s.domain}/", _AllowAll())
        out.append((s, "ok" if ok else why))
    return out


class _AllowAll:
    """Robots placeholder for planning (robots is checked per URL during the real crawl)."""

    def can_fetch(self, *_: Any) -> bool:
        return True


def _default_session(store: Store, qps: float) -> CrawlSession:
    return CrawlSession(store, limiter=RateLimiter(qps=qps))


def crawl_one(
    store: Store,
    out_dir: Path,
    session: CrawlSession,
    firecrawl: fc.FirecrawlClient | None = None,
    decider: Decider | None = None,
    max_products: int | None = None,
    seed_queries: Sequence[str] = (),
) -> tuple[StoreManifest, list[Product]]:
    """Crawl one store; return its manifest and the new/changed products."""
    m = StoreManifest(store.id, store.method, datetime.now(UTC).isoformat())
    products: list[Product] = []
    try:
        if store.method == "shopify":
            products, m.errors = shopify.crawl(session, max_products=max_products)
        elif store.method == "ucp":
            profile = ucp.discover(session)
            if profile is None:
                raise ValueError("no UCP REST catalog advertised at /.well-known/ucp")
            queries = list(seed_queries) or store.verticals or ["*"]
            products, m.errors = ucp.iter_catalog(session, profile, queries, max_products)
        elif store.method == "firecrawl":
            if firecrawl is None:
                raise ValueError("Firecrawl client not configured (FIRECRAWL_API_KEY)")
            res = fc.crawl_store(session, firecrawl, decider, max_products=max_products)
            products, m.errors = res.products, res.errors
            m.firecrawl_credits, m.jev_requests = res.credits_spent, res.jev_requests
            m.budget_exhausted = res.budget_exhausted
        else:
            raise ValueError(f"unknown method '{store.method}'")
    except (PermissionError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        m.errors.append(f"{type(exc).__name__}: {exc}")
    m.denied_urls = len(session.denied)
    m.retries = session.retries
    m.warnings = list(session.warnings)
    changed = _write_incremental(out_dir / store.id, products, m)
    m.finished_at = datetime.now(UTC).isoformat()
    (out_dir / store.id).mkdir(parents=True, exist_ok=True)
    (out_dir / store.id / "manifest.json").write_text(json.dumps(asdict(m), indent=2))
    return m, changed


def _write_incremental(store_dir: Path, products: list[Product], m: StoreManifest) -> list[Product]:
    store_dir.mkdir(parents=True, exist_ok=True)
    hashes_path = store_dir / "hashes.json"
    products_path = store_dir / "products.jsonl"
    old_hashes: dict[str, str] = json.loads(hashes_path.read_text()) if hashes_path.exists() else {}
    existing: dict[str, Product] = (
        {p.id: p for p in read_jsonl(products_path, Product)} if products_path.exists() else {}
    )
    changed = []
    for p in products:
        h = content_hash(p)
        if old_hashes.get(p.id) == h and p.id in existing:
            m.products_unchanged += 1
            continue
        old_hashes[p.id] = h
        existing[p.id] = p
        changed.append(p)
    m.products_new_or_changed = len(changed)
    m.products_total = len(existing)
    write_jsonl(products_path, existing.values())
    hashes_path.write_text(json.dumps(old_hashes, indent=0, sort_keys=True))
    return changed


def crawl_registry(
    registry: Registry,
    out_dir: Path,
    store_ids: Sequence[str] | None = None,
    firecrawl: fc.FirecrawlClient | None = None,
    decider: Decider | None = None,
    max_products: int | None = None,
    qps: float = 1.0,
    session_factory: SessionFactory | None = None,
) -> list[tuple[StoreManifest, list[Product]]]:
    results = []
    for store, reason in plan(registry, store_ids):
        if reason != "ok":
            continue
        session = (session_factory or (lambda s: _default_session(s, qps)))(store)
        results.append(crawl_one(store, out_dir, session, firecrawl, decider, max_products))
    return results
