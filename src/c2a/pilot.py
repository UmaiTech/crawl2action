"""One-command pilot: preflight -> crawl -> label products -> label pairs -> report."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from c2a.config import Settings
from c2a.decide.base import Decider
from c2a.labeling.cache import ResponseCache
from c2a.labeling.gating import load_thresholds
from c2a.labeling.pairs import candidate_pairs, pair_items
from c2a.labeling.pipeline import label, save_run
from c2a.labeling.questions import pair_v1, product_v1
from c2a.labeling.records import LabelItem
from c2a.labeling.state import product_state
from c2a.report import build_report
from c2a.schemas import Product
from c2a.sources.crawl import SessionFactory, crawl_registry
from c2a.sources.firecrawl import FirecrawlClient
from c2a.sources.registry import Registry
from c2a.train.data import read_jsonl


@dataclass
class PilotResult:
    crawled: dict[str, int] = field(default_factory=dict)  # store -> new/changed products
    product_labels: int = 0
    pair_labels: int = 0
    to_review: int = 0
    jev_requests: int = 0
    skipped_budget: int = 0
    report: str = ""


def run_pilot(
    registry: Registry,
    settings: Settings,
    store_ids: Sequence[str],
    decider: Decider,
    data_dir: Path = Path("data"),
    firecrawl: FirecrawlClient | None = None,
    max_products: int | None = 200,
    pairs_k: int = 3,
    max_jev_requests: int = 500,
    session_factory: SessionFactory | None = None,
) -> PilotResult:
    not_ok = [s for s in store_ids if s not in {x.id for x in registry.crawlable()}]
    if not_ok:
        raise ValueError(f"not approved/enabled: {not_ok} (c2a registry approve <id> --by ...)")
    raw_dir, labels_dir = data_dir / "raw", data_dir / "labels"
    res = PilotResult()
    thresholds = load_thresholds()
    cache = ResponseCache(labels_dir / "cache")
    model = settings.labeler.model

    results = crawl_registry(
        registry,
        raw_dir,
        store_ids,
        firecrawl=firecrawl,
        decider=decider,
        max_products=max_products,
        qps=settings.budgets.default_qps_per_domain,
        session_factory=session_factory,
    )
    changed: list[Product] = []
    for m, ch in results:
        res.crawled[m.store_id] = m.products_new_or_changed
        changed += ch

    budget = max_jev_requests
    if changed:
        items = [LabelItem(id=p.id, state=product_state(p)) for p in changed]
        run = label(items, decider, product_v1(), thresholds, cache, budget, model=model)
        save_run(labels_dir / "product_v1", items, run)
        budget -= run.requests_made
        res.jev_requests += run.requests_made
        res.product_labels = len(run.records)
        res.to_review += len(run.review)
        res.skipped_budget += len(run.skipped)

    if pairs_k > 0 and budget > 0:
        by_store: dict[str, list[Product]] = defaultdict(list)
        for sid in store_ids:
            path = raw_dir / sid / "products.jsonl"
            if path.exists():
                by_store[sid] = list(read_jsonl(path, Product))
        items = [
            it
            for prods in by_store.values()
            for it in pair_items(candidate_pairs(prods, k=pairs_k))
        ]
        if items:
            run = label(items, decider, pair_v1(), thresholds, cache, budget, model=model)
            save_run(labels_dir / "pair_v1", items, run)
            res.jev_requests += run.requests_made
            res.pair_labels = len(run.records)
            res.to_review += len(run.review)
            res.skipped_budget += len(run.skipped)

    res.report = build_report(raw_dir, labels_dir)
    report_path = data_dir / "reports" / "pilot.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(res.report)
    return res
