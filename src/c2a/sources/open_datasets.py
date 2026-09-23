"""Loaders for open behavioural datasets, reading files the user has downloaded locally.

Licenses must be checked before use (open item in docs/PLAN.md). Parquet needs the `data`
extra (pyarrow), imported lazily.

- ESCI / Shopping Queries (Amazon): query-product relevance judged by humans
  (Exact/Substitute/Complement/Irrelevant) -> pair_v1 labels (gold + decider training).
- Amazon-M2 (KDD Cup 2023): multilingual sessions (incl. UK, ES) -> next-item ground truth.
- Amazon Reviews 2023: reviews + per-user purchase histories.
- H&M, RetailRocket, Diginetica: M2.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from c2a import NotYetImplemented
from c2a.labeling.records import LabelItem, LabelRecord
from c2a.schemas import Review

DATASETS = (
    "esci",
    "amazon_m2",
    "amazon_reviews_2023",
    "hm_fashion",
    "retailrocket",
    "diginetica",
)
ESCI_TO_PAIR = {"E": "exact", "S": "substitute", "C": "complement", "I": "irrelevant"}
ESCI_LOCALE = {"us": "en-US", "es": "es-ES", "jp": "ja-JP"}
M2_LOCALE = {
    "UK": "en-GB",
    "DE": "de-DE",
    "JP": "ja-JP",
    "ES": "es-ES",
    "FR": "fr-FR",
    "IT": "it-IT",
}


def _read_parquet(path: Path | str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    return pq.read_table(path).to_pylist()


# ---------- ESCI ----------


class EsciExample(BaseModel):
    query_id: str
    query: str
    product_id: str
    locale: str
    label: str  # pair_v1 value
    split: str
    product: dict[str, Any]


def load_esci(
    examples_path: Path | str,
    products_path: Path | str,
    locales: Iterable[str] | None = None,
    small_version_only: bool = True,
) -> Iterator[EsciExample]:
    wanted = set(locales) if locales else None
    products = {(r["product_id"], r["product_locale"]): r for r in _read_parquet(products_path)}
    for r in _read_parquet(examples_path):
        if small_version_only and not r.get("small_version", 1):
            continue
        loc = r["product_locale"]
        if wanted and loc not in wanted:
            continue
        prod = products.get((r["product_id"], loc), {})
        yield EsciExample(
            query_id=str(r["query_id"]),
            query=r["query"],
            product_id=r["product_id"],
            locale=ESCI_LOCALE.get(loc, loc),
            label=ESCI_TO_PAIR[r["esci_label"]],
            split=r.get("split", "train"),
            product={
                k.removeprefix("product_"): v
                for k, v in prod.items()
                if k
                in (
                    "product_title",
                    "product_brand",
                    "product_color",
                    "product_description",
                    "product_bullet_point",
                )
                and v
            },
        )


def esci_to_labels(examples: Iterable[EsciExample]) -> tuple[list[LabelItem], list[LabelRecord]]:
    """ESCI judgements are human labels -> pair@1 records with source='human'."""
    items, records = [], []
    for ex in examples:
        item_id = f"esci:{ex.locale}:{ex.query_id}:{ex.product_id}"
        items.append(
            LabelItem(id=item_id, state={"anchor": {"query": ex.query}, "candidate": ex.product})
        )
        records.append(
            LabelRecord(
                item_id=item_id,
                qset="pair@1",
                question_id="relation",
                qtype="choice",
                value=ex.label,
                probabilities={ex.label: 1.0},
                confidence=1.0,
                source="human",
                model="esci",
                status="accepted",
                reviewer="amazon-esci",
            )
        )
    return items, records


# ---------- Amazon-M2 ----------


class Session(BaseModel):
    prev_items: list[str]
    next_item: str | None
    locale: str


_ITEM_RE = re.compile(r"'([^']+)'")


def load_amazon_m2(
    sessions_csv: Path | str, locales: Iterable[str] | None = None
) -> Iterator[Session]:
    """sessions_*.csv columns: prev_items ("['A' 'B']"), next_item (train only), locale."""
    wanted = set(locales) if locales else None
    with Path(sessions_csv).open(newline="") as f:
        for row in csv.DictReader(f):
            if wanted and row["locale"] not in wanted:
                continue
            yield Session(
                prev_items=_ITEM_RE.findall(row["prev_items"]),
                next_item=row.get("next_item") or None,
                locale=M2_LOCALE.get(row["locale"], row["locale"]),
            )


# ---------- Amazon Reviews 2023 ----------


def load_amazon_reviews(reviews_jsonl: Path | str) -> Iterator[Review]:
    with Path(reviews_jsonl).open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            yield Review(
                product_id=f"amazon:{r.get('parent_asin') or r['asin']}",
                rating=r.get("rating"),
                text=" ".join(x for x in (r.get("title"), r.get("text")) if x),
                locale="en-US",
            )


def user_histories(reviews_jsonl: Path | str, verified_only: bool = True) -> dict[str, list[str]]:
    """user_id -> parent_asins in time order (a proxy for purchase sequences)."""
    events: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with Path(reviews_jsonl).open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if verified_only and not r.get("verified_purchase"):
                continue
            events[r["user_id"]].append(
                (int(r.get("timestamp", 0)), r.get("parent_asin") or r["asin"])
            )
    return {u: [a for _, a in sorted(ev)] for u, ev in events.items()}


def load(name: str, split: str = "train"):
    if name not in DATASETS:
        raise KeyError(name)
    raise NotYetImplemented(
        f"open dataset loader '{name}' (use the specific load_* function)", "M2"
    )
