"""Markdown summary of a crawl + label run, built from files on disk."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from c2a.labeling.records import LabelItem, LabelRecord
from c2a.train.data import read_jsonl

STORE_HEADER = [
    "| store | method | products | new/changed | errors | retries | warnings |",
    "|---|---|---:|---:|---:|---:|---|",
]
LABEL_HEADER = [
    "| question | labels | accepted | human | mean confidence | top accepted values |",
    "|---|---:|---:|---:|---:|---|",
]


def _manifests(raw_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(raw_dir.glob("*/manifest.json"))]


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _store_rows(manifests: list[dict]) -> list[str]:
    rows = []
    for m in manifests:
        warnings = "; ".join(m.get("warnings", [])) or "-"
        rows.append(
            f"| {m['store_id']} | {m['method']} | {m['products_total']} | "
            f"{m['products_new_or_changed']} | {len(m['errors'])} | {m.get('retries', 0)} | "
            f"{warnings} |"
        )
    return rows or ["| (no crawls yet) | | | | | | |"]


def _question_row(qid: str, rs: list[LabelRecord]) -> str:
    acc = sum(r.status == "accepted" for r in rs) / len(rs)
    human = sum(r.source == "human" for r in rs)
    conf = sum(r.confidence for r in rs) / len(rs)
    values = Counter(r.value for r in rs if r.status == "accepted").most_common(4)
    top = ", ".join(f"{v} {n}" for v, n in values) or "-"
    return f"| {qid} | {len(rs)} | {acc:.0%} | {human} | {conf:.2f} | {top} |"


def _title(state: Any, fallback: str) -> str:
    if not isinstance(state, dict):
        return fallback
    if "title" in state:
        return str(state["title"])
    if "candidate" in state:
        anchor = state["anchor"].get("title") or state["anchor"].get("query")
        return f"{anchor} ↔ {state['candidate'].get('title')}"
    return fallback


def _samples(run: Path, records: list[LabelRecord], n: int) -> list[str]:
    items_path = run / "items.jsonl"
    if not items_path.exists() or n <= 0:
        return []
    accepted: dict[str, dict[str, str]] = defaultdict(dict)
    for r in records:
        if r.status == "accepted":
            accepted[r.item_id][r.question_id] = r.value
    out = ["", "Samples:"]
    for it in read_jsonl(items_path, LabelItem):
        if it.id in accepted:
            labels = ", ".join(f"{k}={v}" for k, v in accepted[it.id].items())
            out.append(f"- {_title(it.state, it.id)}: {labels}")
            if len(out) - 2 >= n:
                break
    return out


def build_report(raw_dir: Path, labels_dir: Path, samples: int = 5) -> str:
    manifests = _manifests(raw_dir)
    out = ["# crawl2action pilot report", "", "## Stores", "", *STORE_HEADER]
    out += _store_rows(manifests)
    errors = [(m["store_id"], e) for m in manifests for e in m["errors"]][:10]
    if errors:
        out += ["", "First errors:"] + [f"- `{s}`: {e}" for s, e in errors]

    cache_dir = labels_dir / "cache"
    paid = sum(1 for _ in cache_dir.rglob("*.json")) if cache_dir.exists() else 0
    credits = sum(m.get("firecrawl_credits", 0) for m in manifests)
    out += [
        "",
        "## Cost",
        "",
        f"- Jev requests paid so far (cached responses): {paid}",
        f"- Firecrawl credits in the latest crawls: {credits}",
    ]

    for run in sorted(p.parent for p in labels_dir.glob("*/labels.jsonl")):
        records = list(read_jsonl(run / "labels.jsonl", LabelRecord))
        in_queue = _count_lines(run / "review_queue.jsonl")
        n_items = len({r.item_id for r in records})
        out += [
            "",
            f"## Labels: {run.name}",
            "",
            f"{n_items} items, {len(records)} labels, {in_queue} waiting for review.",
            "",
            *LABEL_HEADER,
        ]
        by_q: dict[str, list[LabelRecord]] = defaultdict(list)
        for r in records:
            by_q[r.question_id].append(r)
        out += [_question_row(q, rs) for q, rs in sorted(by_q.items())]
        out += _samples(run, records, samples)
    return "\n".join(out) + "\n"
