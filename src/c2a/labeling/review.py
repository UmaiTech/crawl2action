"""Human review of escalated labels. Pure functions here; the terminal loop lives in the CLI.

Each decision replaces the label record in labels.jsonl (source='human') and removes the
item from review_queue.jsonl, so progress is saved after every answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from c2a.decide.systemone import Choice, Noul, Score
from c2a.labeling.escalation import apply_review
from c2a.labeling.records import LabelRecord, ReviewItem
from c2a.train.data import read_jsonl, write_jsonl


def load_queue(run_dir: Path) -> list[ReviewItem]:
    path = run_dir / "review_queue.jsonl"
    return list(read_jsonl(path, ReviewItem)) if path.exists() else []


def options(question: Noul | Choice | Score) -> list[str]:
    if isinstance(question, Noul):
        return ["true", "false"]
    if isinstance(question, Score):
        return list(question.criteria)
    return list(question.criteria)


def _key(r: LabelRecord) -> tuple[str, str]:
    return r.item_id, r.question_id


def apply_decision(run_dir: Path, item: ReviewItem, label: str, reviewer: str) -> LabelRecord:
    if label not in options(item.question):
        raise ValueError(f"'{label}' is not an option for {item.record.question_id}")
    decided = apply_review(item.record, label, reviewer)
    labels_path = run_dir / "labels.jsonl"
    records = list(read_jsonl(labels_path, LabelRecord)) if labels_path.exists() else []
    replaced = [decided if _key(r) == _key(decided) else r for r in records]
    if not any(_key(r) == _key(decided) for r in records):
        replaced.append(decided)
    write_jsonl(labels_path, replaced)
    queue = [q for q in load_queue(run_dir) if _key(q.record) != _key(decided)]
    write_jsonl(run_dir / "review_queue.jsonl", queue)
    return decided


def _brief(state: Any) -> dict[str, Any]:
    if isinstance(state, dict):
        keep = ("title", "brand", "price", "category", "description", "query", "url")
        out = {k: state[k] for k in keep if k in state}
        if "description" in out:
            out["description"] = str(out["description"])[:200]
        for side in ("anchor", "candidate"):
            if side in state:
                out[side] = _brief(state[side])
        return out
    return {"text": str(state)[:400]}


def format_item(item: ReviewItem, index: int, total: int) -> str:
    rec, q = item.record, item.question
    top = sorted(rec.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:3]
    lines = [
        f"[{index}/{total}] {rec.item_id} · {rec.qset} · {rec.question_id}",
        json.dumps(_brief(item.state), ensure_ascii=False, indent=2),
        f"Q: {q.instructions or rec.question_id}",
        "Jev: " + ", ".join(f"{k} {v:.2f}" for k, v in top) + f" (confidence {rec.confidence:.2f})",
    ]
    if rec.teacher_value:
        lines.append(f"Teacher: {rec.teacher_value}")
    lines += [f"  {i}) {opt}" for i, opt in enumerate(options(q), 1)]
    lines.append("  s) skip   q) quit")
    return "\n".join(lines)
