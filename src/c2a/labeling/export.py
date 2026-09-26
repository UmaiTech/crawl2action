"""Exports: decider training data (Kev / System One training JSONL) and C2A-Bench gold."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from c2a.bench import contains_canary
from c2a.decide.systemone import Choice, Noul, Score
from c2a.labeling.questions import QuestionSet
from c2a.labeling.records import LabelRecord

# Jev-only labels are excluded by default: TypeSafe's terms on training from Jev outputs must
# be confirmed first (open item in docs/PLAN.md).
DEFAULT_TRAIN_SOURCES = frozenset({"human", "jev+teacher"})


def _kev_label(question: Noul | Choice | Score, value: str) -> Any:
    if isinstance(question, Noul):
        return value == "true"
    if isinstance(question, Score):
        return question.criteria.index(value)
    return value


def to_decider_train(
    states: Mapping[str, Any],
    records: Iterable[LabelRecord],
    qset: QuestionSet,
    sources: Iterable[str] = DEFAULT_TRAIN_SOURCES,
) -> list[dict[str, Any]]:
    """One row per item: {"state", "questions": {qid: {type, instructions, criteria, label}}}."""
    sources = set(sources)
    by_item: dict[str, dict[str, LabelRecord]] = defaultdict(dict)
    for r in records:
        if r.qset == qset.id and r.status == "accepted" and r.source in sources:
            by_item[r.item_id][r.question_id] = r
    rows = []
    for item_id, recs in by_item.items():
        state = states[item_id]
        if contains_canary(json.dumps(state, default=str)):
            continue
        questions = {}
        for qid, rec in recs.items():
            q = qset.questions[qid]
            questions[qid] = {
                **q.model_dump(mode="json", exclude_none=True),
                "label": _kev_label(q, rec.value),
            }
        rows.append({"state": state, "questions": questions})
    return rows


def to_gold(records: Iterable[LabelRecord]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": r.item_id,
            "qset": r.qset,
            "question_id": r.question_id,
            "label": r.value,
            "reviewer": r.reviewer,
        }
        for r in records
        if r.source == "human" and r.status == "accepted"
    ]
