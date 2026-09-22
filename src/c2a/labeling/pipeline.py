"""Label items with hosted Jev: one System One request per item (all questions of the set),
cached, budget-capped, confidence-gated and escalated."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from c2a.decide.base import Decider
from c2a.decide.systemone import (
    ChoiceAnswer,
    NoulAnswer,
    SystemOneRequest,
    SystemOneResponse,
    top_label,
    top_prob,
)
from c2a.labeling.cache import ResponseCache, request_key
from c2a.labeling.escalation import Teacher, escalate, review_item
from c2a.labeling.gating import Thresholds, is_confident
from c2a.labeling.questions import QuestionSet
from c2a.labeling.records import LabelItem, LabelRecord, ReviewItem


@dataclass
class LabelRun:
    records: list[LabelRecord] = field(default_factory=list)
    review: list[ReviewItem] = field(default_factory=list)
    requests_made: int = 0
    cache_hits: int = 0
    skipped: list[str] = field(default_factory=list)  # item ids not labeled (budget)

    @property
    def budget_exhausted(self) -> bool:
        return bool(self.skipped)


def to_records(
    item_id: str, qset: QuestionSet, response: SystemOneResponse, thresholds: Thresholds
) -> list[LabelRecord]:
    out = []
    for qid in qset.questions:
        ans = response.answers.get(qid)
        if ans is None:
            raise ValueError(f"response for {item_id} is missing question '{qid}'")
        if isinstance(ans, NoulAnswer):
            probs = {"true": ans.noul, "false": 1.0 - ans.noul}
            confidence = top_prob(ans)
        else:
            probs = dict(ans.probabilities)
            if not isinstance(ans, ChoiceAnswer):  # score: key by level label
                probs = {ans.legend.get(k, k): v for k, v in probs.items()}
            confidence = ans.confidence
        ok = is_confident(ans, thresholds.for_question(qid))
        out.append(
            LabelRecord(
                item_id=item_id,
                qset=qset.id,
                question_id=qid,
                qtype=ans.type,
                value=top_label(ans),
                probabilities=probs,
                confidence=confidence,
                model=response.model,
                status="accepted" if ok else "escalated",
            )
        )
    return out


def label(
    items: Sequence[LabelItem],
    decider: Decider,
    qset: QuestionSet,
    thresholds: Thresholds,
    cache: ResponseCache | None = None,
    max_requests: int | None = None,
    teacher: Teacher | None = None,
    model: str = "jev-latest",
    max_workers: int = 8,
) -> LabelRun:
    run = LabelRun()
    requests = {
        it.id: SystemOneRequest(state=it.state, model=model, questions=qset.questions)
        for it in items
    }

    # Resolve cache hits first, then spend the budget on misses in input order.
    responses: dict[str, SystemOneResponse] = {}
    misses: list[str] = []
    for it in items:
        cached = cache.get(request_key(requests[it.id], model)) if cache else None
        if cached is not None:
            responses[it.id] = cached
            run.cache_hits += 1
        else:
            misses.append(it.id)
    if max_requests is not None and len(misses) > max_requests:
        run.skipped = misses[max_requests:]
        misses = misses[:max_requests]

    def fetch(item_id: str) -> tuple[str, SystemOneResponse]:
        resp = decider.ask(requests[item_id])
        if cache:
            cache.put(request_key(requests[item_id], model), resp)
        return item_id, resp

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for item_id, resp in pool.map(fetch, misses):
            responses[item_id] = resp
            run.requests_made += 1

    for it in items:
        if it.id not in responses:
            continue
        for rec in to_records(it.id, qset, responses[it.id], thresholds):
            if rec.status == "escalated":
                question = qset.questions[rec.question_id]
                rec = escalate(rec, it.state, question, teacher)
                if rec.status == "review":
                    run.review.append(review_item(rec, it.state, question))
            run.records.append(rec)
    return run
