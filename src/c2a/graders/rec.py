"""Recommendation graders: JSON validity, hallucinated IDs, constraints, ranking quality."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from c2a.graders.base import GradeResult, composite
from c2a.graders.ranking import dedupe_preserving_order, ndcg_at_k, recall_at_k
from c2a.schemas import Candidate, Constraints, RecResponse

DEFAULT_WEIGHTS = {"ndcg": 0.6, "recall": 0.2, "constraints": 0.2}


def parse_rec_output(output: str) -> RecResponse | None:
    try:
        return RecResponse.model_validate(json.loads(output))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def hallucinated_ids(ranked: list[str], candidates: list[Candidate]) -> list[str]:
    allowed = {c.product_id for c in candidates}
    return [pid for pid in ranked if pid not in allowed]


def constraint_violations(
    ranked: list[str], candidates: list[Candidate], constraints: Constraints
) -> list[tuple[str, str]]:
    """Return (product_id, reason) for each constraint violation."""
    by_id = {c.product_id: c for c in candidates}
    out = []
    for pid in ranked:
        c = by_id.get(pid)
        if c is None:
            continue
        if constraints.in_stock_only and not c.in_stock:
            out.append((pid, "out of stock"))
        mp = constraints.max_price
        if mp is not None and c.price.currency == mp.currency and c.price.amount > mp.amount:
            out.append((pid, "over max price"))
    return out


class RecGrader:
    """Grades a model's JSON recommendation list against ground-truth relevance.

    Hard failures (score 0, passed=False): invalid JSON/schema or any hallucinated ID.
    """

    name = "rec"

    def __init__(self, k: int = 10, weights: dict[str, float] | None = None) -> None:
        self.k = k
        self.weights = weights or DEFAULT_WEIGHTS

    def grade(self, output: str, example: dict[str, Any]) -> GradeResult:
        candidates = [Candidate.model_validate(c) for c in example["candidates"]]
        constraints = Constraints.model_validate(example.get("constraints", {}))
        relevance: dict[str, float] = example["relevance"]  # product_id -> graded relevance

        parsed = parse_rec_output(output)
        if parsed is None:
            return GradeResult(score=0.0, passed=False, reasons=["invalid JSON or schema"])
        ranked = dedupe_preserving_order([it.product_id for it in parsed.items])
        bad = hallucinated_ids(ranked, candidates)
        if bad:
            return GradeResult(score=0.0, passed=False, reasons=[f"hallucinated ids: {bad}"])

        top = ranked[: self.k]
        violations = constraint_violations(top, candidates, constraints)
        components = {
            "ndcg": ndcg_at_k(top, relevance, self.k),
            "recall": recall_at_k(top, {p for p, r in relevance.items() if r > 0}, self.k),
            "constraints": 1.0 - (len(violations) / len(top) if top else 1.0),
        }
        return GradeResult(
            score=composite(components, self.weights),
            components=components,
            reasons=[f"{pid}: {why}" for pid, why in violations],
        )
