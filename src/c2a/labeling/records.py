"""Label records written by the labeling pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from c2a.decide.systemone import Question

Source = Literal["jev", "jev+teacher", "human"]
Status = Literal["accepted", "escalated", "review"]


class LabelItem(BaseModel):
    id: str
    state: str | dict[str, Any] | list[Any]


class LabelRecord(BaseModel):
    item_id: str
    qset: str  # e.g. "product@1"
    question_id: str
    qtype: Literal["noul", "choice", "score"]
    value: str  # choice option / "true"|"false" / score level label
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float
    source: Source = "jev"
    model: str
    status: Status
    teacher_value: str | None = None
    reviewer: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReviewItem(BaseModel):
    """Everything a human reviewer needs, self-contained."""

    record: LabelRecord
    state: str | dict[str, Any] | list[Any]
    question: Question
