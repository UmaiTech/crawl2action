"""Escalation of low-confidence labels: teacher first, then the human review queue."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from c2a.decide.systemone import Question
from c2a.labeling.records import LabelRecord, ReviewItem

# (state, question_id, question) -> label string, or None if the teacher cannot answer.
Teacher = Callable[[Any, str, Question], str | None]


def escalate(
    record: LabelRecord, state: Any, question: Question, teacher: Teacher | None
) -> LabelRecord:
    """Return the updated record: accepted (teacher agrees) or queued for human review."""
    teacher_label = teacher(state, record.question_id, question) if teacher else None
    if teacher_label is not None and teacher_label == record.value:
        return record.model_copy(update={"status": "accepted", "source": "jev+teacher"})
    return record.model_copy(update={"status": "review", "teacher_value": teacher_label})


def review_item(record: LabelRecord, state: Any, question: Question) -> ReviewItem:
    return ReviewItem(record=record, state=state, question=question)


def apply_review(record: LabelRecord, label: str, reviewer: str) -> LabelRecord:
    return record.model_copy(
        update={"value": label, "status": "accepted", "source": "human", "reviewer": reviewer}
    )


def append_review_queue(path: Path | str, items: list[ReviewItem]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")
