"""Confidence gating: decide which Jev answers are accepted automatically."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from c2a.config import REPO_ROOT
from c2a.decide.systemone import ChoiceAnswer, NoulAnswer, ScoreAnswer, top_prob

DEFAULT_THRESHOLDS = REPO_ROOT / "configs" / "labeling.yaml"


class Thresholds(BaseModel):
    """choice/score: accept if confidence >= min_confidence AND top prob >= min_top_prob.
    noul: accept if max(p, 1 - p) >= noul_min."""

    min_confidence: float = Field(default=0.6, ge=0, le=1)
    min_top_prob: float = Field(default=0.7, ge=0, le=1)
    noul_min: float = Field(default=0.85, ge=0.5, le=1)
    per_question: dict[str, dict[str, float]] = Field(default_factory=dict)

    def for_question(self, qid: str) -> Thresholds:
        override = self.per_question.get(qid)
        return self.model_copy(update=override) if override else self


def load_thresholds(path: Path | str | None = None) -> Thresholds:
    path = Path(path or DEFAULT_THRESHOLDS)
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    return Thresholds.model_validate((data or {}).get("thresholds", {}))


def is_confident(answer: NoulAnswer | ChoiceAnswer | ScoreAnswer, t: Thresholds) -> bool:
    if isinstance(answer, NoulAnswer):
        return top_prob(answer) >= t.noul_min
    return answer.confidence >= t.min_confidence and top_prob(answer) >= t.min_top_prob
