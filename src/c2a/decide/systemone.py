"""System One contract (TypeSafe Jev; also served by Kev and Decision-1.0 models).

POST /v1/systemone with a `state` document and typed questions; every question is answered
with probabilities in one forward pass. This is the canonical decision contract in c2a: the
hosted Jev labeler and our own post-trained decider both speak it.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_OPTIONS = 255


class Noul(BaseModel):
    """Yes/no question. `criteria` optionally describes what true/false mean."""

    type: Literal["noul"] = "noul"
    instructions: str | dict[str, Any] | list[Any] | None = None
    criteria: dict[Literal["true", "false"], str] | None = None


class Choice(BaseModel):
    """Pick one option. `criteria` maps option name -> optional description."""

    type: Literal["choice"] = "choice"
    instructions: str | dict[str, Any] | list[Any] | None = None
    criteria: dict[str, str | None] = Field(min_length=1, max_length=MAX_OPTIONS)


class Score(BaseModel):
    """Ordered rating. `criteria` lists level labels from lowest to highest."""

    type: Literal["score"] = "score"
    instructions: str | dict[str, Any] | list[Any] | None = None
    criteria: list[str] = Field(min_length=1, max_length=MAX_OPTIONS)

    @field_validator("criteria")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("score levels must be unique")
        return v


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    state: str | dict[str, Any] | list[Any]
    model: str = "jev-latest"
    questions: dict[str, Question] = Field(min_length=1)


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0, le=1)  # P(yes)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float  # expected level index
    confidence: float = Field(ge=0, le=1)
    legend: dict[str, str] = Field(default_factory=dict)  # level index -> label
    probabilities: dict[str, float]  # level index -> probability


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float | None = None


def top_label(answer: NoulAnswer | ChoiceAnswer | ScoreAnswer) -> str:
    """Hard label for an answer: 'true'/'false', the chosen option, or the argmax level label."""
    if isinstance(answer, NoulAnswer):
        return "true" if answer.noul >= 0.5 else "false"
    if isinstance(answer, ChoiceAnswer):
        return answer.choice
    idx = max(answer.probabilities, key=answer.probabilities.get)
    return answer.legend.get(idx, idx)


def top_prob(answer: NoulAnswer | ChoiceAnswer | ScoreAnswer) -> float:
    if isinstance(answer, NoulAnswer):
        return max(answer.noul, 1.0 - answer.noul)
    return max(answer.probabilities.values())
