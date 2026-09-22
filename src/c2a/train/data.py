"""Training record formats (JSONL) for every post-training stage."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

T = TypeVar("T", bound=BaseModel)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class SFTExample(BaseModel):
    id: str
    messages: list[Message] = Field(min_length=2)

    @model_validator(mode="after")
    def _ends_with_assistant(self) -> SFTExample:
        if self.messages[-1].role != "assistant":
            raise ValueError("SFT example must end with an assistant message")
        return self


class PreferencePair(BaseModel):
    """For DPO / reward-model training (human or AI-labelled)."""

    id: str
    prompt: list[Message]
    chosen: str
    rejected: str
    source: Literal["human", "teacher", "implicit"] = "human"


class RLPrompt(BaseModel):
    """RL prompt; `example` carries what the reward function needs (ground truth, candidates)."""

    id: str
    messages: list[Message] = Field(min_length=1)
    reward: str  # key in c2a.train.rewards.REWARD_FNS
    example: dict[str, Any] = Field(default_factory=dict)


class DistillPrompt(BaseModel):
    """On-policy distillation prompt: the student samples, the teacher scores every token."""

    id: str
    messages: list[Message] = Field(min_length=1)


def write_jsonl(path: Path | str, records: Iterable[BaseModel]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path | str, model: type[T]) -> Iterator[T]:
    with Path(path).open() as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield model.model_validate_json(line)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
