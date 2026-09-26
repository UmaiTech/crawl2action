"""Task builders, one per family (see docs/PLAN.md section 2). Implemented in M2."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol

from c2a.schemas import Product, TaskExample

TASK_FAMILIES = ("rec_pre", "rec_post", "copy", "localize", "image_brief", "decide", "agentic")


class TaskBuilder(Protocol):
    task: str

    def build(self, products: Iterable[Product]) -> Iterator[TaskExample]: ...
