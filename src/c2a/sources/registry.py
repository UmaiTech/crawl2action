"""Load and validate the store registry."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml
from pydantic import BaseModel

from c2a.schemas import Store, TosStatus

DEFAULT_REGISTRY = Path(__file__).with_name("registry.yaml")


class Registry(BaseModel):
    version: int = 1
    notes: str = ""
    stores: list[Store]

    def by_id(self, store_id: str) -> Store:
        for s in self.stores:
            if s.id == store_id:
                return s
        raise KeyError(store_id)

    def crawlable(self) -> list[Store]:
        """Stores that are both enabled and ToS-approved. Enabled-but-unreviewed never crawls."""
        return [s for s in self.stores if s.enabled and s.tos_status is TosStatus.approved]


def load_registry(path: Path | str | None = None) -> Registry:
    data = yaml.safe_load(Path(path or DEFAULT_REGISTRY).read_text())
    reg = Registry.model_validate(data)
    dupes = [k for k, n in Counter(s.id for s in reg.stores).items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate store ids: {dupes}")
    return reg
