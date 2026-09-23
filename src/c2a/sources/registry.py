"""Load, validate, update and save the store registry.

Approval (tos_status + enabled) is a human-only action via `approve`/`deny`; automated code
(discovery, fingerprinting) may only update descriptive fields such as platform/method.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel

from c2a.schemas import Store, TosStatus

DEFAULT_REGISTRY = Path(__file__).with_name("registry.yaml")
AUTOMATABLE_FIELDS = frozenset({"platform", "method", "currency"})


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

    def approve(self, store_id: str, by: str, note: str = "") -> Store:
        if not by.strip():
            raise ValueError("approval requires the reviewer's name (--by)")
        s = self.by_id(store_id)
        s.tos_status, s.enabled = TosStatus.approved, True
        s.reviewed_by, s.reviewed_at, s.review_note = by, datetime.now(UTC), note or None
        return s

    def deny(self, store_id: str, by: str, note: str = "") -> Store:
        if not by.strip():
            raise ValueError("denial requires the reviewer's name (--by)")
        s = self.by_id(store_id)
        s.tos_status, s.enabled = TosStatus.denied, False
        s.reviewed_by, s.reviewed_at, s.review_note = by, datetime.now(UTC), note or None
        return s

    def update_descriptive(self, store_id: str, **fields: str) -> Store:
        """Automated updates (fingerprinting). Refuses anything that affects crawl permission."""
        bad = set(fields) - AUTOMATABLE_FIELDS
        if bad:
            raise ValueError(f"not automatable: {sorted(bad)}")
        s = self.by_id(store_id)
        for k, v in fields.items():
            setattr(s, k, v)
        return s


def load_registry(path: Path | str | None = None) -> Registry:
    data = yaml.safe_load(Path(path or DEFAULT_REGISTRY).read_text())
    reg = Registry.model_validate(data)
    dupes = [k for k, n in Counter(s.id for s in reg.stores).items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate store ids: {dupes}")
    return reg


def save_registry(reg: Registry, path: Path | str | None = None) -> Path:
    path = Path(path or DEFAULT_REGISTRY)
    data = reg.model_dump(mode="json", exclude_none=True)
    header = "# Store registry for crawl2action. See docs/PLAN.md section 1.\n"
    path.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return path
