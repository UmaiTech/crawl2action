"""Leakage-safe splits: hold out whole stores AND a time window.

An example goes to `test` if its store is held out OR it was created at/after the cutoff.
Held-out stores never appear in train, so benchmark items test generalization to unseen
catalogs; the time cutoff guards against temporal leakage.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime

from c2a.schemas import TaskExample


def pick_holdout_stores(store_ids: Iterable[str], fraction: float, seed: str = "c2a") -> set[str]:
    """Deterministic hash-based store holdout."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    out = set()
    for sid in sorted(set(store_ids)):
        h = int(hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if h < fraction:
            out.add(sid)
    return out


def split(
    examples: Iterable[TaskExample], holdout_stores: set[str], time_cutoff: datetime | None
) -> dict[str, list[TaskExample]]:
    out: dict[str, list[TaskExample]] = {"train": [], "test": []}
    for ex in examples:
        held_store = ex.store_id is not None and ex.store_id in holdout_stores
        held_time = (
            time_cutoff is not None and ex.created_at is not None and ex.created_at >= time_cutoff
        )
        out["test" if (held_store or held_time) else "train"].append(ex)
    return out
