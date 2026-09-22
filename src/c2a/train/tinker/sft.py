"""Supervised fine-tuning loop (cold start: grader-filtered teacher distillation)."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from c2a.train.base import RunConfig
from c2a.train.data import SFTExample
from c2a.train.tinker.client import TrainingBackend

log = logging.getLogger(__name__)


def run_sft(backend: TrainingBackend, examples: Sequence[SFTExample], cfg: RunConfig) -> dict:
    n_batches = len(examples) // cfg.batch_size  # drop last partial batch
    if n_batches == 0:
        raise ValueError(f"need at least batch_size={cfg.batch_size} examples, got {len(examples)}")
    steps = min(n_batches, cfg.steps or n_batches)
    history, checkpoints = [], []
    for step in range(steps):
        lr = cfg.learning_rate * max(0.0, 1.0 - step / steps)  # linear decay
        batch = examples[step * cfg.batch_size : (step + 1) * cfg.batch_size]
        metrics = backend.sft_step([ex.messages for ex in batch], lr)
        history.append({"step": step, "lr": lr, **metrics})
        if cfg.save_every and step > 0 and step % cfg.save_every == 0:
            checkpoints.append(backend.save(f"{cfg.name}-{step:06d}"))
    checkpoints.append(backend.save(f"{cfg.name}-final"))
    summary = {"stage": "sft", "steps": steps, "checkpoints": checkpoints, "history": history}
    _write_summary(cfg, summary)
    return summary


def _write_summary(cfg: RunConfig, summary: dict) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
