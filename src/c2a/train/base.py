"""Trainer interface and run configuration shared by all training platforms."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import yaml
from pydantic import BaseModel, Field

Stage = Literal["sft", "rl", "opd", "dpo", "genrm"]
Platform = Literal["tinker", "modal", "ac2"]


class RunConfig(BaseModel):
    name: str
    stage: Stage
    platform: Platform = "tinker"
    base_model: str
    dataset: Path
    log_dir: Path = Path("runs")
    lora_rank: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    batch_size: int = Field(default=32, ge=1)
    steps: int | None = Field(default=None, ge=1)
    save_every: int = Field(default=20, ge=0)
    max_length: int = 8192
    # RL / OPD
    group_size: int = Field(default=8, ge=1)
    max_tokens: int = 512
    temperature: float = 1.0
    advantage: Literal["grpo", "dr_grpo", "rank_grpo"] = "dr_grpo"
    drop_zero_variance_groups: bool = True
    reward: str | None = None  # key in c2a.train.rewards.REWARD_FNS
    fail_penalty: float = -0.5
    teacher_model: str | None = None  # OPD teacher (same tokenizer family as base_model)
    kl_coef: float = 1.0

    @property
    def run_dir(self) -> Path:
        return self.log_dir / self.name


def load_run_config(path: Path | str) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(Path(path).read_text()))


class Trainer(Protocol):
    def submit(self, cfg: RunConfig) -> str:
        """Start a run; return a run id."""
        ...

    def export(self, run_id: str, dest: Path) -> Path:
        """Download final weights / LoRA adapter for serving (vLLM on Modal)."""
        ...
