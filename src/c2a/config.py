"""Settings loaded from YAML with environment-variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class Paths(BaseModel):
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")


class Budgets(BaseModel):
    firecrawl_credits_per_run: int = 5000
    teacher_usd_per_run: float = 100.0
    default_qps_per_domain: float = 1.0


class Models(BaseModel):
    student: str = "Qwen/Qwen3.8-27B"
    student_alt: str = "thinkingmachines/Inkling-Small"
    proxy_student: str = "Qwen/Qwen3.5-4B"
    teacher: str = "kimi-k3"
    decider: str = "llm-semantic-router/Decision-1.0-Lux-9B"
    image: str = "Qwen/Qwen-Image-2.1"
    embedding: str = "Qwen/Qwen3-Embedding-0.6B"


class Labeler(BaseModel):
    """Hosted Jev (TypeSafe System One). API key comes from TYPESAFE_API_KEY."""

    provider: str = "typesafe"
    base_url: str = "https://api.typesafe.ai"
    model: str = "jev-latest"


class Settings(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    budgets: Budgets = Field(default_factory=Budgets)
    models: Models = Field(default_factory=Models)
    labeler: Labeler = Field(default_factory=Labeler)
    locales: list[str] = Field(
        default_factory=lambda: ["sv-SE", "en-GB", "es-ES", "en-US", "en-CA", "fr-CA"]
    )


def _apply_env(data: dict[str, Any], prefix: str = "C2A") -> dict[str, Any]:
    """Override nested keys from env vars, e.g. C2A__MODELS__STUDENT=Qwen/Qwen3.5-27B."""
    for key, value in os.environ.items():
        if not key.startswith(prefix + "__"):
            continue
        parts = key[len(prefix) + 2 :].lower().split("__")
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return data


def load_settings(path: Path | str | None = None) -> Settings:
    path = Path(path) if path else DEFAULT_CONFIG
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
    return Settings.model_validate(_apply_env(data))
