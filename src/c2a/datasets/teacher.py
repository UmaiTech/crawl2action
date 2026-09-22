"""Teacher client (Kimi K3 / GLM-5.3 / Qwen3.8-Max via an OpenAI-compatible API). M2.

Responsibilities: batching, on-disk response cache, hard USD budget, rejection sampling
against graders before anything becomes SFT data.
"""

from __future__ import annotations

from c2a import NotYetImplemented


class TeacherClient:
    def __init__(self, model: str, base_url: str, api_key: str, usd_budget: float) -> None:
        self.model, self.base_url, self.usd_budget = model, base_url, usd_budget
        self._api_key = api_key

    def complete(self, messages: list[dict], **kwargs) -> str:
        raise NotYetImplemented("teacher completions", "M2")
