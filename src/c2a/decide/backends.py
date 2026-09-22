"""Decider backends.

- KeywordDecider: deterministic, dependency-free; for tests, the dev gateway and as a floor
  baseline in the decider bake-off.
- OpenAICompatDecider: calls a served decider (vLLM on Modal, OpenAI-compatible) and reads
  per-label log-probs of the first answer token. The exact prompt/format of Decision-1.0
  models must be confirmed from their model card (open item in docs/PLAN.md).
"""

from __future__ import annotations

import re

import httpx

from c2a.decide.base import normalize_probs
from c2a.schemas import DecisionRequest, DecisionResult


class KeywordDecider:
    """Scores each label by word overlap between the label and the evidence+question."""

    def decide(self, request: DecisionRequest) -> DecisionResult:
        words = set(re.findall(r"\w+", f"{request.evidence} {request.question}".lower()))
        scores = {
            label: float(len(set(re.findall(r"\w+", label.lower())) & words))
            for label in request.labels
        }
        return normalize_probs(scores, request.labels)


PROMPT = (
    "Evidence:\n{evidence}\n\nQuestion: {question}\n"
    "Answer with exactly one of these labels: {labels}.\nAnswer:"
)


class OpenAICompatDecider:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 30.0) -> None:
        self.model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    def decide(self, request: DecisionRequest) -> DecisionResult:
        prompt = PROMPT.format(
            evidence=request.evidence, question=request.question, labels=", ".join(request.labels)
        )
        resp = self._client.post(
            "/v1/completions",
            json={
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 1,
                "logprobs": 20,
                "temperature": 0,
            },
        )
        resp.raise_for_status()
        top = resp.json()["choices"][0]["logprobs"]["top_logprobs"][0]  # {token: logprob}
        scores: dict[str, float] = {}
        for label in request.labels:
            # match the label's first token loosely (leading space / case)
            matches = [
                lp
                for tok, lp in top.items()
                if label.lower().startswith(tok.strip().lower()) and tok.strip()
            ]
            if matches:
                scores[label] = max(matches)
        return normalize_probs(scores, request.labels)
