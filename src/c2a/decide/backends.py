"""Decider backends, all speaking the System One contract.

- SystemOneClient: HTTP client for POST /v1/systemone. Defaults to TypeSafe's hosted Jev;
  the same client points at our own post-trained decider once it is served on Modal.
- KeywordDecider: deterministic, dependency-free; for tests, dry runs and as a floor baseline.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Callable

import httpx

from c2a.decide.base import normalize_probs
from c2a.decide.systemone import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
)

TYPESAFE_BASE_URL = "https://api.typesafe.ai"
JEV_MODEL = "jev-latest"
RETRY_STATUS = {429, 500, 502, 503, 504}


class SystemOneClient:
    def __init__(
        self,
        base_url: str = TYPESAFE_BASE_URL,
        model: str = JEV_MODEL,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 4,
        backoff: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        api_key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise RuntimeError("TYPESAFE_API_KEY is not set (needed for hosted Jev)")
        self.model = model
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def ask(self, request: SystemOneRequest) -> SystemOneResponse:
        body = request.model_dump(mode="json", exclude_none=True)
        body["model"] = self.model if request.model == JEV_MODEL else request.model
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post("/v1/systemone", json=body)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
            else:
                if resp.status_code not in RETRY_STATUS or attempt == self.max_retries:
                    resp.raise_for_status()
                    return SystemOneResponse.model_validate(resp.json())
            self._sleep(self.backoff * 2**attempt)
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._client.close()


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


class KeywordDecider:
    """Word overlap between the state and each option/level. Deterministic, offline."""

    model = "keyword"

    def ask(self, request: SystemOneRequest) -> SystemOneResponse:
        words = _words(str(request.state))
        answers = {}
        for qid, q in request.questions.items():
            words_q = words | _words(str(q.instructions or ""))
            if isinstance(q, Choice):
                labels = list(q.criteria)
                scores = {
                    lbl: float(len(_words(f"{lbl} {q.criteria[lbl] or ''}") & words_q))
                    for lbl in labels
                }
                res = normalize_probs(scores, labels)
                answers[qid] = ChoiceAnswer(
                    choice=res.label, confidence=_confidence(res.probs), probabilities=res.probs
                )
            elif isinstance(q, Score):
                keys = [str(i) for i in range(len(q.criteria))]
                scores = {
                    k: float(len(_words(lbl) & words_q))
                    for k, lbl in zip(keys, q.criteria, strict=True)
                }
                res = normalize_probs(scores, keys)
                answers[qid] = ScoreAnswer(
                    score=sum(int(k) * p for k, p in res.probs.items()),
                    confidence=_confidence(res.probs),
                    legend=dict(zip(keys, q.criteria, strict=True)),
                    probabilities=res.probs,
                )
            elif isinstance(q, Noul):
                answers[qid] = NoulAnswer(noul=0.5)
        return SystemOneResponse(model=self.model, answers=answers)


def _confidence(probs: dict[str, float]) -> float:
    """1 - normalized entropy: 0 for uniform, 1 for one-hot."""
    n = len(probs)
    if n < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(n)))
