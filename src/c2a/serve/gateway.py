"""Thin FastAPI gateway. In production vLLM Semantic Router sits in front of it and routes
to the fast reranker / 27B student / image pipeline (configs/router/semantic-router.yaml).

M0 dev behaviour for /recommend: retrieve candidates, rank them with a pluggable ranker
(defaults to retrieval order), and validate with the same grader checks used in training
(no hallucinated IDs, constraints respected)."""

from __future__ import annotations

from collections.abc import Callable

from c2a.decide.backends import KeywordDecider
from c2a.decide.base import Decider
from c2a.graders.rec import constraint_violations, hallucinated_ids
from c2a.index import Retriever
from c2a.schemas import (
    Candidate,
    DecisionRequest,
    DecisionResult,
    RecItem,
    RecRequest,
    RecResponse,
)

Ranker = Callable[[RecRequest, list[Candidate]], list[RecItem]]


def retrieval_order_ranker(req: RecRequest, cands: list[Candidate]) -> list[RecItem]:
    return [RecItem(product_id=c.product_id, reason="retrieval match") for c in cands]


def create_app(
    retriever: Retriever, ranker: Ranker = retrieval_order_ranker, decider: Decider | None = None
):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the serve extra: uv sync --extra serve") from exc

    app = FastAPI(title="crawl2action gateway", version="0.1.0")
    decider = decider or KeywordDecider()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/recommend", response_model=RecResponse)
    def recommend(req: RecRequest) -> RecResponse:
        cands = req.candidates or (retriever.search(req.query, k=50) if req.query else [])
        items = ranker(req, cands)
        ids = [it.product_id for it in items]
        if hallucinated_ids(ids, cands):
            raise HTTPException(502, "ranker returned ids outside the candidate set")
        bad = {pid for pid, _ in constraint_violations(ids, cands, req.constraints)}
        return RecResponse(items=[it for it in items if it.product_id not in bad][: req.k])

    @app.post("/decide", response_model=DecisionResult)
    def decide(req: DecisionRequest) -> DecisionResult:
        return decider.decide(req)

    @app.post("/generate/text")
    def generate_text() -> None:
        raise HTTPException(501, "text generation lands in M4")

    @app.post("/generate/image")
    def generate_image() -> None:
        raise HTTPException(501, "image generation lands in M4")

    @app.post("/feedback")
    def feedback() -> None:
        raise HTTPException(501, "feedback logging lands in M5")

    return app
