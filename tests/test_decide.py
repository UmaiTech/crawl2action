import httpx
import pytest
import respx

from c2a.decide.backends import KeywordDecider, SystemOneClient
from c2a.decide.base import decide, normalize_probs
from c2a.decide.systemone import (
    Choice,
    ChoiceAnswer,
    Noul,
    Score,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    top_label,
    top_prob,
)

# Documented System One example response (Kev README / TypeSafe contract).
EXAMPLE_RESPONSE = {
    "model": "kev-latest",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "returns",
            "confidence": 0.21,
            "probabilities": {"returns": 0.47, "shipping": 0.28, "billing": 0.25},
        },
        "escalate": {"type": "noul", "noul": 0.93},
        "frustration": {
            "type": "score",
            "score": 1.44,
            "confidence": 0.78,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.00, "1": 0.56, "2": 0.44},
        },
    },
    "usage": {"input_tokens": 101, "output_tokens": 161},
    "latency_ms": 495,
}

REQUEST = SystemOneRequest(
    state="Charged twice for order 123, where is my refund?",
    questions={
        "department": Choice(criteria={"returns": None, "shipping": None, "billing": "Payments"}),
        "escalate": Noul(instructions="Should a human take over?"),
        "frustration": Score(criteria=["Calm", "Frustrated", "Very angry"]),
    },
)


def test_example_response_roundtrip_and_helpers():
    resp = SystemOneResponse.model_validate(EXAMPLE_RESPONSE)
    assert SystemOneResponse.model_validate_json(resp.model_dump_json()) == resp
    assert top_label(resp.answers["department"]) == "returns"
    assert top_label(resp.answers["escalate"]) == "true"
    assert top_label(resp.answers["frustration"]) == "Frustrated"
    assert top_prob(resp.answers["escalate"]) == pytest.approx(0.93)


def test_request_serializes_typed_questions():
    body = REQUEST.model_dump(mode="json", exclude_none=True)
    assert body["model"] == "jev-latest"
    assert body["questions"]["frustration"] == {
        "type": "score",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    }
    with pytest.raises(ValueError):
        Choice(criteria={})


@respx.mock
def test_client_posts_with_auth_and_retries_on_429():
    route = respx.post("https://api.typesafe.ai/v1/systemone").mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json=EXAMPLE_RESPONSE)]
    )
    sleeps = []
    client = SystemOneClient(api_key="k", sleep=sleeps.append)
    resp = client.ask(REQUEST)
    assert resp.answers["escalate"].noul == 0.93
    assert route.call_count == 2 and sleeps == [1.0]
    req = route.calls.last.request
    assert req.headers["Authorization"] == "Bearer k"
    assert b'"model":"jev-latest"' in req.content.replace(b" ", b"")


@respx.mock
def test_client_raises_after_retries_and_on_4xx():
    respx.post("https://api.typesafe.ai/v1/systemone").mock(return_value=httpx.Response(401))
    with pytest.raises(httpx.HTTPStatusError):
        SystemOneClient(api_key="k", sleep=lambda s: None).ask(REQUEST)


def test_client_requires_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        SystemOneClient()


def test_keyword_decider_answers_every_question():
    resp = KeywordDecider().ask(REQUEST)
    assert set(resp.answers) == {"department", "escalate", "frustration"}
    assert isinstance(resp.answers["frustration"], ScoreAnswer)
    dep = resp.answers["department"]
    assert isinstance(dep, ChoiceAnswer) and sum(dep.probabilities.values()) == pytest.approx(1.0)
    billing = KeywordDecider().ask(REQUEST.model_copy(update={"state": "billing issue"}))
    assert billing.answers["department"].choice == "billing"


def test_decide_helper_wraps_single_choice():
    res = decide(KeywordDecider(), "running socks for shoes", "relation?", ["complement", "socks"])
    assert res.label == "socks" and sum(res.probs.values()) == pytest.approx(1.0)


def test_normalize_probs_sums_to_one_and_argmax():
    res = normalize_probs({"a": 0.0, "b": 2.0}, ["a", "b", "c"])
    assert res.label == "b" and sum(res.probs.values()) == pytest.approx(1.0)
