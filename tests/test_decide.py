import httpx
import pytest
import respx

from c2a.decide.backends import KeywordDecider, OpenAICompatDecider
from c2a.decide.base import normalize_probs
from c2a.schemas import DecisionRequest

REQ = DecisionRequest(
    evidence="Customer bought running shoes; candidate is running socks",
    question="How does the candidate relate to the purchase?",
    labels=["complement", "substitute", "running socks"],
)


def test_normalize_probs_sums_to_one_and_argmax():
    res = normalize_probs({"a": 0.0, "b": 2.0}, ["a", "b", "c"])
    assert res.label == "b" and sum(res.probs.values()) == pytest.approx(1.0)
    assert set(res.probs) == {"a", "b", "c"}


def test_keyword_decider_is_deterministic():
    a, b = KeywordDecider().decide(REQ), KeywordDecider().decide(REQ)
    assert a == b and a.label == "running socks"


@respx.mock
def test_openai_compat_decider_reads_first_token_logprobs():
    respx.post("http://decider/v1/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"logprobs": {"top_logprobs": [{" complement": -0.1, " substitute": -2.5}]}}
                ]
            },
        )
    )
    res = OpenAICompatDecider("http://decider", "m").decide(
        DecisionRequest(evidence="e", question="q", labels=["complement", "substitute"])
    )
    assert res.label == "complement" and res.probs["complement"] > 0.9
