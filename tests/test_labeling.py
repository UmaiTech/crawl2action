import json

import pytest
from typer.testing import CliRunner

from c2a.cli import app
from c2a.decide.systemone import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
)
from c2a.labeling.cache import ResponseCache
from c2a.labeling.escalation import apply_review
from c2a.labeling.export import to_decider_train, to_gold
from c2a.labeling.gating import Thresholds, is_confident, load_thresholds
from c2a.labeling.pipeline import label
from c2a.labeling.questions import get_question_set, pair_v1, product_v1
from c2a.labeling.records import LabelItem
from c2a.labeling.state import pair_state, product_state
from c2a.sources.shopify import parse_products_json
from c2a.train.data import write_jsonl


class ScriptedDecider:
    """Returns fixed answers for product_v1 and counts calls."""

    model = "jev-latest"

    def __init__(self, confident: bool = True):
        self.calls = 0
        self.confident = confident

    def ask(self, request: SystemOneRequest) -> SystemOneResponse:
        self.calls += 1
        c = 0.9 if self.confident else 0.3
        answers = {}
        for qid, q in request.questions.items():
            if q.type == "noul":
                answers[qid] = NoulAnswer(noul=0.02 if self.confident else 0.6)
            elif q.type == "choice":
                opts = list(q.criteria)
                probs = {
                    o: (c if i == 0 else (1 - c) / (len(opts) - 1)) for i, o in enumerate(opts)
                }
                answers[qid] = ChoiceAnswer(choice=opts[0], confidence=c, probabilities=probs)
            else:
                keys = [str(i) for i in range(len(q.criteria))]
                probs = {k: (c if k == "1" else (1 - c) / (len(keys) - 1)) for k in keys}
                answers[qid] = ScoreAnswer(
                    score=1.0,
                    confidence=c,
                    legend=dict(zip(keys, q.criteria, strict=True)),
                    probabilities=probs,
                )
        return SystemOneResponse(model=self.model, answers=answers)


@pytest.fixture
def products(store, shopify_payload):
    return parse_products_json(shopify_payload, store, "SEK")[0]


@pytest.fixture
def items(products):
    return [LabelItem(id=p.id, state=product_state(p)) for p in products]


def test_question_sets_are_valid_and_versioned():
    for name in ("page_v1", "product_v1", "pair_v1"):
        qs = get_question_set(name)
        assert qs.questions and "@" in qs.id
    assert "apparel" in product_v1().questions["vertical"].criteria
    assert list(pair_v1().questions["relation"].criteria) == [
        "exact",
        "substitute",
        "complement",
        "irrelevant",
    ]
    with pytest.raises(KeyError):
        get_question_set("nope")


def test_product_state_is_compact(products):
    s = product_state(products[0])
    assert s["price"] == "129.00 SEK" and s["description"] == "Soft merino sneaker."
    assert s["options"] == {"Color": ["Grey"], "Size": ["42", "43"]}
    assert "attributes" not in s  # empty fields dropped
    assert pair_state("wool sneakers", products[0])["anchor"] == {"query": "wool sneakers"}


def test_gating_per_type_and_overrides():
    t = Thresholds(
        min_confidence=0.6,
        min_top_prob=0.7,
        noul_min=0.85,
        per_question={"restricted": {"noul_min": 0.95}},
    )
    assert is_confident(NoulAnswer(noul=0.1), t)
    assert not is_confident(NoulAnswer(noul=0.1), t.for_question("restricted"))
    good = ChoiceAnswer(choice="a", confidence=0.7, probabilities={"a": 0.8, "b": 0.2})
    low_conf = good.model_copy(update={"confidence": 0.5})
    assert is_confident(good, t) and not is_confident(low_conf, t)
    assert load_thresholds().per_question["contains_pii"]["noul_min"] == 0.95


def test_confident_labels_are_accepted_and_cached(tmp_path, items):
    cache = ResponseCache(tmp_path / "cache")
    dec = ScriptedDecider()
    run = label(items, dec, product_v1(), load_thresholds(), cache=cache)
    assert dec.calls == 2 and run.requests_made == 2
    assert len(run.records) == 2 * len(product_v1().questions)
    assert all(r.status == "accepted" and r.source == "jev" for r in run.records)
    price = next(r for r in run.records if r.question_id == "price_tier")
    assert price.value == "mid_market" and "mid_market" in price.probabilities
    bundle = next(r for r in run.records if r.question_id == "is_bundle")
    assert bundle.value == "false"
    rerun = label(items, dec, product_v1(), load_thresholds(), cache=cache)
    assert dec.calls == 2 and rerun.cache_hits == 2  # no second payment


def test_budget_cap_skips_items(items):
    dec = ScriptedDecider()
    run = label(items, dec, product_v1(), load_thresholds(), max_requests=1)
    assert dec.calls == 1 and run.budget_exhausted and run.skipped == [items[1].id]


def test_escalation_teacher_agree_and_disagree(items):
    qs = product_v1()
    agree = label(
        items[:1],
        ScriptedDecider(False),
        qs,
        load_thresholds(),
        teacher=lambda state, qid, q: (
            "true"
            if q.type == "noul"
            else (list(q.criteria)[0] if q.type == "choice" else q.criteria[1])
        ),
    )
    assert all(r.status == "accepted" and r.source == "jev+teacher" for r in agree.records)
    assert agree.review == []

    disagree = label(
        items[:1],
        ScriptedDecider(False),
        qs,
        load_thresholds(),
        teacher=lambda state, qid, q: "something else",
    )
    assert all(r.status == "review" for r in disagree.records)
    assert len(disagree.review) == len(qs.questions)
    assert disagree.review[0].record.teacher_value == "something else"

    no_teacher = label(items[:1], ScriptedDecider(False), qs, load_thresholds())
    assert all(r.status == "review" for r in no_teacher.records)


def test_exports_kev_format_and_exclude_jev_only(items):
    qs = product_v1()
    run = label(items, ScriptedDecider(), qs, load_thresholds())
    states = {it.id: it.state for it in items}
    assert to_decider_train(states, run.records, qs) == []  # Jev-only excluded by default
    rows = to_decider_train(states, run.records, qs, sources={"jev"})
    q = rows[0]["questions"]
    assert q["is_bundle"]["label"] is False and q["price_tier"]["label"] == 1
    assert q["vertical"]["label"] == "apparel" and q["vertical"]["type"] == "choice"

    reviewed = [
        apply_review(r, "premium", "ann") for r in run.records if r.question_id == "price_tier"
    ]
    gold = to_gold(reviewed)
    assert gold[0]["label"] == "premium" and gold[0]["reviewer"] == "ann"
    human_rows = to_decider_train(states, reviewed, qs)
    assert human_rows[0]["questions"]["price_tier"]["label"] == 2


def test_cli_dry_run_and_offline_run(tmp_path, products):
    runner = CliRunner()
    inp = tmp_path / "products.jsonl"
    write_jsonl(inp, products)
    res = runner.invoke(app, ["label", "run", "--input", str(inp), "--dry-run"])
    assert res.exit_code == 0 and '"questions"' in res.output and "product@1" in res.output
    res = runner.invoke(
        app,
        [
            "label",
            "run",
            "--input",
            str(inp),
            "--backend",
            "keyword",
            "--out",
            str(tmp_path / "labels"),
        ],
    )
    assert res.exit_code == 0, res.output
    run_dir = tmp_path / "labels" / "product_v1"
    assert (run_dir / "labels.jsonl").exists() and (run_dir / "items.jsonl").exists()
    out = tmp_path / "gold.jsonl"
    res = runner.invoke(
        app, ["label", "export", "--run-dir", str(run_dir), "--kind", "gold", "--out", str(out)]
    )
    assert res.exit_code == 0 and out.read_text() == ""  # no human labels yet
    json.loads((run_dir / "labels.jsonl").read_text().splitlines()[0])
