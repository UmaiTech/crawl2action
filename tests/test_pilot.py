import json
import os

import httpx
import pytest
import respx
from typer.testing import CliRunner

from c2a.cli import app
from c2a.config import Settings, load_dotenv
from c2a.decide.systemone import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse
from c2a.doctor import format_checks, run_doctor
from c2a.labeling.pairs import candidate_pairs, pair_items
from c2a.labeling.review import apply_decision, format_item, load_queue
from c2a.pilot import run_pilot
from c2a.schemas import Money, Product, TosStatus, Variant
from c2a.sources import shopify
from c2a.sources.compliance import CrawlSession, RateLimiter, parse_robots
from c2a.sources.registry import Registry


class UnsureDecider:
    """Answers every question with low confidence, so everything goes to review."""

    model = "jev-latest"

    def __init__(self):
        self.calls = 0

    def ask(self, request):
        self.calls += 1
        answers = {}
        for qid, q in request.questions.items():
            if q.type == "noul":
                answers[qid] = NoulAnswer(noul=0.55)
            elif q.type == "choice":
                opts = list(q.criteria)
                p = 1 / len(opts)
                answers[qid] = ChoiceAnswer(
                    choice=opts[0], confidence=0.1, probabilities=dict.fromkeys(opts, p)
                )
            else:
                keys = [str(i) for i in range(len(q.criteria))]
                answers[qid] = ScoreAnswer(
                    score=1.0,
                    confidence=0.1,
                    legend=dict(zip(keys, q.criteria, strict=True)),
                    probabilities=dict.fromkeys(keys, 1 / len(keys)),
                )
        return SystemOneResponse(model=self.model, answers=answers)


def product(pid, title, category):
    return Product(
        id=f"s:{pid}",
        store_id="s",
        native_id=pid,
        title=title,
        category=category,
        variants=[Variant(id=pid, price=Money(amount=100, currency="SEK"))],
    )


def session(store, **kw):
    return CrawlSession(
        store,
        client=httpx.Client(),
        limiter=RateLimiter(1000, sleep=lambda s: None),
        robots=parse_robots(""),
        sleep=lambda s: None,
        **kw,
    )


def test_load_dotenv_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# c\nexport C2A_A="one"\nC2A_B=two\nC2A_C=\nbad line\n')
    monkeypatch.setenv("C2A_B", "keep")
    monkeypatch.delenv("C2A_A", raising=False)
    assert load_dotenv(env) == ["C2A_A"]
    assert os.environ["C2A_A"] == "one" and os.environ["C2A_B"] == "keep"
    monkeypatch.delenv("C2A_A")


@respx.mock
def test_session_retries_on_429_with_retry_after(store):
    route = respx.get("https://shop.example.com/products/x").mock(
        side_effect=[httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200)]
    )
    sleeps = []
    s = CrawlSession(
        store,
        client=httpx.Client(),
        limiter=RateLimiter(1000, sleep=lambda x: None),
        robots=parse_robots(""),
        sleep=sleeps.append,
    )
    assert s.get("https://shop.example.com/products/x").status_code == 200
    assert route.call_count == 2 and sleeps == [7.0] and s.retries == 1


@respx.mock
def test_shopify_currency_chain_and_page_guards(store):
    store.currency = None
    respx.get("https://shop.example.com/meta.json").mock(return_value=httpx.Response(404))
    page = {
        "products": [
            {"id": 1, "title": "A", "handle": "a", "variants": [{"id": 10, "price": "1.00"}]}
        ]
    }
    route = respx.get("https://shop.example.com/products.json").mock(
        return_value=httpx.Response(200, json=page)  # theme ignores ?page -> same page forever
    )
    s = session(store)
    products, _ = shopify.crawl(s, limit=1)
    assert len(products) == 1 and route.call_count == 2  # stops on the repeated page
    assert products[0].variants[0].price.currency == "SEK"
    assert "assumed SEK" in s.warnings[0]


def test_candidate_pairs_deterministic_no_dupes():
    prods = [
        product("1", "wool socks grey", "socks"),
        product("2", "wool socks black", "socks"),
        product("3", "merino wool socks", "socks"),
        product("4", "running shoes", "shoes"),
    ]
    a = candidate_pairs(prods, k=2, other=1)
    assert a == candidate_pairs(list(reversed(prods)), k=2, other=1)
    keys = [frozenset((x.id, y.id)) for x, y in a]
    assert len(keys) == len(set(keys)) and all(x.id != y.id for x, y in a)
    assert any({x.category, y.category} == {"socks", "shoes"} for x, y in a)
    items = pair_items(a)
    assert items[0].id.startswith("pair:") and "candidate" in items[0].state


@respx.mock
def test_doctor_reports_fixes(store):
    respx.get("https://api.typesafe.ai").mock(return_value=httpx.Response(404))
    respx.get("https://api.firecrawl.dev").mock(side_effect=httpx.ConnectError("no"))
    respx.get("https://shop.example.com/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nDisallow: /products.json\n")
    )
    os.environ.pop("TYPESAFE_API_KEY", None)
    store.currency = None
    checks = {
        c.name: c for c in run_doctor(Registry(stores=[store]), Settings(), client=httpx.Client())
    }
    assert checks["TYPESAFE_API_KEY"].status == "fail"
    assert checks["Jev API reachable"].status == "ok"
    assert checks["Firecrawl API reachable"].status == "warn"  # no Firecrawl stores
    assert checks["acme: /products.json allowed"].status == "fail"
    assert checks["acme: currency"].status == "warn"
    empty = run_doctor(Registry(stores=[]), Settings(), client=httpx.Client())
    assert "c2a registry approve" in format_checks(empty)


@respx.mock
def test_doctor_live_shows_raw_body_on_format_mismatch(store, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    respx.get("https://api.typesafe.ai").mock(return_value=httpx.Response(404))
    respx.get("https://api.firecrawl.dev").mock(return_value=httpx.Response(404))
    respx.get("https://shop.example.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.post("https://api.typesafe.ai/v1/systemone").mock(
        return_value=httpx.Response(200, json={"result": {"unexpected": True}})
    )
    checks = run_doctor(Registry(stores=[store]), Settings(), live=True, client=httpx.Client())
    bad = next(c for c in checks if c.name == "Jev response format")
    assert bad.status == "fail" and "unexpected" in bad.detail


@respx.mock
def test_pilot_end_to_end_then_review(tmp_path, store):
    store.currency = "SEK"
    respx.get("https://shop.example.com/products.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "products": [
                    {
                        "id": i,
                        "title": t,
                        "handle": f"h{i}",
                        "product_type": c,
                        "variants": [{"id": i * 10, "price": "99.00"}],
                    }
                    for i, (t, c) in enumerate(
                        [
                            ("wool socks", "socks"),
                            ("wool socks thick", "socks"),
                            ("trail shoes", "shoes"),
                        ],
                        1,
                    )
                ]
            },
        )
    )
    other = store.model_copy(update={"id": "other", "tos_status": TosStatus.unreviewed})
    reg = Registry(stores=[store, other])
    with pytest.raises(ValueError, match="not approved"):
        run_pilot(reg, Settings(), ["other"], UnsureDecider(), data_dir=tmp_path)

    dec = UnsureDecider()
    res = run_pilot(
        reg,
        Settings(),
        ["acme"],
        dec,
        data_dir=tmp_path,
        pairs_k=1,
        session_factory=lambda s: session(s),
    )
    assert res.crawled == {"acme": 3} and res.product_labels == 3 * 8
    assert res.pair_labels > 0 and res.to_review == res.product_labels + res.pair_labels
    assert "## Labels: product_v1" in res.report and "## Labels: pair_v1" in res.report
    assert (tmp_path / "reports" / "pilot.md").exists()

    # human review: answering one item replaces its record and shrinks the queue
    run_dir = tmp_path / "labels" / "product_v1"
    queue = load_queue(run_dir)
    item = next(q for q in queue if q.record.question_id == "vertical")
    assert "Q: Which vertical" in format_item(item, 1, len(queue))
    decided = apply_decision(run_dir, item, "apparel", "marcus")
    assert decided.source == "human" and len(load_queue(run_dir)) == len(queue) - 1
    labels = [json.loads(line) for line in (run_dir / "labels.jsonl").read_text().splitlines()]
    mine = [
        r for r in labels if r["item_id"] == item.record.item_id and r["question_id"] == "vertical"
    ]
    assert len(mine) == 1 and mine[0]["value"] == "apparel" and mine[0]["reviewer"] == "marcus"
    with pytest.raises(ValueError):
        apply_decision(run_dir, load_queue(run_dir)[0], "not-an-option", "marcus")

    # rerun: nothing changed, no new Jev calls for products, cached pairs
    calls = dec.calls
    res2 = run_pilot(
        reg,
        Settings(),
        ["acme"],
        dec,
        data_dir=tmp_path,
        pairs_k=1,
        session_factory=lambda s: session(s),
    )
    assert res2.crawled == {"acme": 0} and dec.calls == calls

    out = CliRunner().invoke(app, ["report", "--data", str(tmp_path)])
    assert out.exit_code == 0 and "| vertical |" in out.output and "| relation |" in out.output


def test_review_cli_skip_and_quit_leave_files(tmp_path):
    run_dir = tmp_path / "product_v1"
    run_dir.mkdir()
    res = CliRunner().invoke(app, ["label", "review", "--run-dir", str(run_dir), "--by", "m"])
    assert res.exit_code == 0 and "nothing to review" in res.output
