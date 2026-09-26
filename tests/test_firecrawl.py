import json

import httpx
import pytest
import respx

from c2a.decide.systemone import ChoiceAnswer, SystemOneResponse
from c2a.sources import firecrawl as fc
from c2a.sources.compliance import CrawlSession, RateLimiter, parse_robots

API = "https://api.firecrawl.dev"
EXTRACT = {
    "title": "Wool Runner",
    "brand": "Acme",
    "currency": "SEK",
    "price": "1 299,00 kr",
    "images": ["https://cdn/x.jpg"],
    "breadcrumbs": ["Shoes", "Runners"],
    "sku": "WR-1",
}


class PageGate:
    """Scripted Jev: pages whose URL contains 'wool' are product pages."""

    def __init__(self):
        self.calls = 0

    def ask(self, request):
        self.calls += 1
        is_product = "wool" in request.state["url"]
        label = "product_detail" if is_product else "category_listing"
        opts = list(request.questions["page_type"].criteria)
        probs = {o: (0.97 if o == label else 0.01) for o in opts}
        return SystemOneResponse(
            model="jev-latest",
            answers={"page_type": ChoiceAnswer(choice=label, confidence=0.95, probabilities=probs)},
        )


def session(store):
    return CrawlSession(
        store,
        client=httpx.Client(),
        limiter=RateLimiter(1000, sleep=lambda s: None),
        robots=parse_robots("User-agent: *\nDisallow: /checkout\n"),
    )


def scrape_router(request):
    body = json.loads(request.content)
    if body["formats"] == ["markdown"]:
        return httpx.Response(200, json={"success": True, "data": {"markdown": "# page"}})
    return httpx.Response(200, json={"success": True, "data": {"json": EXTRACT}})


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("1 299,00 kr", "1299.00"),
        ("£19.99", "19.99"),
        ("1,299.50", "1299.50"),
        ("1.299,50 €", "1299.50"),
        ("$1,299", "1299"),
        (129.0, "129.0"),
        ("free", None),
    ],
)
def test_parse_price(text, want):
    got = fc.parse_price(text)
    assert (str(got) if got is not None else None) == want


def test_triage_url():
    assert fc.triage_url("https://s.com/products/wool-runner") == "product"
    assert fc.triage_url("https://s.com/collections/shoes/products/x") == "product"
    assert fc.triage_url("https://s.com/blog/how-to") == "skip"
    assert fc.triage_url("https://s.com/img/a.jpg") == "skip"
    assert fc.triage_url("https://s.com/wool-runner-grey") == "ambiguous"


def test_to_product_minor_units(store):
    p = fc.to_product(fc.ExtractedProduct.model_validate(EXTRACT), "https://s/x", store)
    assert p.variants[0].price.amount == 129900 and p.category == "Shoes > Runners"
    assert p.id == "acme:WR-1" and p.source == "firecrawl"
    with pytest.raises(ValueError):
        fc.to_product(
            fc.ExtractedProduct(title="x", price="n/a", currency="SEK"), "https://s/x", store
        )


@respx.mock
def test_crawl_store_page_gate_controls_extract_spend(store):
    respx.post(f"{API}/v2/map").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "links": [
                    {"url": "https://shop.example.com/products/a"},
                    "https://shop.example.com/wool-runner",  # ambiguous -> gate says product
                    "https://shop.example.com/sale-shoes",  # ambiguous -> gate says listing
                    "https://shop.example.com/blog/post",  # skip
                    "https://shop.example.com/checkout",  # robots-denied
                ],
            },
        )
    )
    scrape = respx.post(f"{API}/v2/scrape").mock(side_effect=scrape_router)
    client = fc.FirecrawlClient(api_key="k")
    gate = PageGate()
    res = fc.crawl_store(session(store), client, gate)
    formats = [json.loads(c.request.content)["formats"] for c in scrape.calls]
    json_calls = [f for f in formats if f != ["markdown"]]
    assert formats.count(["markdown"]) == 2 and len(json_calls) == 2  # /products/a + wool-runner
    assert res.triage == {"product": 1, "skip": 1, "ambiguous": 2} and res.urls_denied == 1
    assert res.gated_out == 1 and gate.calls == 2 and len(res.products) == 2
    assert res.credits_spent == 1 + 2 * 1 + 2 * 5


@respx.mock
def test_no_decider_never_extracts_ambiguous(store):
    respx.post(f"{API}/v2/map").mock(
        return_value=httpx.Response(
            200, json={"success": True, "links": ["https://shop.example.com/x"]}
        )
    )
    scrape = respx.post(f"{API}/v2/scrape").mock(side_effect=scrape_router)
    res = fc.crawl_store(session(store), fc.FirecrawlClient(api_key="k"), None)
    assert scrape.call_count == 0 and res.products == []


@respx.mock
def test_budget_stops_crawl(store):
    respx.post(f"{API}/v2/map").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "links": [f"https://shop.example.com/products/{i}" for i in range(5)],
            },
        )
    )
    respx.post(f"{API}/v2/scrape").mock(side_effect=scrape_router)
    res = fc.crawl_store(session(store), fc.FirecrawlClient(api_key="k", credit_budget=12), None)
    assert res.budget_exhausted and len(res.products) == 2 and res.credits_spent == 11


@respx.mock
def test_search_accepts_both_shapes():
    route = respx.post(f"{API}/v2/search")
    route.mock(
        return_value=httpx.Response(
            200, json={"success": True, "data": {"web": [{"url": "https://a.se"}]}}
        )
    )
    client = fc.FirecrawlClient(api_key="k")
    assert client.search("q") == [{"url": "https://a.se"}]
    route.mock(
        return_value=httpx.Response(
            200, json={"success": True, "data": [{"url": "https://b.se"}, {}]}
        )
    )
    assert client.search("q") == [{"url": "https://b.se"}]
