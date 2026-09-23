import json

import httpx
import respx

from c2a.schemas import TosStatus
from c2a.sources.compliance import CrawlSession, RateLimiter, parse_robots
from c2a.sources.crawl import crawl_registry, plan
from c2a.sources.discover import expand, fingerprint
from c2a.sources.firecrawl import FirecrawlClient
from c2a.sources.registry import Registry

from .test_ucp import PROFILE


def mock_site(domain, well_known=None, products_json=None, home=""):
    respx.get(f"https://{domain}/robots.txt").mock(return_value=httpx.Response(404))
    respx.get(f"https://{domain}/.well-known/ucp").mock(
        return_value=httpx.Response(200, json=well_known) if well_known else httpx.Response(404)
    )
    respx.get(f"https://{domain}/products.json").mock(
        return_value=httpx.Response(200, json=products_json)
        if products_json
        else httpx.Response(404)
    )
    respx.get(f"https://{domain}/").mock(return_value=httpx.Response(200, text=home))


@respx.mock
def test_fingerprint_platforms():
    mock_site("u.test", well_known=PROFILE)
    mock_site("s.test", products_json={"products": []})
    mock_site("w.test", home="<link href='/wp-content/plugins/woocommerce/x.css'>")
    mock_site("c.test", home="<html>hello</html>")
    got = {d: fingerprint(d) for d in ("u.test", "s.test", "w.test", "c.test")}
    assert {d: (f.platform, f.method) for d, f in got.items()} == {
        "u.test": ("ucp", "ucp"),
        "s.test": ("shopify", "shopify"),
        "w.test": ("woocommerce", "firecrawl"),
        "c.test": ("custom", "firecrawl"),
    }


@respx.mock
def test_expand_emits_unreviewed_disabled_candidates():
    respx.post("https://api.firecrawl.dev/v2/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "web": [
                        {"url": "https://www.nyshop.se/kläder", "title": "Nyshop"},
                        {"url": "https://www.amazon.se/x"},
                        {"url": "https://known.se/"},
                        {"url": "https://www.nyshop.se/other"},
                    ]
                },
            },
        )
    )
    stores = expand("SE", "apparel", FirecrawlClient(api_key="k"), {"known.se"})
    assert [s.domain for s in stores] == ["www.nyshop.se"]
    s = stores[0]
    assert s.id == "nyshop" and s.locale == "sv-SE" and s.tier.value == "B"
    assert s.tos_status is TosStatus.unreviewed and not s.enabled


def shopify_page(n):
    return {
        "products": [
            {
                "id": i,
                "title": f"P{i}",
                "handle": f"p{i}",
                "variants": [{"id": i * 10, "price": "10.00", "available": True}],
            }
            for i in range(n)
        ]
    }


@respx.mock
def test_crawl_registry_incremental_and_manifest(tmp_path, store):
    store.currency = "SEK"
    other = store.model_copy(
        update={"id": "other", "domain": "o.test", "tos_status": TosStatus.unreviewed}
    )
    reg = Registry(stores=[store, other])
    assert [(s.id, r) for s, r in plan(reg)] == [("acme", "ok"), ("other", "tos_status=unreviewed")]

    route = respx.get("https://shop.example.com/products.json")
    route.mock(
        side_effect=lambda req: httpx.Response(
            200, json=shopify_page(3) if req.url.params["page"] == "1" else {"products": []}
        )
    )

    def factory(s):
        return CrawlSession(
            s,
            client=httpx.Client(),
            limiter=RateLimiter(1000, sleep=lambda x: None),
            robots=parse_robots(""),
        )

    ((m, changed),) = crawl_registry(reg, tmp_path, session_factory=factory)
    assert m.products_new_or_changed == 3 and len(changed) == 3 and m.errors == []
    manifest = json.loads((tmp_path / "acme" / "manifest.json").read_text())
    assert manifest["products_total"] == 3

    ((m2, changed2),) = crawl_registry(reg, tmp_path, session_factory=factory)
    assert m2.products_unchanged == 3 and changed2 == []
    assert not (tmp_path / "other").exists()  # never crawled


def test_crawl_cli_dry_run_reports_skips():
    from typer.testing import CliRunner

    from c2a.cli import app

    res = CliRunner().invoke(app, ["crawl", "--dry-run", "--store", "allbirds"])
    assert res.exit_code == 0 and "skip: tos_status=unreviewed" in res.output
