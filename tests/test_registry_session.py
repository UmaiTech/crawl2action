import httpx
import pytest
import respx

from c2a.schemas import TosStatus
from c2a.sources.compliance import CrawlSession, RateLimiter, parse_robots
from c2a.sources.registry import load_registry, save_registry

from .conftest import FIXTURES


def test_approve_deny_roundtrip_with_audit(tmp_path):
    reg = load_registry()
    path = tmp_path / "registry.yaml"
    s = reg.approve("allbirds", by="marcus", note="ToS allows crawling product pages")
    assert s.enabled and s.tos_status is TosStatus.approved and s.reviewed_at is not None
    save_registry(reg, path)
    again = load_registry(path)
    a = again.by_id("allbirds")
    assert a.reviewed_by == "marcus" and a.review_note.startswith("ToS")
    assert [x.id for x in again.crawlable()] == ["allbirds"]
    again.deny("allbirds", by="legal")
    assert again.crawlable() == [] and again.by_id("allbirds").tos_status is TosStatus.denied
    with pytest.raises(ValueError):
        again.approve("allbirds", by=" ")


def test_automated_updates_cannot_touch_permissions():
    reg = load_registry()
    reg.update_descriptive("kotn", platform="shopify", method="shopify")
    assert reg.by_id("kotn").method == "shopify"
    with pytest.raises(ValueError):
        reg.update_descriptive("kotn", enabled=True)
    with pytest.raises(ValueError):
        reg.update_descriptive("kotn", tos_status="approved")


def _session(store, robots_text=None):
    calls = []
    rl = RateLimiter(qps=1000, sleep=lambda s: None)
    rl.wait = lambda d: calls.append(d)  # type: ignore[method-assign]
    rp = parse_robots(robots_text or (FIXTURES / "robots.txt").read_text())
    return CrawlSession(store, client=httpx.Client(), limiter=rl, robots=rp), calls


@respx.mock
def test_crawl_session_gates_every_request(store):
    respx.get("https://shop.example.com/products/x").mock(return_value=httpx.Response(200))
    session, calls = _session(store)
    assert session.get("https://shop.example.com/products/x").status_code == 200
    assert calls == ["shop.example.com"]
    for url in ("https://shop.example.com/checkout", "https://evil.example.com/products/x"):
        with pytest.raises(PermissionError):
            session.get(url)
    assert len(session.denied) == 2 and calls == ["shop.example.com"]
    store.tos_status = TosStatus.unreviewed
    with pytest.raises(PermissionError, match="tos_status"):
        session.get("https://shop.example.com/products/x")


@respx.mock
def test_api_host_needs_tos_but_not_robots(store):
    respx.post("https://api.platform.example/ucp/catalog/search").mock(
        return_value=httpx.Response(200, json={})
    )
    session, _ = _session(store, robots_text="User-agent: *\nDisallow: /")
    with pytest.raises(PermissionError):
        session.post("https://api.platform.example/ucp/catalog/search")
    session.allow_api_host("https://api.platform.example/ucp")
    assert session.post("https://api.platform.example/ucp/catalog/search").status_code == 200
    store.enabled = False
    with pytest.raises(PermissionError):
        session.post("https://api.platform.example/ucp/catalog/search")
