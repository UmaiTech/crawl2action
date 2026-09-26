import httpx
import pytest
import respx

from c2a.schemas import TosStatus
from c2a.sources.compliance import RateLimiter, can_crawl, fetch_robots, parse_robots

from .conftest import FIXTURES


def rp():
    return parse_robots((FIXTURES / "robots.txt").read_text())


def test_can_crawl_gates(store):
    url = "https://shop.example.com/products/x"
    assert can_crawl(store, url, rp()) == (True, "ok")
    assert not can_crawl(store, "https://shop.example.com/checkout", rp())[0]
    assert not can_crawl(store, "https://other.example.com/products/x", rp())[0]
    assert not can_crawl(store, url, None)[0]  # robots unavailable -> deny
    store.tos_status = TosStatus.unreviewed
    assert can_crawl(store, url, rp()) == (False, "tos_status=unreviewed")


@respx.mock
def test_fetch_robots_statuses():
    respx.get("https://a.example/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://b.example/robots.txt").mock(return_value=httpx.Response(503))
    assert fetch_robots("a.example").can_fetch("x", "https://a.example/anything")
    assert fetch_robots("b.example") is None


def test_rate_limiter_spaces_requests():
    t = [0.0]
    sleeps = []
    rl = RateLimiter(qps=2, clock=lambda: t[0], sleep=sleeps.append)
    rl.wait("d")
    rl.wait("d")
    rl.wait("other")
    assert sleeps == [pytest.approx(0.5)]
