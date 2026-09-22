"""Crawl compliance: ToS gate, robots.txt and per-domain rate limiting.

Default-deny: a store is crawlable only if its ToS status is approved AND robots.txt
allows the path for our user agent. Unknown/unreachable robots.txt denies.
"""

from __future__ import annotations

import threading
import time
from urllib import robotparser
from urllib.parse import urlparse

import httpx

from c2a.schemas import Store, TosStatus

USER_AGENT = "crawl2action-bot/0.1 (+https://github.com/umaitech/crawl2action)"


def parse_robots(text: str) -> robotparser.RobotFileParser:
    rp = robotparser.RobotFileParser()
    rp.parse(text.splitlines())
    return rp


def fetch_robots(
    domain: str, client: httpx.Client | None = None
) -> robotparser.RobotFileParser | None:
    own = client is None
    client = client or httpx.Client(timeout=10, headers={"User-Agent": USER_AGENT})
    try:
        resp = client.get(f"https://{domain}/robots.txt", follow_redirects=True)
    except httpx.HTTPError:
        return None
    finally:
        if own:
            client.close()
    if resp.status_code == 404:
        return parse_robots("")  # no robots.txt: allowed by convention
    if resp.status_code != 200:
        return None
    return parse_robots(resp.text)


def robots_allows(rp: robotparser.RobotFileParser | None, url: str) -> bool:
    return rp is not None and rp.can_fetch(USER_AGENT, url)


def can_crawl(store: Store, url: str, rp: robotparser.RobotFileParser | None) -> tuple[bool, str]:
    if store.tos_status is not TosStatus.approved:
        return False, f"tos_status={store.tos_status.value}"
    if not store.enabled:
        return False, "store disabled"
    if urlparse(url).netloc != store.domain:
        return False, "url outside store domain"
    if not robots_allows(rp, url):
        return False, "robots.txt disallows (or unavailable)"
    return True, "ok"


class RateLimiter:
    """Simple per-domain minimum-interval limiter (thread-safe)."""

    def __init__(self, qps: float, clock=time.monotonic, sleep=time.sleep) -> None:
        if qps <= 0:
            raise ValueError("qps must be > 0")
        self.interval = 1.0 / qps
        self._next: dict[str, float] = {}
        self._lock = threading.Lock()
        self._clock, self._sleep = clock, sleep

    def wait(self, domain: str) -> None:
        with self._lock:
            now = self._clock()
            start = max(now, self._next.get(domain, now))
            self._next[domain] = start + self.interval
        if start > now:
            self._sleep(start - now)
