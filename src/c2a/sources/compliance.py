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


def _store_gate(store: Store) -> tuple[bool, str]:
    if store.tos_status is not TosStatus.approved:
        return False, f"tos_status={store.tos_status.value}"
    if not store.enabled:
        return False, "store disabled"
    return True, "ok"


def can_crawl(store: Store, url: str, rp: robotparser.RobotFileParser | None) -> tuple[bool, str]:
    ok, why = _store_gate(store)
    if not ok:
        return ok, why
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


class CrawlSession:
    """The only way fetchers touch a store: every URL passes `can_crawl` and the rate limiter.

    robots.txt is fetched once per session (unavailable robots -> everything denied).
    """

    def __init__(
        self,
        store: Store,
        client: httpx.Client | None = None,
        limiter: RateLimiter | None = None,
        robots: robotparser.RobotFileParser | None = None,
        max_retries: int = 3,
        sleep=time.sleep,
    ) -> None:
        self.store = store
        self.max_retries = max_retries
        self._sleep = sleep
        self.retries = 0
        self.warnings: list[str] = []
        self.client = client or httpx.Client(
            timeout=20, headers={"User-Agent": USER_AGENT}, follow_redirects=True
        )
        self.limiter = limiter or RateLimiter(qps=1.0)
        self.robots = robots if robots is not None else fetch_robots(store.domain, self.client)
        self.denied: list[tuple[str, str]] = []
        # Hosts of APIs the store itself advertises for programmatic access (UCP profile).
        # They still need the store's ToS approval, but not a robots.txt entry.
        self.api_hosts: set[str] = set()

    def allow_api_host(self, url: str) -> None:
        self.api_hosts.add(urlparse(url).netloc)

    def allowed(self, url: str) -> bool:
        if urlparse(url).netloc in self.api_hosts:
            ok, why = _store_gate(self.store)
        else:
            ok, why = can_crawl(self.store, url, self.robots)
        if not ok:
            self.denied.append((url, why))
        return ok

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        if not self.allowed(url):
            raise PermissionError(f"crawl not allowed: {url} ({self.denied[-1][1]})")
        for attempt in range(self.max_retries + 1):
            self.limiter.wait(self.store.domain)
            resp = self.client.request(method, url, **kwargs)
            if resp.status_code not in RETRY_STATUS or attempt == self.max_retries:
                return resp
            self.retries += 1
            self._sleep(_retry_after(resp, default=2.0**attempt))
        raise AssertionError("unreachable")

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self._request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        """For UCP catalog APIs (same gates as GET)."""
        return self._request("POST", url, **kwargs)


RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_RETRY_AFTER = 60.0


def _retry_after(resp: httpx.Response, default: float) -> float:
    """Seconds from a numeric Retry-After header (capped), else the default backoff."""
    try:
        return min(float(resp.headers.get("Retry-After", "")), MAX_RETRY_AFTER)
    except ValueError:
        return default
