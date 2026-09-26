"""Store discovery: platform fingerprinting and candidate expansion via Firecrawl search.

Fingerprinting only reads public, lightweight endpoints (robots-checked) and may only update
descriptive registry fields (platform/method). Expansion writes *candidates* for human
review; it never edits the registry and never approves anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from c2a.schemas import Store, Tier, TosStatus
from c2a.sources.compliance import USER_AGENT, fetch_robots, robots_allows
from c2a.sources.firecrawl import FirecrawlClient
from c2a.sources.ucp import WELL_KNOWN_PATH, parse_profile

METHOD_FOR_PLATFORM = {"ucp": "ucp", "shopify": "shopify"}  # everything else -> firecrawl

# Localized search phrasing per country (discovery queries only).
QUERY_TEMPLATES = {
    "SE": "bästa nätbutik {category} Sverige",
    "GB": "best online shop {category} UK",
    "ES": "mejor tienda online {category} España",
    "US": "best online store {category} USA",
    "CA": "best online store {category} Canada",
}
LOCALE_FOR_COUNTRY = {"SE": "sv-SE", "GB": "en-GB", "ES": "es-ES", "US": "en-US", "CA": "en-CA"}
EXCLUDE_DOMAINS = re.compile(
    r"(amazon|ebay|google|facebook|instagram|youtube|reddit|wikipedia|pinterest|tiktok|"
    r"trustpilot|pricerunner|prisjakt|idealo)\.",
    re.I,
)


@dataclass
class Fingerprint:
    domain: str
    platform: str  # ucp | shopify | woocommerce | custom | unknown
    method: str  # ucp | shopify | firecrawl
    evidence: str
    currency: str | None = None  # Shopify /meta.json currency when available


def _get(client: httpx.Client, url: str) -> httpx.Response | None:
    try:
        return client.get(url, follow_redirects=True)
    except httpx.HTTPError:
        return None


def fingerprint(domain: str, client: httpx.Client | None = None) -> Fingerprint:
    own = client is None
    client = client or httpx.Client(timeout=15, headers={"User-Agent": USER_AGENT})
    try:
        rp = fetch_robots(domain, client)
        base = f"https://{domain}"
        resp = _get(client, base + WELL_KNOWN_PATH)
        if resp is not None and resp.status_code == 200:
            try:
                if parse_profile(resp.json()) is not None:
                    return Fingerprint(domain, "ucp", "ucp", WELL_KNOWN_PATH)
            except ValueError:
                pass
        if robots_allows(rp, base + "/products.json"):
            resp = _get(client, base + "/products.json?limit=1")
            if resp is not None and resp.status_code == 200:
                try:
                    if "products" in resp.json():
                        return Fingerprint(
                            domain,
                            "shopify",
                            "shopify",
                            "/products.json",
                            _shopify_currency(client, base, rp),
                        )
                except ValueError:
                    pass
        if not robots_allows(rp, base + "/"):
            return Fingerprint(domain, "unknown", "firecrawl", "robots disallows homepage")
        resp = _get(client, base + "/")
        if resp is None or resp.status_code != 200:
            return Fingerprint(domain, "unknown", "firecrawl", "homepage unreachable")
        html = resp.text.lower()
        if "cdn.shopify.com" in html or "shopify.theme" in html:
            return Fingerprint(
                domain, "shopify", "firecrawl", "shopify markers; products.json closed"
            )
        if "woocommerce" in html or "wp-content" in html:
            return Fingerprint(domain, "woocommerce", "firecrawl", "woocommerce markers")
        return Fingerprint(domain, "custom", "firecrawl", "no known platform markers")
    finally:
        if own:
            client.close()


def _shopify_currency(client: httpx.Client, base: str, rp) -> str | None:
    if not robots_allows(rp, base + "/meta.json"):
        return None
    resp = _get(client, base + "/meta.json")
    if resp is None or resp.status_code != 200:
        return None
    try:
        return resp.json().get("currency") or None
    except ValueError:
        return None


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def expand(
    country: str,
    category: str,
    firecrawl: FirecrawlClient,
    known_domains: set[str],
    limit: int = 20,
) -> list[Store]:
    """Candidate stores from web search. Always tier B, unreviewed and disabled."""
    template = QUERY_TEMPLATES.get(country.upper())
    if template is None:
        raise ValueError(f"no query template for country {country}")
    results = firecrawl.search(template.format(category=category), limit=limit)
    out: dict[str, Store] = {}
    for r in results:
        domain = _domain(r["url"])
        if not domain or domain in known_domains or domain in out or EXCLUDE_DOMAINS.search(domain):
            continue
        slug = re.sub(r"[^a-z0-9]+", "", domain.removeprefix("www.").split(".")[0])
        out[domain] = Store(
            id=slug or domain,
            name=(r.get("title") or domain)[:80],
            domain=domain,
            country=country.upper(),
            locale=LOCALE_FOR_COUNTRY[country.upper()],
            tier=Tier.B,
            tos_status=TosStatus.unreviewed,
            enabled=False,
            verticals=[category],
            notes=f"discovered via search: {r.get('description', '')[:160]}",
        )
    return list(out.values())
