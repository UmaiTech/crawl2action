import httpx
import respx

from c2a.sources import ucp
from c2a.sources.compliance import CrawlSession, RateLimiter, parse_robots

PROFILE = {
    "ucp": {
        "version": "2026-01-01",
        "services": {
            "dev.ucp.shopping": [
                {"version": "2026-01-01", "transport": "mcp", "endpoint": "https://x/mcp"},
                {
                    "version": "2026-01-01",
                    "transport": "rest",
                    "endpoint": "https://api.shop.test/ucp/",
                },
            ]
        },
        "capabilities": {
            "dev.ucp.shopping.catalog.search": [{"version": "2026-01-01"}],
            "dev.ucp.shopping.catalog.lookup": [{"version": "2026-01-01"}],
        },
        "payment_handlers": {},
    }
}

# From the UCP spec (catalog REST binding example), trimmed.
PRODUCT = {
    "id": "prod_abc123",
    "handle": "blue-runner-pro",
    "title": "Blue Runner Pro",
    "description": {"plain": "Lightweight running shoes with responsive cushioning."},
    "url": "https://shop.example.com/products/blue-runner-pro",
    "categories": [
        {"value": "187", "taxonomy": "google_product_category"},
        {"value": "Footwear > Running", "taxonomy": "merchant"},
    ],
    "price_range": {
        "min": {"amount": 12000, "currency": "USD"},
        "max": {"amount": 12000, "currency": "USD"},
    },
    "media": [{"type": "image", "url": "https://cdn.example.com/brp.jpg"}],
    "variants": [
        {
            "id": "prod_abc123_size10",
            "sku": "BRP-BLU-10",
            "title": "Size 10",
            "price": {"amount": 12000, "currency": "USD"},
            "availability": {"available": True},
            "options": [{"name": "Size", "label": "10"}],
            "tags": ["running", "road"],
        }
    ],
}


def session(store):
    return CrawlSession(
        store,
        client=httpx.Client(),
        limiter=RateLimiter(1000, sleep=lambda s: None),
        robots=parse_robots(""),
    )


def test_parse_profile_prefers_rest():
    p = ucp.parse_profile(PROFILE)
    assert p.endpoint == "https://api.shop.test/ucp" and p.can_search and p.can_lookup
    assert ucp.parse_profile({"ucp": {"services": {}}}) is None


def test_parse_ucp_product(store):
    p = ucp.parse_ucp_product(PRODUCT, store)
    assert p.id == "acme:prod_abc123" and p.source == "ucp"
    v = p.variants[0]
    assert v.price.amount == 12000 and v.price.currency == "USD" and v.options == {"Size": "10"}
    assert (
        p.category == "Footwear > Running"
        and p.attributes["category_google_product_category"] == "187"
    )
    assert p.image_urls == ["https://cdn.example.com/brp.jpg"] and p.tags == ["road", "running"]
    bare = {k: v for k, v in PRODUCT.items() if k != "variants"}
    assert ucp.parse_ucp_product(bare, store).variants[0].price.amount == 12000


@respx.mock
def test_discover_and_paginated_catalog(store):
    respx.get("https://shop.example.com/.well-known/ucp").mock(
        return_value=httpx.Response(200, json=PROFILE)
    )
    p2 = {**PRODUCT, "id": "prod_2"}
    route = respx.post("https://api.shop.test/ucp/catalog/search").mock(
        side_effect=[
            httpx.Response(
                200,
                json={"products": [PRODUCT], "pagination": {"cursor": "c1", "has_next_page": True}},
            ),
            httpx.Response(
                200, json={"products": [p2, PRODUCT], "pagination": {"has_next_page": False}}
            ),
        ]
    )
    s = session(store)
    profile = ucp.discover(s)
    products, errors = ucp.iter_catalog(s, profile, ["shoes"])
    assert [p.native_id for p in products] == ["prod_abc123", "prod_2"] and errors == []
    assert b'"cursor":"c1"' in route.calls[1].request.content.replace(b" ", b"")
