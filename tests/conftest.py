import json
from pathlib import Path

import pytest

from c2a.schemas import Store, Tier, TosStatus

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def store() -> Store:
    return Store(
        id="acme",
        name="Acme",
        domain="shop.example.com",
        country="SE",
        locale="sv-SE",
        platform="shopify",
        tier=Tier.A,
        tos_status=TosStatus.approved,
        method="shopify",
        enabled=True,
    )


@pytest.fixture
def shopify_payload() -> dict:
    return json.loads((FIXTURES / "shopify_products.json").read_text())


@pytest.fixture
def rec_example() -> dict:
    def cand(pid, amount, in_stock=True):
        return {
            "product_id": pid,
            "title": pid,
            "price": {"amount": amount, "currency": "SEK"},
            "in_stock": in_stock,
        }

    return {
        "candidates": [
            cand("a", 100),
            cand("b", 200),
            cand("c", 300),
            cand("d", 50, in_stock=False),
        ],
        "constraints": {"max_price": {"amount": 250, "currency": "SEK"}, "in_stock_only": True},
        "relevance": {"a": 2, "b": 1},
        "k": 3,
    }
