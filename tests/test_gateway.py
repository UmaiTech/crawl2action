from fastapi.testclient import TestClient

from c2a.index import InMemoryRetriever
from c2a.schemas import Money, Product, RecItem, Variant
from c2a.serve.gateway import create_app


def product(pid, title, amount, available=True):
    return Product(
        id=f"s:{pid}",
        store_id="s",
        native_id=pid,
        title=title,
        variants=[Variant(id=pid, price=Money(amount=amount, currency="SEK"), available=available)],
    )


PRODUCTS = [
    product("1", "wool socks", 100),
    product("2", "wool sweater", 900),
    product("3", "wool hat", 50, False),
]


def test_recommend_filters_constraints():
    client = TestClient(create_app(InMemoryRetriever(PRODUCTS)))
    res = client.post(
        "/recommend",
        json={
            "stage": "pre_checkout",
            "query": "wool",
            "constraints": {"max_price": {"amount": 500, "currency": "SEK"}},
        },
    )
    assert res.status_code == 200
    assert [i["product_id"] for i in res.json()["items"]] == ["s:1"]


def test_recommend_rejects_hallucinating_ranker():
    app = create_app(
        InMemoryRetriever(PRODUCTS), ranker=lambda req, c: [RecItem(product_id="nope")]
    )
    res = TestClient(app).post("/recommend", json={"stage": "pre_checkout", "query": "wool"})
    assert res.status_code == 502


def test_health_and_decide():
    client = TestClient(create_app(InMemoryRetriever(PRODUCTS)))
    assert client.get("/health").json() == {"status": "ok"}
    res = client.post("/decide", json={"evidence": "e", "question": "q", "labels": ["x", "y"]})
    assert res.status_code == 200 and abs(sum(res.json()["probs"].values()) - 1) < 1e-9
