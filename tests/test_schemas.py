import pytest
from pydantic import ValidationError

from c2a.schemas import DecisionRequest, DecisionResult, Money, Product, Variant


def test_money_uppercases_currency():
    assert Money(amount=100, currency="sek").currency == "SEK"


def test_product_requires_variant_and_helpers():
    with pytest.raises(ValidationError):
        Product(id="s:1", store_id="s", native_id="1", title="x", variants=[])
    p = Product(
        id="s:1",
        store_id="s",
        native_id="1",
        title="x",
        variants=[
            Variant(id="1", price=Money(amount=500, currency="EUR"), available=False),
            Variant(id="2", price=Money(amount=300, currency="EUR"), available=True),
        ],
    )
    assert p.min_price.amount == 300 and p.in_stock


def test_decision_schemas_validate():
    with pytest.raises(ValidationError):
        DecisionRequest(evidence="e", question="q", labels=["a", "a"])
    with pytest.raises(ValidationError):
        DecisionResult(label="a", probs={"a": 0.5, "b": 0.4})
    DecisionResult(label="a", probs={"a": 0.6, "b": 0.4})
