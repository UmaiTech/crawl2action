from c2a.sources.shopify import parse_products_json


def test_parse_products_json(store, shopify_payload):
    products, errors = parse_products_json(shopify_payload, store, "SEK")
    assert [p.id for p in products] == ["acme:101", "acme:102"]
    assert len(errors) == 1 and errors[0].startswith("103")
    runner, socks = products
    v = runner.variants[0]
    assert v.price.amount == 12900 and v.compare_at_price.amount == 14900
    assert v.options == {"Size": "42", "Color": "Grey"}
    assert runner.variants[1].sku is None and not runner.variants[1].available
    assert str(runner.url) == "https://shop.example.com/products/merino-runner"
    assert socks.tags == ["socks", "wool"] and socks.variants[0].price.amount == 1999
    assert runner.source == "shopify" and runner.locale == "sv-SE"
