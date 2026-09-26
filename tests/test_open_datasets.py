import json

import pyarrow as pa
import pyarrow.parquet as pq

from c2a.labeling.export import to_decider_train
from c2a.labeling.questions import pair_v1
from c2a.sources import open_datasets as od


def write_esci(tmp_path):
    ex = tmp_path / "examples.parquet"
    pr = tmp_path / "products.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "example_id": 1,
                    "query": "wool socks",
                    "query_id": 7,
                    "product_id": "B1",
                    "product_locale": "us",
                    "esci_label": "E",
                    "small_version": 1,
                    "large_version": 1,
                    "split": "train",
                },
                {
                    "example_id": 2,
                    "query": "wool socks",
                    "query_id": 7,
                    "product_id": "B2",
                    "product_locale": "us",
                    "esci_label": "C",
                    "small_version": 1,
                    "large_version": 1,
                    "split": "test",
                },
                {
                    "example_id": 3,
                    "query": "calcetines",
                    "query_id": 8,
                    "product_id": "B3",
                    "product_locale": "es",
                    "esci_label": "S",
                    "small_version": 0,
                    "large_version": 1,
                    "split": "train",
                },
            ]
        ),
        ex,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "product_id": "B1",
                    "product_title": "Merino socks",
                    "product_description": None,
                    "product_bullet_point": "Warm",
                    "product_brand": "Acme",
                    "product_color": "grey",
                    "product_locale": "us",
                },
                {
                    "product_id": "B2",
                    "product_title": "Boot dryer",
                    "product_description": "Dries boots",
                    "product_bullet_point": None,
                    "product_brand": None,
                    "product_color": None,
                    "product_locale": "us",
                },
            ]
        ),
        pr,
    )
    return ex, pr


def test_esci_to_human_pair_labels(tmp_path):
    ex, pr = write_esci(tmp_path)
    rows = list(od.load_esci(ex, pr))
    assert [(r.product_id, r.label, r.locale) for r in rows] == [
        ("B1", "exact", "en-US"),
        ("B2", "complement", "en-US"),
    ]
    assert rows[0].product == {
        "title": "Merino socks",
        "bullet_point": "Warm",
        "brand": "Acme",
        "color": "grey",
    }
    assert len(list(od.load_esci(ex, pr, small_version_only=False))) == 3
    items, records = od.esci_to_labels(rows)
    assert all(r.source == "human" and r.qset == "pair@1" for r in records)
    train = to_decider_train({i.id: i.state for i in items}, records, pair_v1())
    assert train[1]["questions"]["relation"]["label"] == "complement"
    assert train[0]["state"]["anchor"] == {"query": "wool socks"}


def test_amazon_m2_sessions(tmp_path):
    p = tmp_path / "sessions.csv"
    p.write_text(
        "prev_items,next_item,locale\n"
        "\"['B09W9FND7K' 'B09JSPLN1M']\",B076THCGSG,UK\n"
        "\"['X1']\",X2,DE\n"
    )
    rows = list(od.load_amazon_m2(p, locales=["UK"]))
    assert len(rows) == 1 and rows[0].prev_items == ["B09W9FND7K", "B09JSPLN1M"]
    assert rows[0].next_item == "B076THCGSG" and rows[0].locale == "en-GB"


def test_amazon_reviews_and_histories(tmp_path):
    p = tmp_path / "reviews.jsonl"
    lines = [
        {
            "rating": 5.0,
            "title": "Great",
            "text": "Love it",
            "asin": "A1",
            "parent_asin": "P1",
            "user_id": "u1",
            "timestamp": 200,
            "verified_purchase": True,
        },
        {
            "rating": 2.0,
            "title": "",
            "text": "Meh",
            "asin": "A2",
            "parent_asin": "P2",
            "user_id": "u1",
            "timestamp": 100,
            "verified_purchase": True,
        },
        {
            "rating": 4.0,
            "text": "ok",
            "asin": "A3",
            "user_id": "u2",
            "timestamp": 1,
            "verified_purchase": False,
        },
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines))
    reviews = list(od.load_amazon_reviews(p))
    assert reviews[0].product_id == "amazon:P1" and reviews[0].text == "Great Love it"
    assert od.user_histories(p) == {"u1": ["P2", "P1"]}
