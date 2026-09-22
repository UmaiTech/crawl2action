from datetime import UTC, datetime

from c2a.datasets.splits import pick_holdout_stores, split
from c2a.schemas import TaskExample


def ex(i, store, day):
    return TaskExample(
        id=str(i),
        task="rec_pre",
        locale="en-US",
        store_id=store,
        created_at=datetime(2026, 9, day, tzinfo=UTC),
        payload={},
    )


def test_holdout_is_deterministic():
    ids = [f"s{i}" for i in range(100)]
    a, b = pick_holdout_stores(ids, 0.2), pick_holdout_stores(ids, 0.2)
    assert a == b and 5 < len(a) < 40


def test_split_by_store_and_time():
    out = split(
        [ex(1, "s1", 1), ex(2, "s2", 1), ex(3, "s1", 20)], {"s2"}, datetime(2026, 9, 15, tzinfo=UTC)
    )
    assert [e.id for e in out["train"]] == ["1"]
    assert sorted(e.id for e in out["test"]) == ["2", "3"]
    assert not {e.store_id for e in out["train"]} & {"s2"}
