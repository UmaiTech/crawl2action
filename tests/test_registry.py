from c2a.schemas import TosStatus
from c2a.sources.registry import load_registry


def test_seed_registry_is_valid_and_default_deny():
    reg = load_registry()
    assert len(reg.stores) >= 25
    assert {s.country for s in reg.stores} == {"SE", "GB", "ES", "US", "CA"}
    assert all(not s.enabled and s.tos_status is TosStatus.unreviewed for s in reg.stores)
    assert reg.crawlable() == []


def test_crawlable_requires_enabled_and_approved():
    reg = load_registry()
    s = reg.stores[0]
    s.enabled = True
    assert reg.crawlable() == []  # still unreviewed
    s.tos_status = TosStatus.approved
    assert reg.crawlable() == [s]
