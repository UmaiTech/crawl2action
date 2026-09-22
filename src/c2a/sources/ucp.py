"""UCP (Universal Commerce Protocol) catalog client. Planned for M1.

Discovery: GET https://<domain>/.well-known/ucp -> capabilities incl. catalog (REST/MCP).
Spec: https://ucp.dev/draft/specification/catalog/
"""

from __future__ import annotations

from c2a import NotYetImplemented
from c2a.schemas import Product, Store

WELL_KNOWN_PATH = "/.well-known/ucp"


def discover(store: Store) -> dict:
    raise NotYetImplemented("UCP discovery", "M1")


def search_catalog(store: Store, query: str, limit: int = 50) -> list[Product]:
    raise NotYetImplemented("UCP catalog search", "M1")
