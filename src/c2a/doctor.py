"""Preflight checks before a real crawl + label run: keys, API reachability, approved stores,
robots.txt, platform/currency, and (with live=True) one real Jev and Firecrawl call."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import httpx
from pydantic import ValidationError

from c2a.config import Settings
from c2a.decide.systemone import Choice, SystemOneRequest, SystemOneResponse
from c2a.sources.compliance import USER_AGENT, fetch_robots, robots_allows
from c2a.sources.registry import Registry

Status = Literal["ok", "warn", "fail"]
FIRECRAWL_URL = "https://api.firecrawl.dev"


@dataclass
class Check:
    name: str
    status: Status
    detail: str = ""
    fix: str = ""


def _reachable(client: httpx.Client, url: str) -> tuple[bool, str]:
    try:
        resp = client.get(url, timeout=10)
    except httpx.HTTPError as exc:
        return False, type(exc).__name__
    return True, f"HTTP {resp.status_code}"


def run_doctor(
    registry: Registry,
    settings: Settings,
    live: bool = False,
    client: httpx.Client | None = None,
) -> list[Check]:
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True)
    checks: list[Check] = []
    stores = registry.crawlable()
    needs_firecrawl = any(s.method == "firecrawl" for s in stores)

    # keys
    jev_key = os.environ.get("TYPESAFE_API_KEY")
    checks.append(
        Check("TYPESAFE_API_KEY", "ok", "set")
        if jev_key
        else Check("TYPESAFE_API_KEY", "fail", "not set", "add TYPESAFE_API_KEY=... to .env")
    )
    fc_key = os.environ.get("FIRECRAWL_API_KEY")
    if fc_key:
        checks.append(Check("FIRECRAWL_API_KEY", "ok", "set"))
    else:
        checks.append(
            Check(
                "FIRECRAWL_API_KEY",
                "fail" if needs_firecrawl else "warn",
                "not set" + ("" if needs_firecrawl else " (only needed for non-Shopify stores)"),
                "add FIRECRAWL_API_KEY=... to .env",
            )
        )

    # reachability
    for name, url in (("Jev API", settings.labeler.base_url), ("Firecrawl API", FIRECRAWL_URL)):
        ok, detail = _reachable(client, url)
        status: Status = (
            "ok" if ok else ("warn" if name == "Firecrawl API" and not needs_firecrawl else "fail")
        )
        checks.append(
            Check(
                f"{name} reachable", status, f"{url}: {detail}", "" if ok else "check network/proxy"
            )
        )

    # stores
    if not stores:
        checks.append(
            Check(
                "approved stores",
                "fail",
                "0 stores are approved and enabled",
                'after reviewing its terms: c2a registry approve <id> --by "<name>"',
            )
        )
    else:
        checks.append(Check("approved stores", "ok", ", ".join(s.id for s in stores)))
    for s in stores:
        base = f"https://{s.domain}"
        rp = fetch_robots(s.domain, client)
        if rp is None:
            checks.append(
                Check(
                    f"{s.id}: robots.txt",
                    "fail",
                    "unreachable (crawl would deny all)",
                    "check the domain in the registry",
                )
            )
            continue
        if s.method == "shopify":
            allowed = robots_allows(rp, base + "/products.json")
            checks.append(
                Check(
                    f"{s.id}: /products.json allowed",
                    "ok" if allowed else "fail",
                    "robots.txt allows" if allowed else "robots.txt disallows",
                    "" if allowed else "use method firecrawl or drop this store",
                )
            )
            checks.append(
                Check(f"{s.id}: currency", "ok", s.currency)
                if s.currency
                else Check(
                    f"{s.id}: currency",
                    "warn",
                    "not set (will try /meta.json, then the country default)",
                    f"c2a discover fingerprint --store {s.id} --write",
                )
            )
        elif s.platform == "unknown":
            checks.append(
                Check(
                    f"{s.id}: platform",
                    "warn",
                    f"unknown (method={s.method})",
                    f"c2a discover fingerprint --store {s.id} --write",
                )
            )

    if live:
        checks += _live_checks(registry, settings, client, jev_key, fc_key)
    return checks


def _live_checks(registry, settings, client, jev_key, fc_key) -> list[Check]:
    out: list[Check] = []
    if jev_key:
        req = SystemOneRequest(
            state={"title": "Merino wool running shoe", "price": "129.00 USD"},
            model=settings.labeler.model,
            questions={
                "vertical": Choice(
                    instructions="Which vertical?", criteria={"apparel": None, "electronics": None}
                )
            },
        )
        try:
            resp = client.post(
                settings.labeler.base_url.rstrip("/") + "/v1/systemone",
                json=req.model_dump(mode="json", exclude_none=True),
                headers={"Authorization": f"Bearer {jev_key}"},
                timeout=30,
            )
        except httpx.HTTPError as exc:
            out.append(Check("Jev live call", "fail", type(exc).__name__))
        else:
            if resp.status_code != 200:
                out.append(
                    Check(
                        "Jev live call",
                        "fail",
                        f"HTTP {resp.status_code}: {resp.text[:300]}",
                        "check the key and base_url",
                    )
                )
            else:
                try:
                    parsed = SystemOneResponse.model_validate(resp.json())
                    ans = parsed.answers["vertical"]
                    out.append(
                        Check(
                            "Jev live call",
                            "ok",
                            f"answer={ans.choice} confidence={ans.confidence:.2f}",
                        )
                    )
                except (ValidationError, ValueError, KeyError, AttributeError) as exc:
                    out.append(
                        Check(
                            "Jev response format",
                            "fail",
                            f"{type(exc).__name__}; raw body: {resp.text[:800]}",
                            "send this output back; decide/systemone.py needs updating",
                        )
                    )
    fc_store = next((s for s in registry.crawlable() if s.method == "firecrawl"), None)
    if fc_key and fc_store:
        try:
            resp = client.post(
                FIRECRAWL_URL + "/v2/map",
                json={"url": f"https://{fc_store.domain}", "limit": 5},
                headers={"Authorization": f"Bearer {fc_key}"},
                timeout=60,
            )
            body = resp.text[:400]
            ok = resp.status_code == 200 and "links" in resp.json()
            out.append(
                Check(
                    "Firecrawl live map",
                    "ok" if ok else "fail",
                    f"HTTP {resp.status_code}" if ok else body,
                )
            )
        except (httpx.HTTPError, ValueError) as exc:
            out.append(Check("Firecrawl live map", "fail", type(exc).__name__))
    return out


ICON = {"ok": "✅", "warn": "⚠️ ", "fail": "❌"}


def format_checks(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        lines.append(f"{ICON[c.status]} {c.name}: {c.detail}")
        if c.fix and c.status != "ok":
            lines.append(f"     fix: {c.fix}")
    return "\n".join(lines)
