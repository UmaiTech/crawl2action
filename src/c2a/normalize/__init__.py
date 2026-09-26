"""Normalization helpers: money, text. FX, taxonomy mapping and near-dup detection land in M1."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

# ISO-4217 minor-unit exponents for currencies we expect; default is 2.
_EXPONENTS = {"JPY": 0, "KRW": 0, "ISK": 0, "CLP": 0, "BHD": 3, "KWD": 3}


def currency_exponent(currency: str) -> int:
    return _EXPONENTS.get(currency.upper(), 2)


def to_minor_units(amount: Decimal | str | float, currency: str) -> int:
    value = Decimal(str(amount))
    if value < 0:
        raise ValueError("negative price")
    scaled = value * (Decimal(10) ** currency_exponent(currency))
    return int(scaled.quantize(Decimal(1), rounding=ROUND_HALF_UP))


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(html: str) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", html)).strip()
