"""Core data model. Aligned with the UCP catalog (products/variants, minor-unit prices)
and schema.org Product so crawled, UCP and Shopify data normalize to one shape."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


class Money(BaseModel):
    """Price in integer minor units (e.g. cents/öre) plus ISO-4217 currency."""

    amount: int = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)

    @field_validator("currency")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


class Variant(BaseModel):
    id: str
    title: str = ""
    sku: str | None = None
    options: dict[str, str] = Field(default_factory=dict)
    price: Money
    compare_at_price: Money | None = None
    available: bool = True


class Product(BaseModel):
    id: str  # globally unique: "<store_id>:<native_id>"
    store_id: str
    native_id: str
    title: str
    description: str = ""
    brand: str | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)
    url: HttpUrl | None = None
    image_urls: list[str] = Field(default_factory=list)
    variants: list[Variant] = Field(min_length=1)
    locale: str | None = None
    source: str = "unknown"  # shopify | ucp | firecrawl | open_dataset
    crawled_at: datetime | None = None

    @property
    def min_price(self) -> Money:
        return min((v.price for v in self.variants), key=lambda m: m.amount)

    @property
    def in_stock(self) -> bool:
        return any(v.available for v in self.variants)


class Review(BaseModel):
    product_id: str
    rating: float | None = Field(default=None, ge=0, le=5)
    text: str = ""
    locale: str | None = None


class Tier(StrEnum):
    A = "A"  # structured / permissioned: UCP, Shopify products.json, open datasets
    B = "B"  # Firecrawl extract, robots + ToS permitting
    C = "C"  # marketplaces: off unless explicitly cleared


class TosStatus(StrEnum):
    unreviewed = "unreviewed"
    approved = "approved"
    denied = "denied"


class Store(BaseModel):
    id: str
    name: str
    domain: str
    country: str = Field(min_length=2, max_length=2)
    locale: str
    platform: str = "unknown"  # shopify | ucp | woocommerce | custom | unknown
    tier: Tier
    tos_status: TosStatus = TosStatus.unreviewed
    method: str = "firecrawl"  # shopify | ucp | firecrawl
    enabled: bool = False
    verticals: list[str] = Field(default_factory=list)
    notes: str = ""


# ---------- Task I/O ----------


class Constraints(BaseModel):
    max_price: Money | None = None
    in_stock_only: bool = True
    sizes: list[str] = Field(default_factory=list)
    locale: str | None = None


class Candidate(BaseModel):
    product_id: str
    title: str
    price: Money
    in_stock: bool = True
    attributes: dict[str, str] = Field(default_factory=dict)


class RecRequest(BaseModel):
    stage: str = Field(pattern="^(pre_checkout|post_checkout)$")
    query: str | None = None
    context_product_ids: list[str] = Field(default_factory=list)  # anchor/cart/order
    candidates: list[Candidate] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    locale: str = "en-US"
    k: int = Field(default=10, ge=1, le=50)


class RecItem(BaseModel):
    product_id: str
    reason: str = ""


class RecResponse(BaseModel):
    items: list[RecItem]


class CopyRequest(BaseModel):
    product: Product
    kind: str = Field(pattern="^(title|description|bullets|seo|ad)$")
    locale: str = "en-US"
    brand_voice: str | None = None


class CopyResponse(BaseModel):
    text: str
    claims: list[str] = Field(default_factory=list)


class ImageBrief(BaseModel):
    product_id: str
    use_case: str = Field(pattern="^(hero|lifestyle|banner|edit)$")
    scene: str
    style: str = ""
    aspect_ratio: str = "1:1"
    reference_image_urls: list[str] = Field(default_factory=list)
    negative_prompt: str = ""


class DecisionRequest(BaseModel):
    evidence: str
    question: str
    labels: list[str] = Field(min_length=2)

    @field_validator("labels")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("labels must be unique")
        return v


class DecisionResult(BaseModel):
    label: str
    probs: dict[str, float]

    @model_validator(mode="after")
    def _check(self) -> DecisionResult:
        total = sum(self.probs.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probabilities must sum to 1, got {total}")
        if self.label not in self.probs:
            raise ValueError("label must be one of probs")
        return self


class TaskExample(BaseModel):
    """One dataset item for any task family; `payload` holds the typed request."""

    id: str
    task: str  # rec_pre | rec_post | copy | localize | image_brief | decide | agentic
    locale: str
    store_id: str | None = None
    created_at: datetime | None = None
    payload: dict[str, Any]
    ground_truth: dict[str, Any] = Field(default_factory=dict)
