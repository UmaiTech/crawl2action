"""Versioned System One question sets for labeling scraped data.

Each set has a version; changing any question means bumping the version so labels from
different question wordings are never mixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from c2a.decide.systemone import Choice, Noul, Question, Score

TAXONOMY_DIR = Path(__file__).with_name("taxonomy")


def verticals() -> dict[str, str]:
    return yaml.safe_load((TAXONOMY_DIR / "verticals.yaml").read_text())


@dataclass(frozen=True)
class QuestionSet:
    name: str
    version: str
    target: str  # page | product | pair
    questions: dict[str, Question] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"


def page_v1() -> QuestionSet:
    return QuestionSet(
        name="page",
        version="1",
        target="page",
        questions={
            "page_type": Choice(
                instructions="What kind of web page is this (scraped as markdown)?",
                criteria={
                    "product_detail": "A single product page with price and buy/add-to-cart",
                    "category_listing": "A grid or list of multiple products",
                    "content": "Blog, guide, about, help or policy page",
                    "other": "Anything else (cart, login, error, empty)",
                },
            )
        },
    )


def product_v1() -> QuestionSet:
    return QuestionSet(
        name="product",
        version="1",
        target="product",
        questions={
            "vertical": Choice(
                instructions="Which vertical does this product belong to?", criteria=verticals()
            ),
            "audience": Choice(
                instructions="Who is the primary intended user of this product?",
                criteria={
                    "women": None,
                    "men": None,
                    "unisex": "Adults of any gender",
                    "kids": "Children roughly 3-14",
                    "baby": "Infants and toddlers",
                    "pets": None,
                    "n_a": "Audience is not meaningful for this product",
                },
            ),
            "price_tier": Score(
                instructions="Price positioning relative to typical products of the same kind.",
                criteria=["budget", "mid_market", "premium", "luxury"],
            ),
            "use_occasion": Choice(
                instructions="Main use or occasion for this product.",
                criteria={
                    "everyday": None,
                    "sport_outdoor": None,
                    "work_office": None,
                    "formal_party": None,
                    "home_leisure": None,
                    "travel": None,
                    "gift": None,
                    "other": None,
                },
            ),
            "is_bundle": Noul(instructions="Is this listing a bundle, set or multipack?"),
            "restricted": Noul(
                instructions=(
                    "Is this an age-restricted or regulated product "
                    "(alcohol, tobacco/vape, weapons, drugs/medicines, adult content)?"
                )
            ),
            "contains_pii": Noul(
                instructions="Does the text contain personal data about a private individual "
                "(names with contact details, emails, phone numbers, addresses)?"
            ),
            "copy_quality": Score(
                instructions="Quality of the product title and description as shopping copy.",
                criteria=["poor", "fair", "good", "excellent"],
            ),
        },
    )


def pair_v1() -> QuestionSet:
    return QuestionSet(
        name="pair",
        version="1",
        target="pair",
        questions={
            "relation": Choice(
                instructions="How does the candidate product relate to the anchor "
                "(a shopper query or another product)?",
                criteria={
                    "exact": "Satisfies the anchor directly / same product",
                    "substitute": "An alternative the shopper could buy instead",
                    "complement": "Goes together with the anchor (buy both)",
                    "irrelevant": "Not related in a useful way",
                },
            )
        },
    )


QUESTION_SETS = {"page_v1": page_v1, "product_v1": product_v1, "pair_v1": pair_v1}


def get_question_set(name: str) -> QuestionSet:
    try:
        return QUESTION_SETS[name]()
    except KeyError:
        raise KeyError(
            f"unknown question set '{name}', choose from {sorted(QUESTION_SETS)}"
        ) from None
