"""Evaluation harness (Inspect AI + custom scorers) over public benchmarks and C2A-Bench. M2."""

from c2a import NotYetImplemented

PUBLIC_BENCHMARKS = (
    "esci",
    "sqid",
    "amazon_m2",
    "shopping_mmlu",
    "ecinstruct",
    "shoppingbench",
    "shoppingcomp",
    "shopping_companion",
    "comboshoppingbench",
    "webmall",
    "shopgym",
    "tau_bench_retail",
)


def run(model: str, suites: list[str]) -> dict:
    raise NotYetImplemented("evaluation harness", "M2")
