"""C2A-Bench: our versioned benchmark (see docs/PLAN.md section 4b). Builder lands in M2."""

from __future__ import annotations

from dataclasses import dataclass

# Embedded in every benchmark file so leakage into training corpora is detectable.
CANARY = "C2A-BENCH CANARY 3f6c1e2a-crawl2action-do-not-train"


@dataclass(frozen=True)
class Track:
    id: str
    title: str
    metrics: tuple[str, ...]


TRACKS: tuple[Track, ...] = (
    Track(
        "rec_pre",
        "Pre-checkout recommendations",
        ("ndcg@10", "recall@10", "constraint_violations", "hallucinated_id_rate"),
    ),
    Track(
        "rec_post",
        "Post-checkout cross-sell",
        ("complement_precision", "substitute_confusion", "recall@k"),
    ),
    Track(
        "copy",
        "Product copy",
        ("attribute_faithfulness", "locale_accuracy", "rubric_score", "human_win_rate"),
    ),
    Track(
        "localize", "Localization", ("attribute_preservation", "locale_accuracy", "rubric_score")
    ),
    Track("image", "Image briefs + renders", ("vlm_fidelity", "text_accuracy", "human_win_rate")),
    Track("agentic", "Conversational shopping", ("task_success", "constraint_adherence", "turns")),
    Track(
        "decide",
        "Calibrated decisions",
        ("weighted_accuracy", "ece", "brier", "router_cost_saving"),
    ),
)


def contains_canary(text: str) -> bool:
    return CANARY in text
