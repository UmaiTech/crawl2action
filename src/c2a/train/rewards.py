"""RL rewards and advantage estimators. Rewards reuse the graders, so eval and RL agree.

Advantages
- grpo:     (r - mean) / (std + eps) per group
- dr_grpo:  r - mean per group (no std/length normalization; Dr.GRPO)
- rank_grpo: per-rank advantages for ranked lists (Rank-GRPO-style): each rank position k
  gets the normalized DCG reward-to-go from k onward, group-centered per position.
Dynamic sampling (DAPO): groups with zero reward variance carry no signal and are dropped.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError

from c2a.graders.ranking import dedupe_preserving_order
from c2a.graders.rec import RecGrader, hallucinated_ids, parse_rec_output
from c2a.schemas import Candidate, CopyResponse, DecisionResult

RewardFn = Callable[[str, dict[str, Any], float], float]


def rec_reward(output: str, example: dict[str, Any], fail_penalty: float) -> float:
    res = RecGrader(k=example.get("k", 10)).grade(output, example)
    return res.score if res.passed else fail_penalty


def decide_reward(output: str, example: dict[str, Any], fail_penalty: float) -> float:
    """Calibration-aware: 1 - Brier/2 in [0, 1]. Output must be a DecisionResult JSON
    over exactly the example's labels."""
    try:
        res = DecisionResult.model_validate(json.loads(output))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return fail_penalty
    if set(res.probs) != set(example["labels"]):
        return fail_penalty
    truth = example["label"]
    b = sum((p - (1.0 if lbl == truth else 0.0)) ** 2 for lbl, p in res.probs.items())
    return 1.0 - b / 2.0


def copy_reward(output: str, example: dict[str, Any], fail_penalty: float) -> float:
    """Placeholder faithfulness reward until the post-trained decider/GenRM lands (M2):
    fraction of claims whose numbers are all present in the product attributes."""
    try:
        res = CopyResponse.model_validate(json.loads(output))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return fail_penalty
    if not res.text.strip():
        return fail_penalty
    if not res.claims:
        return 0.5
    attrs = " ".join(f"{k} {v}" for k, v in example.get("attributes", {}).items()).lower()
    supported = 0
    for claim in res.claims:
        numbers = re.findall(r"\d+(?:[.,]\d+)?", claim)
        if all(n in attrs for n in numbers):
            supported += 1
    return supported / len(res.claims)


REWARD_FNS: dict[str, RewardFn] = {
    "rec": rec_reward,
    "decide": decide_reward,
    "copy": copy_reward,
}


def group_advantages(
    rewards: Sequence[float], mode: str = "dr_grpo", eps: float = 1e-6
) -> list[float]:
    if not rewards:
        return []
    mean = statistics.fmean(rewards)
    centered = [r - mean for r in rewards]
    if mode == "dr_grpo":
        return centered
    if mode == "grpo":
        std = statistics.pstdev(rewards)
        return [c / (std + eps) for c in centered]
    raise ValueError(f"unknown advantage mode: {mode}")


def has_signal(rewards: Sequence[float], tol: float = 1e-9) -> bool:
    return len(rewards) > 1 and (max(rewards) - min(rewards)) > tol


def rank_rewards_to_go(
    output: str, example: dict[str, Any], k: int = 10, fail_penalty: float = -0.5
) -> list[float]:
    """Per-rank normalized DCG reward-to-go for one ranked output.

    Position i gets sum_{j>=i} gain_j / IDCG@k, so credit only flows from a rank to the
    items at or after it (no non-causal credit). Hallucinated items get `fail_penalty`
    as their gain; invalid output returns [fail_penalty].
    """
    parsed = parse_rec_output(output)
    if parsed is None:
        return [fail_penalty]
    ranked = dedupe_preserving_order([it.product_id for it in parsed.items])[:k]
    if not ranked:
        return [fail_penalty]
    candidates = [Candidate.model_validate(c) for c in example["candidates"]]
    bad = set(hallucinated_ids(ranked, candidates))
    relevance: dict[str, float] = example["relevance"]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal)) or 1.0
    gains = [
        fail_penalty if pid in bad else (2 ** relevance.get(pid, 0.0) - 1) / math.log2(i + 2) / idcg
        for i, pid in enumerate(ranked)
    ]
    out, acc = [0.0] * len(gains), 0.0
    for i in range(len(gains) - 1, -1, -1):
        acc += gains[i]
        out[i] = acc
    return out


def rank_group_advantages(per_rank: Sequence[Sequence[float]]) -> list[list[float]]:
    """Center each rank position across the group (positions absent in a sample count as 0)."""
    width = max((len(r) for r in per_rank), default=0)
    means = [
        statistics.fmean([r[i] if i < len(r) else 0.0 for r in per_rank]) for i in range(width)
    ]
    return [[r[i] - means[i] for i in range(len(r))] for r in per_rank]
