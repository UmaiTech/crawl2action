"""Modal-hosted rollout environments / reward endpoints called from Tinker loops (M3):
candidate retrieval, graders, judge/decider scoring and image rendering for rewards."""

from c2a import NotYetImplemented


def remote_reward(name: str, output: str, example: dict) -> float:
    raise NotYetImplemented(f"remote reward endpoint '{name}'", "M3")
