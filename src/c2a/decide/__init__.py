"""Decision layer: decide(evidence, question, labels) -> calibrated label probabilities.

Inspired by vLLM Semantic Router's Decision-1.0 models. The production decider is
post-trained by us (SFT on ESCI + C2A gold, then RL with a calibration reward).
"""

from c2a.decide.base import Decider, normalize_probs

__all__ = ["Decider", "normalize_probs"]
