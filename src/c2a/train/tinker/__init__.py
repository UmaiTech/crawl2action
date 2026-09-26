"""Tinker (Thinking Machines) training loops: SFT, RL (GRPO family) and on-policy distillation.

Loops are platform-agnostic orchestration over `TrainingBackend`; `TinkerSDKBackend`
adapts the real tinker SDK + tinker-cookbook renderers. Tests use a fake backend.
"""
