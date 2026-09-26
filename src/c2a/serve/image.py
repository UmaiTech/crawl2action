"""Modal app: Qwen-Image-2.1 (diffusers) with optional post-trained product LoRA. M4."""

from __future__ import annotations

MODEL_ID = "Qwen/Qwen-Image-2.1"
GPU = "L40S"

try:  # pragma: no cover - optional dependency
    import modal

    app = modal.App("c2a-image")
except ImportError:  # pragma: no cover
    app = None
