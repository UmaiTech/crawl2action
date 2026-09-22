"""Modal app: vLLM OpenAI-compatible server for the post-trained student (+ task LoRAs). M4.

Deploy: `modal deploy src/c2a/serve/llm.py` (requires the modal extra and exported weights
on the `c2a-models` Volume)."""

from __future__ import annotations

MODEL_VOLUME = "c2a-models"
GPU = "H100"  # Qwen3.8-27B in FP8 fits on 1x H100; use "H100:2" for BF16
VLLM_ARGS = ["--enable-lora", "--max-lora-rank", "64", "--guided-decoding-backend", "auto"]

try:  # pragma: no cover - optional dependency
    import modal

    app = modal.App("c2a-llm")
except ImportError:  # pragma: no cover
    app = None
