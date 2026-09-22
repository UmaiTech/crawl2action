"""Backend protocol + Tinker SDK adapter.

The SDK adapter follows the public tinker-cookbook recipes (sl_loop.py, rl_loop.py,
distillation/train_on_policy.py). All SDK calls are isolated here so API drift is a
one-file fix. It requires the `train` extra and TINKER_API_KEY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from c2a import NotYetImplemented
from c2a.train.data import Message


@dataclass
class Sample:
    text: str
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)


@dataclass
class RLDatum:
    """One sampled completion with its advantage: a scalar (sequence-level) or one value
    per sampled token (OPD / token-level credit)."""

    messages: list[Message]
    sample: Sample
    advantage: float | list[float]


class TrainingBackend(Protocol):
    def sft_step(self, batch: list[list[Message]], learning_rate: float) -> dict[str, float]: ...

    def sample(
        self, messages: list[Message], n: int, max_tokens: int, temperature: float
    ) -> list[Sample]: ...

    def rl_step(self, batch: list[RLDatum], learning_rate: float) -> dict[str, float]: ...

    def teacher_logprobs(self, messages: list[Message], sample: Sample) -> list[float]:
        """Teacher log-prob of each sampled token (same tokenizer as the student)."""
        ...

    def save(self, name: str) -> str: ...


def _msgs(messages: list[Message]) -> list[dict[str, str]]:
    return [m.model_dump() for m in messages]


class TinkerSDKBackend:
    """Adapter over `tinker` + `tinker_cookbook`. Not exercised in CI (needs credentials)."""

    def __init__(
        self,
        base_model: str,
        lora_rank: int = 32,
        teacher_model: str | None = None,
        max_length: int = 8192,
    ) -> None:
        try:
            import tinker
            from tinker_cookbook import model_info, renderers
            from tinker_cookbook.tokenizer_utils import get_tokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install the train extra: uv sync --extra train") from exc

        self._tinker = tinker
        self._renderers = renderers
        self.max_length = max_length
        tokenizer = get_tokenizer(base_model)
        self.renderer = renderers.get_renderer(
            model_info.get_recommended_renderer_name(base_model), tokenizer
        )
        self.service = tinker.ServiceClient()
        self.training = self.service.create_lora_training_client(
            base_model=base_model, rank=lora_rank
        )
        self._sampler = None
        self._teacher = (
            self.service.create_sampling_client(base_model=teacher_model) if teacher_model else None
        )

    def _adam(self, lr: float):
        return self._tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)

    def sft_step(self, batch: list[list[Message]], learning_rate: float) -> dict[str, float]:
        from tinker_cookbook.supervised.data import conversation_to_datum

        datums = [
            conversation_to_datum(
                _msgs(conv),
                self.renderer,
                self.max_length,
                self._renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            for conv in batch
        ]
        fb = self.training.forward_backward(datums, loss_fn="cross_entropy")
        opt = self.training.optim_step(self._adam(learning_rate))
        fb.result()
        res = opt.result()
        self._sampler = None
        return dict(res.metrics or {})

    def _sampling_client(self):
        if self._sampler is None:
            self._sampler = self.training.save_weights_and_get_sampling_client()
        return self._sampler

    def sample(
        self, messages: list[Message], n: int, max_tokens: int, temperature: float
    ) -> list[Sample]:
        prompt = self.renderer.build_generation_prompt(_msgs(messages))
        params = self._tinker.types.SamplingParams(
            max_tokens=max_tokens, temperature=temperature, stop=self.renderer.get_stop_sequences()
        )
        result = (
            self._sampling_client()
            .sample(prompt=prompt, num_samples=n, sampling_params=params)
            .result()
        )
        out = []
        for seq in result.sequences:
            parsed, _ = self.renderer.parse_response(seq.tokens)
            out.append(
                Sample(
                    text=self._renderers.get_text_content(parsed),
                    tokens=list(seq.tokens),
                    logprobs=list(seq.logprobs or []),
                )
            )
        return out

    def _datum(self, d: RLDatum) -> Any:
        import torch
        from tinker import types
        from tinker.types.tensor_data import TensorData

        prompt = self.renderer.build_generation_prompt(_msgs(d.messages))
        toks, lps = d.sample.tokens, d.sample.logprobs
        ob_len = prompt.length - 1
        model_input = prompt.append(types.EncodedTextChunk(tokens=toks[:-1]))
        adv = d.advantage if isinstance(d.advantage, list) else [d.advantage] * len(toks)
        if len(adv) != len(toks):
            raise ValueError("per-token advantages must match sampled token count")
        return types.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor([0] * ob_len + toks)),
                "logprobs": TensorData.from_torch(torch.tensor([0.0] * ob_len + lps)),
                "advantages": TensorData.from_torch(torch.tensor([0.0] * ob_len + adv)),
            },
        )

    def rl_step(self, batch: list[RLDatum], learning_rate: float) -> dict[str, float]:
        datums = [self._datum(d) for d in batch]
        fb = self.training.forward_backward(datums, loss_fn="importance_sampling")
        opt = self.training.optim_step(self._adam(learning_rate))
        fb.result()
        res = opt.result()
        self._sampler = None
        return dict(res.metrics or {})

    def teacher_logprobs(self, messages: list[Message], sample: Sample) -> list[float]:
        if self._teacher is None:
            raise RuntimeError("teacher_model not configured")
        from tinker import types

        prompt = self.renderer.build_generation_prompt(_msgs(messages))
        full = prompt.append(types.EncodedTextChunk(tokens=sample.tokens))
        lps = self._teacher.compute_logprobs(full).result()
        # logprobs are per position of `full`; keep those of the sampled tokens
        return [float(x) for x in lps[-len(sample.tokens) :]]

    def save(self, name: str) -> str:
        return self.training.save_weights_for_sampler(name).result().path

    def export(self, tinker_path: str, dest: str) -> str:  # pragma: no cover
        raise NotYetImplemented("checkpoint archive download for vLLM serving", "M3")
