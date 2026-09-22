"""On-policy distillation: the student samples, the teacher scores every sampled token;
per-token advantage = -kl_coef * (log p_student - log p_teacher) (negative reverse KL).
Teacher and student must share a tokenizer (e.g. RL'd Qwen3.8-27B -> small Qwen, or
Inkling -> Inkling-Small)."""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from c2a.train.base import RunConfig
from c2a.train.data import DistillPrompt
from c2a.train.tinker.client import RLDatum, TrainingBackend
from c2a.train.tinker.sft import _write_summary


def reverse_kl_advantages(
    student_lps: Sequence[float], teacher_lps: Sequence[float], kl_coef: float
) -> list[float]:
    if len(student_lps) != len(teacher_lps):
        raise ValueError("student and teacher logprobs must align token-by-token")
    return [-kl_coef * (s - t) for s, t in zip(student_lps, teacher_lps, strict=True)]


def run_opd(backend: TrainingBackend, prompts: Sequence[DistillPrompt], cfg: RunConfig) -> dict:
    if not cfg.teacher_model:
        raise ValueError("OPD requires teacher_model")
    n_batches = len(prompts) // cfg.batch_size
    if n_batches == 0:
        raise ValueError(f"need at least batch_size={cfg.batch_size} prompts, got {len(prompts)}")
    steps = min(n_batches, cfg.steps or n_batches)
    history, checkpoints = [], []
    for step in range(steps):
        batch = prompts[step * cfg.batch_size : (step + 1) * cfg.batch_size]
        datums, kls = [], []
        for p in batch:
            for s in backend.sample(p.messages, cfg.group_size, cfg.max_tokens, cfg.temperature):
                t_lps = backend.teacher_logprobs(p.messages, s)
                adv = reverse_kl_advantages(s.logprobs, t_lps, cfg.kl_coef)
                kls.append(
                    statistics.fmean(a - b for a, b in zip(s.logprobs, t_lps, strict=True))
                    if t_lps
                    else 0.0
                )
                datums.append(RLDatum(p.messages, s, adv))
        metrics = backend.rl_step(datums, cfg.learning_rate)
        history.append({"step": step, "reverse_kl": statistics.fmean(kls), **metrics})
        if cfg.save_every and step > 0 and step % cfg.save_every == 0:
            checkpoints.append(backend.save(f"{cfg.name}-{step:06d}"))
    checkpoints.append(backend.save(f"{cfg.name}-final"))
    summary = {"stage": "opd", "steps": steps, "checkpoints": checkpoints, "history": history}
    _write_summary(cfg, summary)
    return summary
