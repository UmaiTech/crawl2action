"""RL loop (GRPO family) with grader-backed rewards.

Per step: sample `group_size` completions per prompt -> reward -> group advantages
(dr_grpo / grpo) -> drop zero-variance groups (DAPO dynamic sampling) -> importance-
sampling policy-gradient step. `rank_grpo` computes per-rank advantages; mapping ranks to
token spans in the SDK backend is scheduled for M3, so here it falls back to the rank-0
advantage (the full-list reward-to-go) as a sequence-level signal.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from c2a.train.base import RunConfig
from c2a.train.data import RLPrompt
from c2a.train.rewards import (
    REWARD_FNS,
    group_advantages,
    has_signal,
    rank_group_advantages,
    rank_rewards_to_go,
)
from c2a.train.tinker.client import RLDatum, TrainingBackend
from c2a.train.tinker.sft import _write_summary


def run_rl(backend: TrainingBackend, prompts: Sequence[RLPrompt], cfg: RunConfig) -> dict:
    n_batches = len(prompts) // cfg.batch_size
    if n_batches == 0:
        raise ValueError(f"need at least batch_size={cfg.batch_size} prompts, got {len(prompts)}")
    steps = min(n_batches, cfg.steps or n_batches)
    history, checkpoints = [], []
    for step in range(steps):
        batch = prompts[step * cfg.batch_size : (step + 1) * cfg.batch_size]
        datums: list[RLDatum] = []
        mean_rewards, dropped = [], 0
        for p in batch:
            reward_fn = REWARD_FNS[cfg.reward or p.reward]
            samples = backend.sample(p.messages, cfg.group_size, cfg.max_tokens, cfg.temperature)
            rewards = [reward_fn(s.text, p.example, cfg.fail_penalty) for s in samples]
            mean_rewards.append(statistics.fmean(rewards))
            if cfg.advantage == "rank_grpo":
                per_rank = [
                    rank_rewards_to_go(s.text, p.example, p.example.get("k", 10), cfg.fail_penalty)
                    for s in samples
                ]
                advs = [a[0] if a else 0.0 for a in rank_group_advantages(per_rank)]
                signal = has_signal([r[0] for r in per_rank])
            else:
                advs = group_advantages(rewards, cfg.advantage)
                signal = has_signal(rewards)
            if cfg.drop_zero_variance_groups and not signal:
                dropped += 1
                continue
            datums += [RLDatum(p.messages, s, a) for s, a in zip(samples, advs, strict=True)]
        metrics = backend.rl_step(datums, cfg.learning_rate) if datums else {"skipped": 1.0}
        history.append(
            {
                "step": step,
                "reward_mean": statistics.fmean(mean_rewards),
                "groups_dropped": dropped,
                "datums": len(datums),
                **metrics,
            }
        )
        if cfg.save_every and step > 0 and step % cfg.save_every == 0:
            checkpoints.append(backend.save(f"{cfg.name}-{step:06d}"))
    checkpoints.append(backend.save(f"{cfg.name}-final"))
    summary = {"stage": "rl", "steps": steps, "checkpoints": checkpoints, "history": history}
    _write_summary(cfg, summary)
    return summary
