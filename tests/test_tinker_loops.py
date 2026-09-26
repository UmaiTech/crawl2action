import json

import pytest

from c2a.train.base import RunConfig
from c2a.train.data import DistillPrompt, Message, RLPrompt, SFTExample
from c2a.train.tinker.client import Sample
from c2a.train.tinker.opd import reverse_kl_advantages, run_opd
from c2a.train.tinker.rl import run_rl
from c2a.train.tinker.sft import run_sft


class FakeBackend:
    """Records calls; samples alternate between a perfect and a hallucinated ranking."""

    def __init__(self):
        self.sft_batches, self.rl_batches, self.saved = [], [], []
        self._i = 0

    def sft_step(self, batch, learning_rate):
        self.sft_batches.append((len(batch), learning_rate))
        return {"loss": 1.0 / (len(self.sft_batches))}

    def sample(self, messages, n, max_tokens, temperature):
        outs = []
        for _ in range(n):
            ids = ["a", "b"] if self._i % 2 == 0 else ["zzz"]
            self._i += 1
            outs.append(
                Sample(
                    text=json.dumps({"items": [{"product_id": i} for i in ids]}),
                    tokens=[1, 2, 3],
                    logprobs=[-0.1, -0.2, -0.3],
                )
            )
        return outs

    def rl_step(self, batch, learning_rate):
        self.rl_batches.append(batch)
        return {"loss": 0.0}

    def teacher_logprobs(self, messages, sample):
        return [-0.2, -0.2, -0.2]

    def save(self, name):
        self.saved.append(name)
        return f"tinker://fake/{name}"


def cfg(tmp_path, **kw):
    base = dict(
        name="t",
        stage="sft",
        base_model="m",
        dataset=tmp_path / "d.jsonl",
        log_dir=tmp_path,
        batch_size=2,
        save_every=1,
    )
    return RunConfig(**{**base, **kw})


def msgs():
    return [Message(role="user", content="q"), Message(role="assistant", content="a")]


def test_sft_loop(tmp_path):
    be = FakeBackend()
    exs = [SFTExample(id=str(i), messages=msgs()) for i in range(5)]
    summary = run_sft(be, exs, cfg(tmp_path))
    assert summary["steps"] == 2 and [n for n, _ in be.sft_batches] == [2, 2]
    assert be.sft_batches[0][1] > be.sft_batches[1][1]  # lr decays
    assert be.saved[-1] == "t-final" and (tmp_path / "t" / "summary.json").exists()


@pytest.mark.parametrize("mode", ["dr_grpo", "grpo", "rank_grpo"])
def test_rl_loop_uses_grader_rewards(tmp_path, rec_example, mode):
    be = FakeBackend()
    prompts = [
        RLPrompt(id=str(i), messages=msgs()[:1], reward="rec", example=rec_example)
        for i in range(2)
    ]
    summary = run_rl(be, prompts, cfg(tmp_path, stage="rl", group_size=4, advantage=mode, steps=1))
    batch = be.rl_batches[0]
    assert len(batch) == 8
    good = [d.advantage for d in batch if "zzz" not in d.sample.text]
    bad = [d.advantage for d in batch if "zzz" in d.sample.text]
    assert min(good) > 0 > max(bad)
    assert summary["history"][0]["groups_dropped"] == 0


def test_rl_loop_drops_zero_variance_groups(tmp_path, rec_example):
    be = FakeBackend()
    be.sample = lambda m, n, t, temp: [Sample(text="bad")] * n
    prompts = [RLPrompt(id="1", messages=msgs()[:1], reward="rec", example=rec_example)]
    summary = run_rl(be, prompts, cfg(tmp_path, stage="rl", batch_size=1, steps=1))
    assert be.rl_batches == [] and summary["history"][0]["groups_dropped"] == 1


def test_opd_loop(tmp_path):
    be = FakeBackend()
    prompts = [DistillPrompt(id=str(i), messages=msgs()[:1]) for i in range(2)]
    run_opd(be, prompts, cfg(tmp_path, stage="opd", teacher_model="t", group_size=2, steps=1))
    adv = be.rl_batches[0][0].advantage
    assert adv == pytest.approx([-0.1, 0.0, 0.1])  # -(student - teacher)
    with pytest.raises(ValueError):
        run_opd(be, prompts, cfg(tmp_path, stage="opd"))


def test_reverse_kl_advantages_alignment():
    with pytest.raises(ValueError):
        reverse_kl_advantages([0.0], [0.0, 1.0], 1.0)
