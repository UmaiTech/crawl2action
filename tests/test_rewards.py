import json

import pytest

from c2a.train.rewards import (
    copy_reward,
    decide_reward,
    group_advantages,
    has_signal,
    rank_group_advantages,
    rank_rewards_to_go,
    rec_reward,
)


def out(*ids):
    return json.dumps({"items": [{"product_id": i} for i in ids]})


def test_rec_reward_penalizes_hard_failures(rec_example):
    assert rec_reward(out("a", "b"), rec_example, -0.5) == 1.0
    assert rec_reward(out("zzz"), rec_example, -0.5) == -0.5
    assert rec_reward("oops", rec_example, -0.5) == -0.5


def test_decide_reward_is_calibration_aware():
    ex = {"labels": ["a", "b"], "label": "a"}
    confident_right = decide_reward(
        json.dumps({"label": "a", "probs": {"a": 0.9, "b": 0.1}}), ex, -1
    )
    hedged = decide_reward(json.dumps({"label": "a", "probs": {"a": 0.6, "b": 0.4}}), ex, -1)
    confident_wrong = decide_reward(
        json.dumps({"label": "b", "probs": {"a": 0.1, "b": 0.9}}), ex, -1
    )
    assert confident_right > hedged > confident_wrong >= 0
    assert decide_reward(json.dumps({"label": "a", "probs": {"a": 1.0}}), ex, -1) == -1


def test_copy_reward_checks_numbers_against_attributes():
    ex = {"attributes": {"weight": "250 g", "material": "merino"}}
    good = json.dumps({"text": "t", "claims": ["Weighs 250 g"]})
    bad = json.dumps({"text": "t", "claims": ["Weighs 300 g", "Weighs 250 g"]})
    assert copy_reward(good, ex, -1) == 1.0 and copy_reward(bad, ex, -1) == 0.5


def test_group_advantages_modes():
    assert group_advantages([1, 0, 0, 1], "dr_grpo") == [0.5, -0.5, -0.5, 0.5]
    assert group_advantages([1, 0], "grpo") == pytest.approx([1.0, -1.0], abs=1e-4)
    assert not has_signal([0.3, 0.3, 0.3]) and has_signal([0, 1])


def test_rank_rewards_to_go_are_causal(rec_example):
    rtg = rank_rewards_to_go(out("x_bad", "a"), rec_example, k=3, fail_penalty=-0.5)
    assert len(rtg) == 2 and rtg[0] < rtg[1]  # hallucinated top item lowers rank-0 return only
    perfect = rank_rewards_to_go(out("a", "b"), rec_example, k=3)
    assert perfect[0] == pytest.approx(1.0) and perfect[0] > perfect[1] > 0
    assert rank_rewards_to_go("nope", rec_example) == [-0.5]


def test_rank_group_advantages_center_per_position():
    advs = rank_group_advantages([[1.0, 0.5], [0.0]])
    assert advs == [[0.5, 0.25], [-0.5]]
