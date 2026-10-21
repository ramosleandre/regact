"""The MiniGrid CWM task set: deterministic within an episode, with the carried object shown."""

import random

import pytest

from regact.env.renderer import jsonify
from regact.problems.base import build_problem
from regact.problems.minigrid.tasks import BABYAI_TASKS, CWM_MINIGRID_TASKS

pytest.importorskip("minigrid")


def _rollout(env, actions, *, swap_rng: bool):
    import numpy as np

    obs, _ = env.reset(seed=0)
    if swap_rng:  # a step that draws randomness would now see different numbers
        env._env.unwrapped.np_random = np.random.default_rng(12345)
    out = [obs]
    for action in actions:
        obs, reward, terminated, truncated, _ = env.step(action)
        out.append((obs, reward, terminated, truncated))
        if terminated or truncated:
            break
    return jsonify(out)


@pytest.mark.parametrize("task", CWM_MINIGRID_TASKS)
def test_cwm_tasks_draw_no_randomness_within_an_episode(task):
    problem = build_problem("minigrid", {"fully_obs": True})
    actions = [random.Random(task).randrange(7) for _ in range(120)]
    plain = _rollout(problem.make_env(task), actions, swap_rng=False)
    swapped = _rollout(problem.make_env(task), actions, swap_rng=True)
    assert plain == swapped


def test_the_full_frame_shows_the_carried_object_and_rewards_are_rounded():
    problem = build_problem("minigrid", {"fully_obs": True})
    assert set(CWM_MINIGRID_TASKS) <= set(problem.get_task_names())
    assert "BabyAI-PutNextLocal-v0" in BABYAI_TASKS
    env = problem.make_env("MiniGrid-DoorKey-5x5-v0")
    obs, _ = env.reset(seed=0)
    assert obs["carrying"] is None
    from minigrid.core.world_object import Key

    env._env.unwrapped.carrying = Key("yellow")
    obs, reward, *_ = env.step(6)  # "done" changes nothing; the frame reports what is carried
    assert list(obs["carrying"][:2]) == [5, 4] and reward == round(reward, 6)


def test_the_check_catches_randomness_within_an_episode():
    """Control: Dynamic-Obstacles moves its obstacles at random on every step."""
    problem = build_problem("minigrid", {"fully_obs": True})
    task = "MiniGrid-Dynamic-Obstacles-8x8-v0"
    actions = [0] * 30  # turn in place: only the obstacles move
    plain = _rollout(problem.make_env(task), actions, swap_rng=False)
    assert plain != _rollout(problem.make_env(task), actions, swap_rng=True)
