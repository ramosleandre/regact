"""Crafter as a regact problem: the observation is the picture's content, written as names."""

import json

import pytest

from regact.config.schema import InfoMode, ObsMode
from regact.envclient.errors import InvalidActionError
from regact.problems.base import build_problem

pytest.importorskip("crafter")
from regact.problems import crafter as cr  # noqa: E402


def _play(problem, actions, task=cr.TASK):
    env = problem.make_env(task)
    frames = [env.reset()[0]]
    for action in actions:
        frame, _, dead, _, info = env.step(action)
        frames.append(frame)
        if dead:
            break
    return env, frames, info


def test_crafter_observation_and_actions():
    problem = build_problem("crafter", {})
    env = problem.make_env(cr.TASK)
    obs = problem.obs_renderer(cr.TASK, mode=ObsMode.RAW).render(*env.reset())
    view = obs.frame["view"]
    assert (len(view[0]), len(view)) == cr.VIEW and view[3][4] == "player"
    assert obs.available_actions == list(range(len(cr.ACTIONS))) and obs.info["achievements"] == []
    json.dumps(obs.frame)
    assert problem.render_frame(obs).shape == (9 * 48, 9 * 48, 3)  # 7 map rows + 2 of inventory
    frame, _, dead, over, _ = env.step(cr.ACTIONS.index("move_left"))
    assert frame["facing"] == "left" and not dead and not over
    for bad in (17, -1, True, "do"):
        with pytest.raises(InvalidActionError):
            env.step(bad)


def test_crafter_metrics_per_life_and_on_held_out_worlds():
    problem = build_problem("crafter", {"seeds": [0, 7], "eval_seeds": [100, 101]})
    assert problem.get_task_names() == ["crafter_s0", "crafter_s7"]
    assert problem.evaluation_tasks("crafter_s0") == ["crafter_s100", "crafter_s101"]
    wood, both = ["collect_wood"], ["collect_wood", "wake_up"]
    trace = [  # the second life starts after a reset, which counts as one action
        (1, {"achievements": [], "life_achievements": []}),
        (2, {"achievements": wood, "life_achievements": wood}),
        (4, {"achievements": wood, "life_achievements": []}),
        (5, {"achievements": both, "life_achievements": ["wake_up"]}),
    ]
    assert problem.derived_trace_metrics(cr.TASK, trace) == {
        "achievements": 2,
        "actions_to_last": 5,
        "lives": [[2, 1], [5, 1]],
        "first_life_achievements": 1,
        "best_life_achievements": 1,
    }
    scores = problem.evaluation_metrics([{"info": {"achievements": both}}, {"info": {}}])
    assert scores["eval_progress"] == pytest.approx(100 / 22)
    assert scores["eval_success_rates"]["wake_up"] == 50
    # two achievements at 50 %, twenty at 0: exp(2 ln(51) / 22) - 1
    assert scores["eval_crafter_score"] == pytest.approx(51 ** (2 / 22) - 1)
    assert cr.crafter_score([100] * 22) == pytest.approx(100)


def test_crafter_lives_share_the_map_and_keep_achievements():
    """A reset starts a new life on the same map with an empty inventory; what was unlocked
    stays unlocked, and unlocking everything ends the task."""
    problem = build_problem("crafter", {})
    env, frames, _ = _play(problem, [cr.ACTIONS.index("noop")], task="crafter_s3")
    first = frames[0]["view"]
    env._env._player.achievements["collect_wood"] += 1
    env._env._player.inventory["wood"] = 5
    _, _, _, _, info = env.step(0)
    assert info["achievements"] == ["collect_wood"]
    frame, info = env.reset()
    terrain = lambda view: [[c for c in row if c not in ("cow", "zombie")] for row in view]  # noqa: E731
    assert terrain(frame["view"]) == terrain(first) and frame["inventory"]["wood"] == 0
    assert info["achievements"] == ["collect_wood"] and info["life_achievements"] == []
    for name in cr.ACHIEVEMENTS:
        env._env._player.achievements[name] += 1
    _, _, done, _, info = env.step(0)
    assert done and len(info["achievements"]) == len(cr.ACHIEVEMENTS)


def test_crafter_prompts_state_its_own_rules():
    problem = build_problem("crafter", {})
    section = problem.environment_prompt()
    assert "stochastic" in section and "levels: completing" not in section
    for name in (*cr.ACTIONS, *cr.ACHIEVEMENTS):  # information-minimal: none is named
        assert name == "do" or name not in section
    assert "final_controller.py" not in section  # no held-out worlds configured
    held_out = build_problem("crafter", {"eval_seeds": [100, 101]}).environment_prompt()
    assert "in 2 other worlds" in held_out and "`final_controller.py`" in held_out
    assert "fully solved" not in problem.continue_reminder()
    prompt = problem.build_prompt(cr.TASK, info_mode=InfoMode.INFORMATIVE)
    assert "`5` do" in prompt and "{" not in prompt
