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
    assert obs.available_actions == list(cr.ACTIONS) and obs.info["achievements"] == []
    json.dumps(obs.frame)
    assert problem.render_frame(obs).shape == (9 * 48, 9 * 48, 3)  # 7 map rows + 2 of inventory
    frame, _, dead, over, _ = env.step("move_left")
    assert frame["facing"] == "left" and not dead and not over
    for bad in (5, True, "jump", None):
        with pytest.raises(InvalidActionError):
            env.step(bad)


def test_crafter_metrics_per_life_and_on_held_out_worlds():
    problem = build_problem(
        "crafter", {"seeds": [0, 7], "unseeded": [2], "synthesis": [2], "eval_seeds": [100, 101]}
    )
    assert problem.get_task_names() == ["crafter_s0", "crafter_s7", "crafter_u2", "crafter_c2"]
    for played in ("crafter_s0", "crafter_u2"):  # only a synthesis task is scored by a program
        assert problem.evaluation_tasks(played) == []
        assert problem.evaluation_trajectories(played) == []
    assert problem.evaluation_tasks("crafter_c2") == ["crafter_s100", "crafter_s101"]
    replays = problem.evaluation_trajectories("crafter_c2")
    assert [name for name, _ in replays] == ["crafter_s100"] * 2 + ["crafter_s101"] * 2
    assert set(replays[0][1]) <= set(cr.ACTIONS)
    wood, both = ["collect_wood"], ["collect_wood", "wake_up"]
    trace = [  # a reset counts as one action: lives end at 2 (death), 5 (abandoned) and 8
        (1, {"achievements": []}),
        (2, {"achievements": wood, "milestones": ["the player died"]}),
        (4, {"achievements": ["wake_up"]}),
        (5, {"achievements": ["wake_up"]}),
        (7, {"achievements": wood}),
        (8, {"achievements": both, "milestones": ["the player died"]}),
    ]
    assert problem.derived_trace_metrics(cr.TASK, trace) == {
        "achievements": 2,
        "actions_to_last": 4,
        "lives": [[2, 1, 1], [5, 1, 0], [8, 2, 1]],
        "first_life_achievements": 1,
        "best_life_achievements": 2,
        "mean_life_achievements": 1.5,
        "worst_life_achievements": 1,
    }
    assert not problem.is_perfect({"success_rate": 1.0})  # one good life is not a solved game
    scores = problem.evaluation_metrics([{"info": {"achievements": both}}, {"info": {}}])
    assert scores["eval_progress"] == pytest.approx(100 / 22)
    assert scores["eval_success_rates"]["wake_up"] == 50
    # two achievements at 50 %, twenty at 0: exp(2 ln(51) / 22) - 1
    assert scores["eval_crafter_score"] == pytest.approx(51 ** (2 / 22) - 1)
    assert cr.crafter_score([100] * 22) == pytest.approx(100)


def test_crafter_lives_share_the_map_and_start_with_nothing():
    problem = build_problem("crafter", {})
    env, frames, _ = _play(problem, ["noop"], task="crafter_s3")
    first = frames[0]["view"]
    env._env._player.achievements["collect_wood"] += 1
    env._env._player.inventory["wood"] = 5
    _, _, _, _, info = env.step("noop")
    assert info["achievements"] == ["collect_wood"]
    frame, info = env.reset()
    terrain = lambda view: [[c for c in row if c not in ("cow", "zombie")] for row in view]  # noqa: E731
    assert terrain(frame["view"]) == terrain(first) and frame["inventory"]["wood"] == 0
    assert info["achievements"] == []
    for name in cr.ACHIEVEMENTS:
        env._env._player.achievements[name] += 1
    _, _, done, _, info = env.step("noop")
    assert not done and len(info["achievements"]) == len(cr.ACHIEVEMENTS)


def test_an_unseeded_task_gives_each_life_a_new_map():
    problem = build_problem("crafter", {"unseeded": [0, 1], "synthesis": [0]})
    env = problem.make_env("crafter_u0")
    first, _ = env.reset()
    second, _ = env.reset()
    assert second["view"] != first["view"]
    for same_maps in ("crafter_u0", "crafter_c0"):  # the stream of maps is the index's own
        again = problem.make_env(same_maps)
        assert again.reset()[0] == first and again.reset()[0] == second
    assert problem.make_env("crafter_u1").reset()[0]["view"] != first["view"]


def test_crafter_prompts_state_its_own_rules():
    problem = build_problem("crafter", {"unseeded": [0], "synthesis": [0], "eval_seeds": [100, 101]})
    section = problem.environment_prompt(cr.TASK)
    assert "stochastic" in section and "{" not in section
    assert "- `make_iron_sword`: craft an iron sword" in section
    assert all(f"`{name}`" in section for name in cr.ACHIEVEMENTS)
    assert "same world" in section and "final_controller.py" not in section
    assert "final_controller.py" not in problem.objective_prompt(cr.TASK)
    unseeded = problem.environment_prompt("crafter_u0")
    assert "in a new world" in unseeded and "final_controller.py" not in unseeded
    assert "final_controller.py" not in problem.continue_reminder("crafter_u0")
    synthesis = problem.environment_prompt("crafter_c0")
    assert "in a new world" in synthesis and "{" not in synthesis
    assert "in 2 more new worlds" in synthesis and "`final_controller.py`" in synthesis
    assert problem.objective_prompt("crafter_c0").startswith("Produce a program")
    assert "`final_controller.py`" in problem.continue_reminder("crafter_c0")
    prompt = problem.build_prompt(cr.TASK, info_mode=InfoMode.INFORMATIVE)
    assert "`do`" in prompt and "{" not in prompt
