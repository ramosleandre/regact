"""Grid games (v8-envs): the shared level/RESET layer and Sokoban's packs."""

import json
import random
from pathlib import Path

import pytest

from regact.config.schema import InfoMode, ObsMode
from regact.envclient.errors import InvalidActionError
from regact.problems.base import build_problem
from regact.problems.grid_games.base import RESET

_PACKS = json.loads(
    (
        Path(__file__).parents[1] / "src/regact/problems/grid_games/data/sokoban_packs.json"
    ).read_text()
)["packs"]


def _play(problem, task, actions):
    env = problem.make_env(task)
    first = env.reset()
    return first, [env.step(a) for a in actions]


def test_every_stored_plan_solves_its_level_in_exactly_the_optimal_moves():
    problem = build_problem("sokoban", {})
    for pack in _PACKS:
        plans = [level["plan"] for level in pack["levels"]]
        assert [len(p) for p in plans] == [level["optimal"] for level in pack["levels"]]
        _, steps = _play(problem, pack["key"], [a for plan in plans for a in plan])
        completed = [info["levels_completed"] for *_, info in steps]
        boundaries = [sum(map(len, plans[: i + 1])) for i in range(len(plans))]
        for level, end in enumerate(boundaries):  # each level completes on its plan's last move
            assert completed[end - 1] == level + 1 and completed[end - 2] == level
        _, reward, done, _, info = steps[-1]
        assert done and reward == 1.0 and info["state"] == "WIN" and info["available_actions"] == []


def test_reset_restarts_the_current_level_and_play_is_deterministic():
    problem = build_problem("sokoban", {})
    plan = _PACKS[0]["levels"][0]["plan"]
    first, steps = _play(problem, "sokoban_p01", plan + plan[:5] + [RESET])
    level_two_start = steps[len(plan) - 1][0]
    assert steps[-1][0] == level_two_start and steps[-1][4]["levels_completed"] == 1
    again = _play(problem, "sokoban_p01", plan + plan[:5] + [RESET])
    assert again[1] == steps and again[0] == first


def test_actions_outside_the_game_and_after_the_win_are_refused():
    problem = build_problem("sokoban", {"max_levels": 1})
    env = problem.make_env("sokoban_p01")
    env.reset()
    for bad in (5, -1, True, "1", None):
        with pytest.raises(InvalidActionError):
            env.step(bad)
    for action in _PACKS[0]["levels"][0]["plan"]:
        *_, done, _, info = env.step(action)
    assert done and info["win_levels"] == 1
    with pytest.raises(InvalidActionError):
        env.step(1)


def test_optimal_play_scores_rhae_one_and_the_reference_is_cumulative():
    problem = build_problem("sokoban", {"max_levels": 3})
    optimal = [level["optimal"] for level in _PACKS[0]["levels"][:3]]
    trace, actions = [], 0
    for level, n in enumerate(optimal, start=1):
        for i in range(n):
            actions += 1
            trace.append((actions, {"levels_completed": level if i == n - 1 else level - 1}))
    scores = problem.derived_trace_metrics("sokoban_p01", trace)
    assert scores["roae"] == 1.0 and scores["lroae"] == 1.0 and scores["relative_env_actions"] == 1.0
    assert problem.progress_reference("sokoban_p01") == [
        (float(sum(optimal[:k])), float(k)) for k in range(1, 4)
    ]
    assert not problem.counts_as_action(RESET) and problem.counts_as_action(1)


def test_the_prompt_and_helper_never_name_the_game():
    problem = build_problem("sokoban", {})
    prompt = problem.build_prompt("sokoban_p01", info_mode=InfoMode.INFORMATIVE, direct_interaction=False)
    helper = problem.helper_templates("sokoban_p01")[0].content
    for text in (prompt, helper):
        assert not any(
            word in text.lower() for word in ("sokoban", "box", "push", "crate", "goal tile")
        )
    assert "Levels to win: 7" in prompt and 'obs["frame"]' in prompt
    obs = problem.obs_renderer("sokoban_p01", mode=ObsMode.RAW).render(*problem.make_env("sokoban_p01").reset())
    assert len(obs.frame) == 1 and problem.render_frame(obs).shape == (240, 240, 3)


def _babaisai():
    pytest.importorskip("baba")
    return build_problem("babaisai", {})


def test_babaisai_plans_solve_their_levels_in_the_optimal_moves():
    problem = _babaisai()
    data = json.loads(
        (Path(__file__).parents[1] / "src/regact/problems/grid_games/data/babaisai_packs.json").read_text()
    )
    for pack in data["packs"][:2]:
        plans = [level["plan"] for level in pack["levels"]]
        assert [len(p) for p in plans] == [level["optimal"] for level in pack["levels"]]
        _, steps = _play(problem, pack["key"], [a for plan in plans for a in plan])
        assert steps[-1][2] and steps[-1][4]["state"] == "WIN"
        assert all(99 not in row for frame, *_ in steps for row in frame)  # every cell kind is coded


def test_babaisai_hides_the_game_and_lifts_the_step_limit():
    problem = _babaisai()
    task = problem.get_task_names()[0]
    prompt = problem.build_prompt(task, info_mode=InfoMode.INFORMATIVE, direct_interaction=False)
    assert "baba" not in prompt.lower() and "one integer per kind of object" in prompt
    env = problem.make_env(task)
    env.reset()
    for i in range(150):  # past the engine's own 100-step limit: the level is still running
        assert not env.step(RESET if i % 50 == 49 else 1)[2]
    obs = problem.obs_renderer(task, mode=ObsMode.RAW).render(*env.reset())
    frame = problem.render_frame(obs)
    assert frame.shape[2] == 3 and problem.secret_modules() == ("baba",)



class _NoDraws:
    def __getattr__(self, name):
        raise AssertionError(f"a step drew randomness ({name})")


def test_babaisai_levels_are_fixed_by_their_seed_and_steps_draw_no_randomness():
    """A level is (env id, seed) whatever the global generators hold, and play draws nothing."""
    import numpy as np

    from regact.problems.grid_games import babaisai

    _babaisai()
    rng = random.Random(0)
    for env_id, seed in [lvl for task in babaisai._load()[0].values() for lvl in task.levels][::5]:
        layouts = []
        for global_seed in (1, 2):
            np.random.seed(global_seed)
            random.seed(global_seed)
            layouts.append(babaisai.encoded_rows(babaisai.make_engine(env_id, seed)))
        assert layouts[0] == layouts[1], env_id
        engine = babaisai.make_engine(env_id, seed)
        engine._np_random = _NoDraws()
        for _ in range(60):
            engine.step(babaisai.ENGINE_ACTION[rng.randrange(1, 5)])
