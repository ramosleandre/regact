"""Grid games: the shared level/RESET layer, Sokoban, BabaIsAI and PuzzleScript."""

import json
import random
from pathlib import Path

import pytest

from regact.config.schema import InfoMode, ObsMode
from regact.envclient.errors import InvalidActionError
from regact.problems.base import build_problem
from regact.problems.grid_games.base import RESET

_DATA = Path(__file__).parents[1] / "src/regact/problems/grid_games/data"
_PACKS = json.loads((_DATA / "sokoban_packs.json").read_text())["packs"]


def _play(problem, task, actions):
    env = problem.make_env(task)
    first = env.reset()
    steps = [env.step(a) for a in actions]
    env.close()
    return first, steps


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
    assert scores["roae"] == scores["lroae"] == scores["mean_relative_actions"] == 1.0
    assert problem.progress_reference("sokoban_p01") == [
        (float(sum(optimal[:k])), float(k)) for k in range(1, 4)
    ]
    assert not problem.counts_as_action(RESET) and problem.counts_as_action(1)


def test_the_prompt_and_helper_never_name_the_game():
    problem = build_problem("sokoban", {})
    prompt = problem.build_prompt(
        "sokoban_p01", info_mode=InfoMode.INFORMATIVE, direct_interaction=False
    )
    helper = problem.helper_templates("sokoban_p01")[0].content
    for text in (prompt, helper):
        assert not any(
            word in text.lower() for word in ("sokoban", "box", "push", "crate", "goal tile")
        )
    assert "Levels to win: 8" in prompt and 'obs["frame"]' in prompt
    native, info = problem.make_env("sokoban_p01").reset()
    obs = problem.obs_renderer("sokoban_p01", mode=ObsMode.RAW).render(native, info)
    assert len(obs.frame) == 10 and problem.render_frame(obs).shape == (240, 240, 3)  # the grid itself


def _babaisai():
    pytest.importorskip("baba")
    return build_problem("babaisai", {})


def test_babaisai_plans_solve_their_levels_in_the_optimal_moves():
    problem = _babaisai()
    data = json.loads((_DATA / "babaisai_packs.json").read_text())
    for pack in data["packs"][:2]:
        plans = [level["plan"] for level in pack["levels"]]
        assert [len(p) for p in plans] == [level["optimal"] for level in pack["levels"]]
        _, steps = _play(problem, pack["key"], [a for plan in plans for a in plan])
        assert steps[-1][2] and steps[-1][4]["state"] == "WIN"
        assert all(99 not in row for frame, *_ in steps for row in frame)  # every kind is coded


def test_babaisai_hides_the_game_and_lifts_the_step_limit():
    problem = _babaisai()
    task = problem.get_task_names()[0]
    prompt = problem.build_prompt(task, info_mode=InfoMode.INFORMATIVE, direct_interaction=False)
    assert "baba" not in prompt.lower() and "one per kind of object" in prompt
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


def _puzzlescript():
    import shutil

    from regact.problems.grid_games import puzzlescript

    games = json.loads((_DATA / "puzzlescript_games.json").read_text())["games"]
    if shutil.which("node") is None or not all(
        puzzlescript.game_source(g["sha256"]).is_file() for g in games
    ):
        pytest.skip("needs Node and make install-puzzlescript")
    return build_problem("puzzlescript", {}), games


def test_puzzlescript_plans_solve_their_levels_in_the_optimal_moves():
    problem, games = _puzzlescript()
    for game in games[:3]:
        plans = [level["plan"] for level in game["levels"]]
        assert [len(p) for p in plans] == [level["optimal"] for level in game["levels"]]
        _, steps = _play(problem, game["key"], [a for plan in plans for a in plan])
        assert steps[-1][2] and steps[-1][4]["state"] == "WIN"
        assert sum(done for _, _, done, *_ in steps) == 1


def test_puzzlescript_sources_match_and_cells_list_object_ids():
    import hashlib

    from regact.problems.grid_games import puzzlescript

    problem, games = _puzzlescript()
    for game in games:
        source = puzzlescript.game_source(game["sha256"]).read_bytes()
        assert hashlib.sha256(source).hexdigest() == game["sha256"]
    env = problem.make_env(games[0]["key"])
    grid, info = env.reset()
    assert all(isinstance(cell, list) and cell == sorted(cell) for row in grid for cell in row)
    assert info["available_actions"] == [RESET, *games[0]["actions"]]
    env.step(games[0]["levels"][0]["plan"][0])
    assert env.step(RESET)[0] == grid  # RESET restores the level
    env.close()


def test_puzzlescript_play_is_deterministic_and_hides_the_game():
    problem, games = _puzzlescript()
    rng = random.Random(0)
    for game in games:
        actions = [rng.choice(game["actions"]) for _ in range(40)]
        runs = [[obs for obs, *_ in _play(problem, game["key"], actions)[1]] for _ in range(2)]
        assert runs[0] == runs[1], game["key"]
    prompt = problem.build_prompt(
        games[0]["key"], info_mode=InfoMode.INFORMATIVE, direct_interaction=False
    )
    assert games[0]["title"].lower() not in prompt.lower() and "puzzlescript" not in prompt.lower()


def test_the_viewer_draws_game_art_while_the_agent_keeps_its_neutral_picture():
    problem = _babaisai()
    task = problem.get_task_names()[0]
    native, info = problem.make_env(task).reset()
    obs = problem.obs_renderer(task, mode=ObsMode.RAW).render(native, info)
    agent, viewer = problem.render_frame(obs), problem.render_frame_for_viewer(obs, task)
    rows, cols = len(native), len(native[0])
    assert agent.shape[:2] != viewer.shape[:2] and viewer.shape[:2] == (rows * 32, cols * 32)
    sokoban = build_problem("sokoban", {})  # a game without its own art: the same picture
    native, info = sokoban.make_env("sokoban_p01").reset()
    obs = sokoban.obs_renderer("sokoban_p01", mode=ObsMode.RAW).render(native, info)
    assert (sokoban.render_frame_for_viewer(obs, "sokoban_p01") == sokoban.render_frame(obs)).all()


def test_puzzlescript_viewer_draws_each_game_with_its_sprites():
    problem, games = _puzzlescript()
    task = games[0]["key"]
    env = problem.make_env(task)
    native, info = env.reset()
    env.close()
    obs = problem.obs_renderer(task, mode=ObsMode.RAW).render(native, info)
    assert problem.render_frame_for_viewer(obs, task).shape[:2] == (
        len(native) * 25,
        len(native[0]) * 25,
    )
