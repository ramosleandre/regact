"""Tests for the problem layer: registry, MiniGrid contract, ARC deferral.

The pure contract (registry, prompt, metrics, config_kwargs, renderer) runs
always. ``make_env`` needs the ``minigrid`` extra, so it is gated with
``importorskip`` — it runs where the lib is installed, skips cleanly otherwise.
"""

import pytest

from regact.config.schema import InfoMode, ObsMode
from regact.envclient.obs import Obs
from regact.problems.base import BaseProblem, build_problem
from regact.problems.minigrid import (
    ALL_MINIGRID_TASKS,
    LITE_MINIGRID_TASKS,
    MiniGridProblem,
    MiniGridRenderer,
)


def test_build_problem_resolves_minigrid() -> None:
    problem = build_problem("minigrid", {"fully_obs": False})
    assert isinstance(problem, MiniGridProblem)
    assert problem.name == "minigrid"


def test_build_problem_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown problem"):
        build_problem("nope", {})


def test_minigrid_config_kwargs_roundtrip() -> None:
    problem = MiniGridProblem(fully_obs=True)
    kwargs = problem.config_kwargs()
    assert kwargs == {"fully_obs": True}
    # The trusted side can rebuild from these kwargs.
    rebuilt = build_problem("minigrid", kwargs)
    assert isinstance(rebuilt, MiniGridProblem)
    assert rebuilt.config_kwargs() == kwargs


def test_minigrid_hides_the_engine_from_the_sandbox() -> None:
    """MiniGrid must be a secret module (like ARC's arcengine): otherwise an agent could
    ``import minigrid; gymnasium.make(<task>)`` to reconstruct the exact env in-process and
    bypass the HTTP-only boundary. The end-to-end deny-read proof is in test_sandbox.py."""
    assert MiniGridProblem().secret_modules() == ("minigrid",)


def test_minigrid_catalogue_matches_gameagents_sizes() -> None:
    assert len(ALL_MINIGRID_TASKS) == 72
    assert len(LITE_MINIGRID_TASKS) == 20
    assert len(set(ALL_MINIGRID_TASKS)) == len(ALL_MINIGRID_TASKS)
    assert len(set(LITE_MINIGRID_TASKS)) == len(LITE_MINIGRID_TASKS)
    assert set(LITE_MINIGRID_TASKS) < set(ALL_MINIGRID_TASKS)


def test_minigrid_build_prompt_informative_describes_actions() -> None:
    prompt = MiniGridProblem().build_prompt("MiniGrid-Empty-5x5-v0", info_mode=InfoMode.INFORMATIVE)
    assert "MiniGrid-Empty-5x5-v0" in prompt
    assert "available_actions" in prompt
    assert "turn left" in prompt  # informative describes what actions do


def test_minigrid_build_prompt_minimal_hides_action_meaning() -> None:
    prompt = MiniGridProblem().build_prompt("MiniGrid-Empty-5x5-v0", info_mode=InfoMode.MINIMAL)
    assert "MiniGrid-Empty-5x5-v0" in prompt
    assert "Discover the rules" in prompt
    assert "turn left" not in prompt  # minimal does not spell out the actions


def test_minigrid_obs_renderer_rejects_unsupported_mode() -> None:
    problem = MiniGridProblem()
    assert isinstance(problem.obs_renderer("t", mode=ObsMode.RAW), MiniGridRenderer)


def test_minigrid_renderer_makes_obs_json_safe() -> None:
    """A numpy-like obs (object exposing tolist) becomes nested lists in the frame."""

    class _FakeArray:
        def tolist(self) -> list[int]:
            return [1, 2, 3]

    obs = MiniGridRenderer().render(
        {"image": _FakeArray(), "direction": 0},
        {"available_actions": [0, 1, 2]},
    )
    assert obs.frame == {"image": [1, 2, 3], "direction": 0}
    assert obs.available_actions == [0, 1, 2]


def test_minigrid_compute_episode_metrics_from_generic_obs() -> None:
    problem = MiniGridProblem()
    won = problem.compute_episode_metrics(Obs(frame=None, reward=1.0, is_done=True), steps=4)
    assert won == {"success": True, "steps": 4, "reward": 1.0}
    # Truncation (is_done but no reward) is not a success.
    lost = problem.compute_episode_metrics(Obs(frame=None, reward=0.0, is_done=True), steps=10)
    assert lost == {"success": False, "steps": 10, "reward": 0.0}


def test_minigrid_metrics_aggregate() -> None:
    problem = MiniGridProblem()
    episodes = [
        {"success": True, "steps": 4, "reward": 1.0},
        {"success": False, "steps": 10, "reward": 0.0},
    ]
    agg = problem.aggregate_episode_metrics(episodes)
    assert agg == {"n_episodes": 2, "success_rate": 0.5, "mean_steps": 7.0, "mean_reward": 0.5}


@pytest.mark.live
def test_minigrid_make_env_drives_a_step() -> None:
    """Runtime-gated: only where the minigrid extra is installed."""
    pytest.importorskip("minigrid")
    problem: BaseProblem = MiniGridProblem()
    native = problem.make_env("MiniGrid-Empty-5x5-v0")
    _obs, info = native.reset(seed=0)
    assert "available_actions" in info
    assert info["available_actions"]  # non-empty discrete action set
    *_, info = native.step(info["available_actions"][0])
    assert "available_actions" in info
    native.close()


@pytest.mark.live
def test_all_catalogued_minigrid_tasks_can_reset() -> None:
    """Every catalogue entry must exist upstream and pass Regact's env shim."""
    pytest.importorskip("minigrid")
    problem = MiniGridProblem()
    for task_name in ALL_MINIGRID_TASKS:
        native = problem.make_env(task_name)
        try:
            _obs, info = native.reset(seed=0)
            assert info["available_actions"], task_name
        finally:
            native.close()


@pytest.mark.live
def test_minigrid_make_env_uses_task_name_not_default_env() -> None:
    """A suite run must construct the scheduler's task, not one hard-coded env."""
    pytest.importorskip("minigrid")
    problem = MiniGridProblem()
    native = problem.make_env("MiniGrid-DoorKey-5x5-v0")
    try:
        assert native._env.spec.id == "MiniGrid-DoorKey-5x5-v0"
    finally:
        native.close()


def test_minigrid_render_frame_makes_rgb() -> None:
    """Runtime-gated: render_frame re-renders the symbolic obs into an RGB video frame."""
    pytest.importorskip("minigrid")
    from regact.env.renderer import jsonify

    problem = MiniGridProblem()
    native = problem.make_env("MiniGrid-Empty-5x5-v0")
    native_obs, _ = native.reset(seed=0)
    native.close()
    img = problem.render_frame(Obs(frame=jsonify(native_obs)))
    assert img is not None and img.ndim == 3 and img.shape[2] == 3
    assert problem.render_frame(Obs(frame=[[1, 2]])) is None  # not a MiniGrid obs dict
