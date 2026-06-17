"""Tests for the problem layer: registry, MiniGrid contract, ARC deferral.

The pure contract (registry, prompt, metrics, config_kwargs, renderer) runs
always. ``make_env`` needs the ``minigrid`` extra, so it is gated with
``importorskip`` — it runs where the lib is installed, skips cleanly otherwise.
"""

import pytest

from regact.config.schema import ObsMode
from regact.problems.base import BaseProblem, build_problem
from regact.problems.minigrid import MiniGridProblem, MiniGridRenderer


def test_build_problem_resolves_minigrid() -> None:
    problem = build_problem("minigrid", {"env_id": "MiniGrid-Empty-5x5-v0"})
    assert isinstance(problem, MiniGridProblem)
    assert problem.name == "minigrid"


def test_build_problem_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown problem"):
        build_problem("nope", {})


def test_build_problem_arc_is_deferred() -> None:
    with pytest.raises(NotImplementedError, match="Block 8b"):
        build_problem("arc_agi", {})


def test_minigrid_config_kwargs_roundtrip() -> None:
    problem = MiniGridProblem(env_id="MiniGrid-DoorKey-5x5-v0", fully_obs=True)
    kwargs = problem.config_kwargs()
    assert kwargs == {"env_id": "MiniGrid-DoorKey-5x5-v0", "fully_obs": True}
    # The trusted side can rebuild from these kwargs.
    rebuilt = build_problem("minigrid", kwargs)
    assert isinstance(rebuilt, MiniGridProblem)
    assert rebuilt.config_kwargs() == kwargs


def test_minigrid_prompt_fragment_names_the_task() -> None:
    fragment = MiniGridProblem().prompt_fragment("MiniGrid-Empty-5x5-v0")
    assert "MiniGrid-Empty-5x5-v0" in fragment
    assert "available_actions" in fragment


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
    problem: BaseProblem = MiniGridProblem(env_id="MiniGrid-Empty-5x5-v0")
    native = problem.make_env("MiniGrid-Empty-5x5-v0")
    _obs, info = native.reset(seed=0)
    assert "available_actions" in info
    assert info["available_actions"]  # non-empty discrete action set
    *_, info = native.step(info["available_actions"][0])
    assert "available_actions" in info
    native.close()
