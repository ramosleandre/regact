"""Tests for the eval result schema."""

from regact.obs.errors import ErrorCategory
from regact.obs.result import EpisodeResult, EvalResult


def test_eval_result_to_json() -> None:
    res = EvalResult(
        task="ls20",
        aggregate={"n_episodes": 2, "success_rate": 0.5},
        episodes=[
            EpisodeResult(episode=0, metrics={"success": True, "steps": 7}),
            EpisodeResult(episode=1, error="boom", error_category=ErrorCategory.AGENT_SOLUTION),
        ],
        executor="in_process",
    )
    out = res.to_json()
    assert out["task"] == "ls20"
    assert out["executor"] == "in_process"
    assert out["episodes"][0]["metrics"]["steps"] == 7
    assert out["episodes"][1]["error_category"] == "agent_solution"
    assert out["error_category"] is None


def test_episode_result_defaults() -> None:
    ep = EpisodeResult(episode=0)
    out = ep.to_json()
    assert out["metrics"] == {}
    assert out["milestones"] == []
    assert out["error"] is None
