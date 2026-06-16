"""Tests for WrappedEnv (over the deterministic FakeNativeEnv)."""

from collections.abc import Callable

import pytest

from regact.env.renderer import RawRenderer
from regact.env.wrapped_env import WrappedEnv
from regact.obs.errors import ErrorCategory, RegactError
from regact.testing.fakes import FakeNativeEnv


def _wrap(
    *,
    record_frames: bool = False,
    milestone_detector: Callable[[WrappedEnv], list[str]] | None = None,
) -> WrappedEnv:
    return WrappedEnv(
        FakeNativeEnv(goal=3),
        task_name="fake",
        renderer=RawRenderer(),
        record_frames=record_frames,
        milestone_detector=milestone_detector,
    )


def test_reset_returns_obs_and_zero_count() -> None:
    env = _wrap()
    obs = env.reset()
    assert obs.frame == {"pos": 0, "grid": [1, 0, 0, 0]}
    assert obs.is_done is False
    assert env.action_count == 0


def test_step_increments_and_sets_reward_done() -> None:
    env = _wrap()
    env.reset()
    obs = env.step(1)
    assert env.action_count == 1
    assert obs.reward == 0.0
    assert obs.is_done is False


def test_episode_reaches_goal() -> None:
    env = _wrap()
    env.reset()
    obs = env.last_obs
    for _ in range(3):
        obs = env.step(1)
    assert obs is not None
    assert env.is_done is True
    assert obs.reward == 1.0
    assert obs.available_actions == [0, 1]


def test_record_frames() -> None:
    env = _wrap(record_frames=True)
    env.reset()
    env.step(1)
    assert len(env.frame_trace) == 2  # reset + 1 step
    assert env.frame_trace[0] == [1, 0, 0, 0]


def test_milestone_detector_drain() -> None:
    def detector(e: WrappedEnv) -> list[str]:
        return ["goal reached"] if e.last_reward == 1.0 else []

    env = _wrap(milestone_detector=detector)
    env.reset()
    for _ in range(3):
        env.step(1)
    assert env.drain_milestones() == ["goal reached"]
    assert env.drain_milestones() == []  # cleared


def test_unexpected_arity_raises() -> None:
    class BadEnv:
        def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, object]]:
            return 0, {}

        def step(self, action: int) -> tuple[int, float, bool]:  # 3-tuple, invalid
            return 0, 0.0, False

    env = WrappedEnv(BadEnv(), task_name="bad", renderer=RawRenderer())
    env.reset()
    with pytest.raises(RegactError) as exc:
        env.step(0)
    assert exc.value.category is ErrorCategory.ENV_RUNTIME
