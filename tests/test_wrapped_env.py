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


def test_milestone_detector_surfaces_in_obs() -> None:
    """Each step's obs carries that step's milestones; nothing is left to drain."""

    def detector(e: WrappedEnv) -> list[str]:
        return ["goal reached"] if e.last_reward == 1.0 else []

    env = _wrap(milestone_detector=detector)
    env.reset()
    obs = env.step(1)
    assert obs.info["milestones"] == []  # not at the goal yet
    env.step(1)
    obs = env.step(1)
    assert obs.info["milestones"] == ["goal reached"]  # surfaced on the step that reached it
    assert env.drain_milestones() == []  # the step already drained it


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


@pytest.mark.parametrize("native_reward", [0, 0.0, None])
def test_initial_reset_and_noop_share_one_observation_contract(native_reward):
    """A no-op must not produce a new serialized observation just because reset differs."""
    import json

    class Stationary:
        info = {"available_actions": [0], "milestones": ["stale native event"]}

        def current(self):
            return {"cell": 1}, self.info

        def reset(self, **kwargs):
            return self.current()

        def step(self, action):
            return {"cell": 1}, native_reward, False, False, self.info

    native = Stationary()
    env = WrappedEnv(native, task_name="stationary", renderer=RawRenderer())
    initial = env.last_obs
    assert initial.reward == env.last_reward == 0.0
    reset = env.reset()
    after = env.step(0)
    assert json.dumps(initial.to_json(), sort_keys=True) == json.dumps(
        reset.to_json(), sort_keys=True
    )
    assert json.dumps(reset.to_json(), sort_keys=True) == json.dumps(
        after.to_json(), sort_keys=True
    )
    assert after.info["milestones"] == []
    assert env.last_reward == after.reward == 0.0
    assert native.info["milestones"] == ["stale native event"]  # do not mutate native metadata


def test_reset_clears_previous_reward_and_milestone_without_changing_prior_observation():
    env = _wrap(milestone_detector=lambda e: ["goal reached"] if e.is_done else [])
    initial = env.reset().to_json()
    for _ in range(3):
        final = env.step(1)
    assert final.reward == 1.0 and final.info["milestones"] == ["goal reached"]
    assert env.reset().to_json() == initial
    assert env.last_reward == 0.0
    assert final.reward == 1.0 and final.info["milestones"] == ["goal reached"]
    assert env.drain_milestones() == []
