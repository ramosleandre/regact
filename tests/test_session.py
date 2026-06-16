"""Tests for EnvSession (lifecycle-managed env access)."""

from regact.env.lifecycle import EnvLifecyclePolicy, MultiInstancePolicy, SingleInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.session import EnvSession
from regact.testing.fakes import FakeNativeEnv


def _session(lifecycle: EnvLifecyclePolicy) -> EnvSession:
    return EnvSession(
        make_native=lambda: FakeNativeEnv(goal=3),
        key="fake",
        renderer=RawRenderer(),
        lifecycle=lifecycle,
    )


def test_session_drives_episode() -> None:
    session = _session(MultiInstancePolicy())
    env = session.make()
    obs = session.reset()
    assert obs.is_done is False
    while not env.is_done:
        obs = env.step(1)
    assert obs.reward == 1.0
    assert env.action_count == 3


def test_single_instance_session_same_handle() -> None:
    session = _session(SingleInstancePolicy())
    a = session.make()
    b = session.make()
    assert a is b
    assert session.live is a
