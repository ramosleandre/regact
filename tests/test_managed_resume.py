"""Rebuilding a task's live environment from its experience store."""

import pytest

from regact.config.schema import Lifecycle
from regact.env.lifecycle import MultiInstancePolicy, SingleInstancePolicy
from regact.env.session import EnvSession
from regact.protocols.managed.resume import RebuildMismatch, rebuild_environment
from test_cwm_hidden_state import HiddenNative, HiddenProblem, hidden_model
from test_cwm_hidden_state import hidden_rig as hidden_rig
from test_cwm_protocol import exploration

pytestmark = pytest.mark.integration


def fresh_session(lifecycle, native=HiddenNative):
    problem = HiddenProblem()
    return EnvSession(
        make_native=native,
        key="counter",
        renderer=problem.obs_renderer(),
        lifecycle=SingleInstancePolicy()
        if lifecycle is Lifecycle.SINGLE_INSTANCE
        else MultiInstancePolicy(),
    )


def played(hidden_rig, lifecycle):
    """A task with a collection, two explorations and, in single_instance, a reset between."""
    c = hidden_rig(lifecycle=lifecycle, target=3)
    hidden_model(c.workdir)
    assert c.tool("UpdateCodeWorldModel", {})["accepted"]
    exploration(c, (1, 1, 1))
    assert c.tool("RunController", {})["real_actions"] == 3
    if lifecycle is Lifecycle.SINGLE_INSTANCE:
        c.tool("ResetEnvironment", {})
    exploration(c, (1, 1))
    assert c.tool("RunController", {})["real_actions"] == 2
    return c


@pytest.mark.parametrize("lifecycle", list(Lifecycle))
def test_a_fresh_environment_replays_to_the_recorded_position(hidden_rig, lifecycle):
    c = played(hidden_rig, lifecycle)
    single = lifecycle is Lifecycle.SINGLE_INSTANCE
    env = fresh_session(lifecycle)
    last = rebuild_environment(c.store, env, seed=0, single_instance=single)
    assert last == c.env.live.last_obs.to_json() == c.store.observation(c.current_id)
    # The hidden counter is restored too, not only the screen: the next real step agrees.
    assert env.live.step(1).to_json() == c.env.live.step(1).to_json()
    assert env.live.episode_action_count == c.env.live.episode_action_count
    if single:
        assert env.total_action_count == c.env.total_action_count
    env.close()


def test_a_different_game_is_refused_at_the_first_difference(hidden_rig):
    c = played(hidden_rig, Lifecycle.SINGLE_INSTANCE)

    class Faster(HiddenNative):
        def step(self, a):
            self.n += 1
            return super().step(a)

    env = fresh_session(Lifecycle.SINGLE_INSTANCE, native=Faster)
    with pytest.raises(RebuildMismatch) as failure:
        rebuild_environment(c.store, env, seed=0, single_instance=True)
    # The first screen is the same; the first action already lands one screen further.
    assert failure.value.where["episode_id"] == 1 and failure.value.where["step"] == 1
    assert failure.value.differences["differences"]
    env.close()


def test_an_environment_already_in_use_is_refused(hidden_rig):
    c = played(hidden_rig, Lifecycle.SINGLE_INSTANCE)
    env = fresh_session(Lifecycle.SINGLE_INSTANCE)
    env.make()
    with pytest.raises(ValueError, match="never made"):
        rebuild_environment(c.store, env, seed=0, single_instance=True)
    env.close()


def test_an_empty_store_rebuilds_nothing(tmp_path):
    from regact.protocols.cwm.store import ExperienceStore

    store = ExperienceStore(tmp_path / "experience.sqlite3")
    env = fresh_session(Lifecycle.SINGLE_INSTANCE)
    try:
        assert rebuild_environment(store, env, seed=0, single_instance=True) is None
        assert env.live is None
    finally:
        store.close()
