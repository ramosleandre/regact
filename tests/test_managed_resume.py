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
    assert failure.value.differences["frame"]
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


def killed(c):
    """The task's process died: nothing is closed properly and no final status is written."""
    c.env.close()
    c.store.close()
    c.closed = True
    return c


def reopened(c, lifecycle, **config):
    """A new coordinator on the task directory of ``c``, as a later process would build it."""
    import dataclasses

    cfg = dataclasses.replace(c.config, resume=str(c.output), **config)
    return type(c)(
        cfg, c.options, fresh_session(lifecycle), c.problem, c.task, c.output, c.workdir
    )


@pytest.mark.parametrize("lifecycle", list(Lifecycle))
def test_a_task_resumes_where_its_store_ends(hidden_rig, lifecycle):
    c = played(hidden_rig, lifecycle)
    c.persist()
    before = c.store.summary()
    expected_next = c.env.live.step(1).to_json() if lifecycle is Lifecycle.SINGLE_INSTANCE else None
    killed(c)

    r = reopened(c, lifecycle)
    try:
        assert r.store.summary() == before
        assert (r.current_id, r.initial_id, r.reset_actions) == (
            c.current_id,
            c.initial_id,
            c.reset_actions,
        )
        assert r.milestones == c.milestones and r.initial_collection == c.initial_collection
        assert r.phase == "Active Exploration"
        assert (r.accepted["cwm_version"], r.accepted["bundle"]) == (
            c.accepted["cwm_version"],
            c.accepted["bundle"],
        )
        exploration(r, (1,))
        result = r.tool("RunController", {})
        assert result["real_actions"] == 1 and "error" not in result, result
        if expected_next is not None:
            # c's game took one unrecorded step above; the rebuilt one is at the recorded position.
            assert r.store.observation(r.current_id) == expected_next
    finally:
        r.close("test_finished")


def test_a_stale_status_file_does_not_lose_the_position_or_the_accepted_cwm(hidden_rig):
    c = hidden_rig(lifecycle=Lifecycle.SINGLE_INSTANCE, target=3)
    stale = c.root / "status.json"
    saved = stale.read_text()  # written before any CWM was accepted
    hidden_model(c.workdir)
    assert c.tool("UpdateCodeWorldModel", {})["accepted"]
    exploration(c, (1, 1))
    assert c.tool("RunController", {})["real_actions"] == 2
    c.tool("ResetEnvironment", {})
    stale.write_text(saved)  # the process died before any of that reached status.json
    killed(c)

    r = reopened(c, Lifecycle.SINGLE_INSTANCE)
    try:
        assert r.reset_actions == 1 and r.current_id == c.current_id
        # The phase and the accepted CWM come from the store: a reset keeps Active Exploration.
        assert r.phase == "Active Exploration"
        assert r.accepted["cwm_version"] == c.accepted["cwm_version"]
        exploration(r, (1, 1))
        result = r.tool("RunController", {})
        assert result["real_actions"] == 2 and "error" not in result, result
    finally:
        r.close("test_finished")


def test_steps_the_accepted_cwm_never_saw_are_checked_on_resume(hidden_rig):
    c = played(hidden_rig, Lifecycle.SINGLE_INSTANCE)
    c.persist()
    # The process dies mid-RunController: two real steps are recorded, nothing else.
    for _ in range(2):
        c._step(1)
    killed(c)
    r = reopened(c, Lifecycle.SINGLE_INSTANCE)
    try:
        assert r.phase == "Active Exploration"
        # The carried State is rebuilt through those two steps before the controller acts.
        exploration(r, (1,))
        result = r.tool("RunController", {})
        assert result["real_actions"] == 1 and "error" not in result, result
    finally:
        r.close("test_finished")


def test_a_recorded_task_is_not_reopened_without_resume_or_by_another_commit(
    hidden_rig, monkeypatch
):
    import dataclasses

    c = played(hidden_rig, Lifecycle.SINGLE_INSTANCE)
    killed(c)
    fresh = dataclasses.replace(c.config, resume=None)
    with pytest.raises(RuntimeError, match="resume="):
        type(c)(fresh, c.options, fresh_session(Lifecycle.SINGLE_INSTANCE), c.problem, c.task,
                c.output, c.workdir)
    from regact.protocols.managed import session

    monkeypatch.setattr(session, "regact_commit", lambda: "f" * 40)
    with pytest.raises(RuntimeError, match="recorded by regact"):
        reopened(c, Lifecycle.SINGLE_INSTANCE)
    reopened(c, Lifecycle.SINGLE_INSTANCE, resume_any_version=True).close("test_finished")


def test_a_store_without_a_recorded_commit_needs_the_explicit_override(hidden_rig):
    import sqlite3

    c = played(hidden_rig, Lifecycle.SINGLE_INSTANCE)
    killed(c)
    with sqlite3.connect(c.root / "experience.sqlite3") as db:  # as stores older than the record
        db.execute("UPDATE meta SET value='' WHERE key='regact_commit'")
    with pytest.raises(RuntimeError, match="unknown commit"):
        reopened(c, Lifecycle.SINGLE_INSTANCE)
    reopened(c, Lifecycle.SINGLE_INSTANCE, resume_any_version=True).close("test_finished")


def test_only_an_experiment_deadline_makes_a_walltime_end_resumable():
    from regact.session.state import ExperimentState

    state = ExperimentState(problem_name="fake", task_name="counter")
    assert state.resumable()  # no exit reason: the process died
    for reason, detail, expected in (
        ("walltime_limit", "task_budget", False),
        ("walltime_limit", "experiment_deadline", True),
        ("agent_api", None, True),
        ("solved", None, False),
        ("tool_call_limit", None, False),
    ):
        state.exit_reason, state.exit_detail = reason, detail
        assert state.resumable() is expected, reason


class Levels:
    """Each action completes one of the game's 5 levels; a cap ends the game earlier and reports
    only that many, as ARC's level cap does."""

    def __init__(self, max_levels):
        self.max_levels, self.n = max_levels, 0

    def current(self):
        total = 5 if self.max_levels is None else min(5, self.max_levels)
        info = {"available_actions": [1], "levels_completed": min(self.n, total)}
        info["win_levels"] = total
        return [self.n] * 30, info

    def reset(self, seed=None):
        self.n = 0
        return self.current()

    def step(self, a):
        self.n += 1
        obs, info = self.current()
        won = info["levels_completed"] >= info["win_levels"]
        return obs, float(won), won, False, info

    def close(self):
        pass


def leveled(tmp_path, max_levels, previous=None, **config):
    """A vanilla task on a game capped at ``max_levels``; with ``previous``, its later launch."""
    import dataclasses
    import json

    from regact.config.schema import (
        AgentConfig,
        AgentName,
        ProblemConfig,
        ProtocolConfig,
        RunConfig,
    )
    from regact.features.base import FeatureContext
    from regact.protocols.registry import build_protocol
    from test_cwm_protocol import Problem

    class LevelProblem(Problem):
        env_cap = max_levels

        def make_env(self, task):
            return Levels(self.env_cap)

        def cap_levels(self, cap, native_env=None):
            self.env_cap = cap
            if native_env is not None:
                native_env.max_levels = cap

    root, work = tmp_path / "task", tmp_path / "task" / "workdir"
    problem = ProblemConfig(
        "fake", seed=0, lifecycle=Lifecycle.SINGLE_INSTANCE, kwargs={"max_levels": max_levels}
    )
    cfg = RunConfig(
        AgentConfig(AgentName.SCRIPTED),
        problem,
        protocol=ProtocolConfig("vanilla", {"n_unique_observations_in_initial_collection": 1}),
    )
    if previous is None:
        work.mkdir(parents=True)
        (root / "config.json").write_text(json.dumps({"problem": {"kwargs": problem.kwargs}}))
    else:
        cfg = dataclasses.replace(cfg, resume=str(root), **config)
    definition = build_protocol(cfg)
    lp = LevelProblem()
    env = EnvSession(
        make_native=lambda: lp.make_env("counter"),
        key="counter",
        renderer=lp.obs_renderer(),
        lifecycle=SingleInstancePolicy(),
    )
    c = definition.coordinator_type(cfg, definition.options, env, lp, "counter", root, work)
    if previous is None:
        for f in definition.templates(
            FeatureContext(problem_name="fake", task_name="counter", workdir=str(work))
        ):
            (work / f.relpath).parent.mkdir(parents=True, exist_ok=True)
            (work / f.relpath).write_text(f.content)
        c.collect_initial()
    return c


def test_an_extended_task_goes_on_past_its_level_cap_in_a_new_episode(tmp_path):
    c = leveled(tmp_path, 2)
    exploration(c, (1, 1))
    assert c.tool("RunController", {})["real_actions"] == 2
    assert c.terminal == "solved"
    won = c.store.observation(c.current_id)
    assert won["is_done"] and won["info"]["win_levels"] == 2
    c.close("solved")

    r = leveled(tmp_path, None, previous=c, extend=True)
    try:
        assert r.terminal is None
        first, extension = r.store.episodes()
        assert extension["purpose"] == "extension" and extension["episode_id"] == r.episode
        # The recorded part stays as it was: the cap-2 win ends the first episode.
        assert r.store.observation(first["initial_obs_id"])["info"]["win_levels"] == 2
        ended = r.store.db.execute("SELECT stop_reason FROM episodes WHERE id=1").fetchone()[0]
        assert ended == "environment_done"
        start = r.store.observation(r.current_id)
        assert start["frame"] == won["frame"] and not start["is_done"]
        assert start["info"]["levels_completed"] == 2 and start["info"]["win_levels"] == 5
        exploration(r, (1,))
        assert r.tool("RunController", {})["real_actions"] == 1
        stepped = r.store.observation(r.current_id)["info"]
        assert (stepped["levels_completed"], stepped["win_levels"]) == (3, 5)
        r.persist()
    finally:
        killed(r)

    # A later resume replays the extension too, and does not extend again.
    again = leveled(tmp_path, None, previous=r)
    try:
        assert [e["purpose"] for e in again.store.episodes()] == [first["purpose"], "extension"]
        assert again.current_id == r.current_id
        exploration(again, (1, 1))
        assert again.tool("RunController", {})["real_actions"] == 2
        assert again.terminal == "solved"
    finally:
        again.close("test_finished")
