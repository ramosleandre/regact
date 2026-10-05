"""CWM v6: get_initial_state at an episode start, then step carries the State, hidden part included."""

import json

import pytest

from regact.config.schema import (
    AgentConfig,
    AgentName,
    Lifecycle,
    ProblemConfig,
    ProtocolConfig,
    RunConfig,
    redacted_config_dict,
)
from regact.env.lifecycle import MultiInstancePolicy, SingleInstancePolicy
from regact.env.session import EnvSession
from regact.features.base import FeatureContext
from regact.problems import base as problems
from regact.protocols.cwm.viewer import Playback
from regact.protocols.cwm.worker import Worker
from regact.protocols.registry import build_protocol
from test_cwm_protocol import Native, Problem, exploration

pytestmark = pytest.mark.integration


class HiddenNative(Native):
    """The screen shows n // 2: one frame cannot tell n = 2 from n = 3 (a coarse counter)."""

    def step(self, a):
        self.n += 1
        done = self.n >= 20
        return [self.n // 2] * 30, float(done), done, False, {"available_actions": [1]}


class HiddenProblem(Problem):
    def make_env(self, task):
        return HiddenNative()


class LeakyNative(HiddenNative):
    """A reset does not restore everything: the second episode counts twice as fast."""

    def reset(self, seed=None):
        self.resets = getattr(self, "resets", 0) + 1
        return super().reset(seed)

    def step(self, a):
        self.n += self.resets - 1
        return super().step(a)


RENDER = (
    'def render(s): return {"frame":[s.n//2]*30, "reward":float(s.n>=20),"is_done":s.n>=20,'
    '"available_actions":[1],"info":{"available_actions":[1],"milestones":[]}}\n'
)


def hidden_model(root, *, step="State(s.n+1)", parse="State(2*o['frame'][0])", extra=""):
    d = root / "world_model"
    d.mkdir(exist_ok=True, parents=True)
    (d / "model_state.py").write_text(
        "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass State:\n n:int\n"
    )
    (d / "model_initial_state.py").write_text(
        f"from world_model.model_state import State\ndef get_initial_state(o): return {parse}\n"
    )
    (d / "model_render.py").write_text(RENDER)
    (d / "model_transition.py").write_text(
        f"from world_model.model_state import State\ndef step(s,a): return {step}\n{extra}"
    )


@pytest.fixture
def hidden_rig(tmp_path):
    coordinators = []

    def make(lifecycle=Lifecycle.MULTI_INSTANCE, target=3, **options):
        root = tmp_path / str(len(coordinators))
        work = root / "workdir"
        work.mkdir(parents=True)
        problem = HiddenProblem()
        cfg = RunConfig(
            AgentConfig(AgentName.SCRIPTED),
            ProblemConfig("fake", seed=0, lifecycle=lifecycle),
            protocol=ProtocolConfig(
                "cwm",
                {
                    "n_unique_observations_in_initial_collection": target,
                    "n_tmp_images_saved_per_exploration": 0,
                    **options,
                },
            ),
        )
        definition = build_protocol(cfg)
        env = EnvSession(
            make_native=lambda: problem.make_env("counter"),
            key="counter",
            renderer=problem.obs_renderer(),
            lifecycle=SingleInstancePolicy()
            if lifecycle is Lifecycle.SINGLE_INSTANCE
            else MultiInstancePolicy(),
        )
        c = definition.coordinator_type(cfg, definition.options, env, problem, "counter", root, work)
        definition.coordinator = c
        for f in definition.templates(
            FeatureContext(problem_name="fake", task_name="counter", workdir=str(work))
        ):
            path = work / f.relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f.content)
        coordinators.append(c)
        c.collect_initial()
        return c

    yield make
    for c in coordinators:
        c.close("test_finished")


@pytest.mark.parametrize("lifecycle", list(Lifecycle))
def test_a_hidden_counter_is_accepted_when_step_carries_it(hidden_rig, lifecycle):
    c = hidden_rig(lifecycle=lifecycle, target=3)  # collection: n = 0..4, screens 0 0 1 1 2
    hidden_model(c.workdir)
    accepted = c.tool("UpdateCodeWorldModel", {})
    assert accepted["accepted"], accepted
    exploration(c, (1,))
    first = c.tool("RunController", {})
    assert first["real_actions"] == 1 and "error" not in first, first
    exploration(c, (1, 1))
    second = c.tool("RunController", {})
    # single_instance: the second call starts at n = 5, behind screen 2. A State read from
    # screen 2 (n = 4) would mispredict the next screen; the one carried from the first call
    # does not.
    assert second["real_actions"] == 2 and "error" not in second, second
    # The same screen and action led to different screens, as hidden state does: no terminal.
    assert c.terminal is None


def test_a_step_that_ignores_the_hidden_counter_is_refused(hidden_rig):
    c = hidden_rig(lifecycle=Lifecycle.SINGLE_INSTANCE, target=3)
    hidden_model(c.workdir, step="State(s.n+2)")  # skips the hidden odd values
    refused = c.tool("UpdateCodeWorldModel", {})
    assert refused["status"] == "Refused" if "status" in refused else not refused["accepted"]
    assert refused["failures"] == {"prediction_mismatch": 1}
    [example] = refused["counterexamples"]
    assert (example["episode_id"], example["step"]) == (1, 1)


def test_a_wrong_hidden_update_is_caught_where_it_shows(hidden_rig):
    c = hidden_rig(lifecycle=Lifecycle.SINGLE_INSTANCE, target=4)  # screens 0 0 1 1 2 2 3
    # Freezes the counter at n = 2. Step 3 still renders screen 1 (2 and 3 both show 1); the
    # replay catches the frozen value at step 4.
    hidden_model(c.workdir, step="State(min(s.n+1, 2) if s.n < 2 else s.n)")
    refused = c.tool("UpdateCodeWorldModel", {})
    [example] = refused["counterexamples"]
    assert example["kind"] == "prediction_mismatch" and example["step"] == 4, example


def test_a_reset_starts_a_new_episode_from_its_first_observation(hidden_rig):
    c = hidden_rig(lifecycle=Lifecycle.SINGLE_INSTANCE, target=3)  # episode 1: n = 0..4
    hidden_model(c.workdir)
    assert c.tool("UpdateCodeWorldModel", {})["accepted"]
    c.tool("ResetEnvironment", {})
    exploration(c, (1, 1, 1))
    result = c.tool("RunController", {})  # starts from get_initial_state(reset screen): n = 0
    assert result["real_actions"] == 3 and "error" not in result, result
    assert c.terminal is None
    calls = []
    original = Worker.call

    def counting(self, op, **kwargs):
        calls.append(op)
        return original(self, op, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Worker, "call", counting)
        again = c.tool("UpdateCodeWorldModel", {})
    # Episode 2 repeats episode 1's first three actions from the same screen: validated once.
    assert again["accepted"] and again["episodes_checked"] == 2 and again["steps_checked"] == 7
    assert calls.count("step") == 4 + 1  # episode 1's four steps + one repeatability sample
    assert calls.count("get_initial_state") == 2  # one shared start, checked twice


def test_something_hidden_surviving_a_reset_stops_the_task(hidden_rig, monkeypatch):
    monkeypatch.setattr(HiddenProblem, "make_env", lambda self, task: LeakyNative())
    c = hidden_rig(lifecycle=Lifecycle.SINGLE_INSTANCE, target=3)  # screens 0 0 1 1 2
    hidden_model(c.workdir)
    assert c.tool("UpdateCodeWorldModel", {})["accepted"]
    c.tool("ResetEnvironment", {})  # the same first screen, but the game now counts by 2
    exploration(c, (1, 1))
    result = c.tool("RunController", {})
    # The same first screen and the same action gave screen 0 in episode 1 and screen 1 now.
    assert c.terminal == "observation_determinism_violation", result


def test_only_repeated_histories_are_kept_during_validation(hidden_rig):
    c = hidden_rig(lifecycle=Lifecycle.SINGLE_INSTANCE, target=3)
    assert c.store.repeated_histories() == set()  # one episode: nothing to share
    c.tool("ResetEnvironment", {})
    hidden_model(c.workdir)
    assert c.tool("UpdateCodeWorldModel", {})["accepted"]
    exploration(c, (1, 1))
    assert c.tool("RunController", {})["real_actions"] == 2
    start = c.store.episodes()[1]["start_hash"]
    shared = [item["history_hash"] for item in c.store.episode_steps(2)]
    assert c.store.repeated_histories() == {start, *shared}


def test_replays_share_validated_prefixes(hidden_rig, monkeypatch):
    c = hidden_rig(lifecycle=Lifecycle.MULTI_INSTANCE, target=3)
    hidden_model(c.workdir)
    assert c.tool("UpdateCodeWorldModel", {})["accepted"]
    for _ in range(3):  # three calls replaying the same 4 actions from the same start
        exploration(c, (1, 1, 1, 1))
        assert c.tool("RunController", {})["real_actions"] == 4
    calls = []
    original = Worker.call

    def counting(self, op, **kwargs):
        calls.append(op)
        return original(self, op, **kwargs)

    monkeypatch.setattr(Worker, "call", counting)
    result = c.tool("UpdateCodeWorldModel", {})
    # The collection (4 actions) and the three calls (4 actions each) all start from the same
    # screen with the same actions: one distinct history, stepped once.
    assert result["accepted"] and result["episodes_checked"] == 4
    assert result["steps_checked"] == 4 + 3 * 4
    assert calls.count("step") == 4 + 1  # + the sampled repeatability check on step 1
    assert calls.count("get_initial_state") == 2  # one shared start, checked twice


def test_a_bug_repeated_by_identical_episodes_is_reported_once(hidden_rig):
    c = hidden_rig(lifecycle=Lifecycle.MULTI_INSTANCE, target=3)
    hidden_model(c.workdir)
    assert c.tool("UpdateCodeWorldModel", {})["accepted"]
    for _ in range(3):  # with the collection: four episodes sharing their first four actions
        exploration(c, (1, 1, 1, 1))
        assert c.tool("RunController", {})["real_actions"] == 4
    hidden_model(c.workdir, step="State(s.n+1 if s.n != 2 else 2)")  # wrong from step 3 on
    refused = c.tool("UpdateCodeWorldModel", {})
    assert refused["failures"] == {"prediction_mismatch": 1}, refused
    [example] = refused["counterexamples"]
    assert (example["episode_id"], example["step"]) == (1, 4)


def test_validation_budget_scales_with_recorded_steps(hidden_rig, monkeypatch):
    c = hidden_rig(target=3)
    assert c.validation_budget() == 90  # the floor while the dataset is small
    monkeypatch.setattr(
        c.store, "summary", lambda: {"n_total_transitions": 6000, "n_unique_observations": 1}
    )
    assert c.validation_budget() == 180  # 30 s per 1,000 recorded steps
    c.options.execution.max_seconds_per_UpdateCodeWorldModel_per_1000_steps = None
    assert c.validation_budget() is None


def test_a_state_that_grows_along_an_episode_fails_compactness(hidden_rig):
    c = hidden_rig(lifecycle=Lifecycle.SINGLE_INSTANCE, target=6)
    c.options.threshold_max_state_obs_size_ratio = 100
    d = c.workdir / "world_model"
    hidden_model(c.workdir)
    (d / "model_state.py").write_text(
        "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass State:\n n:int\n log:tuple\n"
    )
    (d / "model_initial_state.py").write_text(
        "from world_model.model_state import State\n"
        "def get_initial_state(o): return State(2*o['frame'][0], ())\n"
    )
    (d / "model_transition.py").write_text(
        "from world_model.model_state import State\n"
        "def step(s,a): return State(s.n+1, s.log + ('x'*12,))\n"
    )
    sizes = c.tool("UpdateCodeWorldModel", {}) and c.accepted["validation"]
    average, largest = sizes["state_obs_size_ratio"], sizes["largest_ratio_state"]["ratio"]
    assert average < largest
    # A threshold the average passes but the State at the end of the episode does not.
    c.options.threshold_max_state_obs_size_ratio = (average + largest) / 2
    refused = c.tool("UpdateCodeWorldModel", {})
    assert refused["failures"] == {"compression_ratio": 1}, refused
    assert refused["counterexamples"][0]["largest_state"]["ratio"] == largest


def test_plans_replay_from_their_recorded_start_state(hidden_rig, monkeypatch):
    c = hidden_rig(lifecycle=Lifecycle.SINGLE_INSTANCE, target=3, planner={"enabled": True})
    hidden_model(c.workdir)
    assert c.tool("UpdateCodeWorldModel", {})["accepted"]
    exploration(c, (1,))
    assert c.tool("RunController", {})["real_actions"] == 1  # n = 5, behind screen 2
    (c.workdir / "goal.py").write_text('"""Reach n = 7."""\ndef achieved(s): return s.n == 7\n')
    plan = c.tool("PlanInCWM", {})
    assert plan["achieved"] and plan["n_actions"] == 2, plan
    record = c.store.record_payload(plan["plan_id"])
    assert record["initial_state"]["fields"]["n"] == 5  # the carried State; screen 2 reads as 4
    (c.output / "config.json").write_text(json.dumps(redacted_config_dict(c.config)))
    monkeypatch.setitem(problems._REGISTRY, "fake", lambda _: HiddenProblem())
    assert Playback().load(c.output, "plan", plan["plan_id"])["frames"] == 3


def test_make_cwm_env_starts_from_the_carried_state(hidden_rig, monkeypatch):
    import importlib
    import sys

    c = hidden_rig(lifecycle=Lifecycle.SINGLE_INSTANCE, target=3)  # n = 0..4
    hidden_model(c.workdir)
    assert c.tool("UpdateCodeWorldModel", {})["accepted"]
    exploration(c, (1,))
    assert c.tool("RunController", {})["real_actions"] == 1  # n = 5, behind screen 2
    for name in [m for m in sys.modules if m.split(".")[0] in ("framework", "world_model")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(c.workdir))
    data_api = importlib.import_module("framework.data_api")
    monkeypatch.setattr(data_api, "_query", lambda op, **args: c.data({"op": op, **args}))
    cwm_env = importlib.import_module("framework.cwm_env")
    assert cwm_env.make_cwm_env().state.n == 5  # rebuilt from the episode; screen 2 reads as 4
