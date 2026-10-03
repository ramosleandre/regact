"""Lifecycle, reset, fresh-controller and baseline invariants with actual isolated code."""

import concurrent.futures
import time
from types import SimpleNamespace

import pytest

from regact.config.schema import (
    AgentConfig,
    AgentName,
    Lifecycle,
    LimitsConfig,
    ProblemConfig,
    ProtocolConfig,
    RunConfig,
)
from regact.env.lifecycle import MultiInstancePolicy, SingleInstancePolicy
from regact.env.session import EnvSession
from regact.features.base import FeatureContext
from regact.orchestration.task import _bootstrap_workdir
from regact.protocols.cwm.feedback import present
from regact.protocols.cwm.viewer import Playback
from regact.protocols.registry import build_protocol
from test_cwm_protocol import Native, Problem, exploration, model

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("protocol_name", ["vanilla", "cwm"])
def test_image_previews_need_a_vision_agent(protocol_name):
    options = {"n_tmp_images_saved_per_exploration": 2}
    blind = RunConfig(
        AgentConfig(AgentName.SCRIPTED),
        ProblemConfig("fake"),
        protocol=ProtocolConfig(protocol_name, options),
    )
    with pytest.raises(ValueError, match=r"agent\.vision"):
        build_protocol(blind).validate()
    sighted = RunConfig(
        AgentConfig(AgentName.SCRIPTED, vision=True),
        ProblemConfig("fake"),
        protocol=ProtocolConfig(protocol_name, options),
    )
    build_protocol(sighted).validate()


@pytest.mark.parametrize("protocol_name", ["vanilla", "cwm"])
@pytest.mark.parametrize("dialect", ["client_cli", "hermes_xml"])
def test_managed_workspace_and_prompt_use_renamed_files(tmp_path, protocol_name, dialect):
    config = RunConfig(
        AgentConfig(AgentName.SCRIPTED),
        ProblemConfig("fake"),
        protocol=ProtocolConfig(protocol_name),
    )
    protocol = build_protocol(config)
    problem = Problem()
    _bootstrap_workdir(
        config,
        problem,
        "counter",
        workdir=str(tmp_path),
        conn=SimpleNamespace(base_url="http://unused"),
        protocol=protocol,
    )
    prompt = protocol.build_system_prompt(
        problem,
        "counter",
        tool_protocol=dialect,
        tool_names=["RunController", "ResetEnvironment"],
        verbalize_variant="off",
    )
    for name in ("controller.py", "framework/commands.py"):
        assert (tmp_path / name).is_file()
        assert name in prompt
    for name in ("exploration.py", "framework/control.py"):
        assert not (tmp_path / name).exists()
        assert name not in prompt
    assert "python framework/commands.py RunController" in prompt


@pytest.fixture
def make_rig(tmp_path):
    coordinators = []

    def make(
        protocol="vanilla",
        lifecycle=Lifecycle.SINGLE_INSTANCE,
        problem=None,
        target=3,
        limits=None,
        **options,
    ):
        root = tmp_path / str(len(coordinators))
        work = root / "workdir"
        work.mkdir(parents=True)
        problem = problem or Problem()
        cfg = RunConfig(
            AgentConfig(AgentName.SCRIPTED),
            ProblemConfig("fake", seed=0, lifecycle=lifecycle),
            protocol=ProtocolConfig(
                protocol,
                {
                    "n_unique_observations_in_initial_collection": target,
                    "n_tmp_images_saved_per_exploration": 0,
                    **options,
                },
            ),
            limits=limits or LimitsConfig(),
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
        c = definition.coordinator_type(
            cfg, definition.options, env, problem, "counter", root, work
        )
        definition.coordinator = c
        for f in definition.templates(
            FeatureContext(problem_name="fake", task_name="counter", workdir=str(work))
        ):
            p = work / f.relpath
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f.content)
        coordinators.append(c)
        c.collect_initial()
        if protocol == "cwm":
            model(work)
            assert c.tool("UpdateCodeWorldModel", {}).get("accepted")
        return c

    yield make
    for c in coordinators:
        c.close("test_finished")


@pytest.mark.parametrize("protocol", ["vanilla", "cwm"])
def test_continuation_fresh_controller_and_exact_call_evidence(make_rig, protocol):
    c = make_rig(protocol)
    assert c.env.live.last_obs.frame[0] == 2
    # Each call's private iterator starts at zero, but the real env keeps advancing.
    exploration(c, (1,))
    first = c.tool("RunController", {})
    second = c.tool("RunController", {})
    assert first["real_actions"] == second["real_actions"] == 1, (first, second)
    assert first["episode_id"] == second["episode_id"] == 1
    assert c.env.live.last_obs.frame[0] == 4
    assert first["observation_sequence"] == [3, 4]
    assert second["observation_sequence"] == [4, 5]
    assert first["transition_ids"] == [3]
    assert second["transition_ids"] == [4]
    assert c.data({"op": "summary"})["current_observation_id"] == 5
    assert c.store.summary()["n_started_episodes"] == 1


@pytest.mark.parametrize("protocol", ["vanilla", "cwm"])
def test_multi_starts_fresh(make_rig, protocol):
    c = make_rig(protocol, Lifecycle.MULTI_INSTANCE)
    exploration(c, (1,))
    first = c.tool("RunController", {})
    second = c.tool("RunController", {})
    assert first["episode_id"] != second["episode_id"]
    assert c.env.live.last_obs.frame[0] == 1
    assert first["observation_sequence"] == second["observation_sequence"] == [1, 2]


class FailureNative(Native):
    def step(self, a):
        frame, reward, done, trunc, info = super().step(a)
        return frame, 0.0, self.n >= 3, trunc, info


class FailureProblem(Problem):
    def make_env(self, task):
        return FailureNative()

    def compute_episode_metrics(self, obs, steps):
        return {"success": False, "steps": steps}


def test_done_stops_prefill_and_allows_reset(make_rig):
    c = make_rig(problem=FailureProblem(), target=20)
    assert c.initial_collection["stop_reason"] == "environment_done"
    assert c.terminal is None
    exploration(c, (1,))
    result = c.tool("RunController", {})
    assert result["stop_reason"] == "environment_done" and result["real_actions"] == 0
    assert c.terminal is None
    reset = c.tool("ResetEnvironment", {"request_id": "reset-1"})
    assert reset["real_actions"] == 1 and c.env.live.last_obs.frame[0] == 0
    assert c.store.summary()["n_started_episodes"] == 2
    again = c.tool("ResetEnvironment", {"request_id": "reset-1"})
    assert again["replayed"] and c.reset_actions == 1
    assert c.tool("RunController", {})["real_actions"] == 1


def test_info_trace_counts_real_steps_and_explicit_resets(make_rig):
    from regact.protocols.cwm.viewer import info_trace

    c = make_rig(target=3)  # initial collection: 2 real steps
    exploration(c, (1,))
    c.tool("RunController", {})
    c.tool("ResetEnvironment", {})
    c.tool("RunController", {})
    assert [actions for actions, _ in info_trace(c.output)] == [1, 2, 3, 5]


def test_experiment_deadline_bounds_the_initial_collection(make_rig):
    c = make_rig(target=20, limits=LimitsConfig(experiment_deadline_unix=int(time.time()) - 1))
    assert c.terminal == "walltime_limit"
    assert c.initial_collection["stop_reason"] == "walltime_limit"
    assert c.store.summary()["n_total_transitions"] == 0


def test_reset_budget_is_task_wide(make_rig):
    c = make_rig(target=1)
    c.config.limits.max_actions_per_task = 2
    c.tool("ResetEnvironment", {})
    c.tool("ResetEnvironment", {})
    assert c.terminal == "real_action_limit"
    assert c.reset_actions == 2
    assert c.tool("ResetEnvironment", {}).get("error")
    assert c.reset_actions == 2


def test_concurrent_retry_runs_once(make_rig):
    c = make_rig()
    exploration(c, (1,))
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results = list(
            pool.map(lambda _: c.tool("RunController", {"request_id": "same"}), range(2))
        )
    assert sum(bool(x.get("replayed")) for x in results) == 1
    assert c.env.live.last_obs.frame[0] == 3


def test_cwm_mismatch_preserves_position_and_repairs(make_rig):
    c = make_rig("cwm")
    model(c.workdir, bad=True)
    assert c.tool("UpdateCodeWorldModel", {}).get("accepted")
    exploration(c, (1, 1))
    result = c.tool("RunController", {})
    assert result["stop_reason"] == "prediction_mismatch"
    assert result["real_actions"] == 1 and c.env.live.last_obs.frame[0] == 3
    assert c.phase == "CWM Modeling"
    model(c.workdir)
    assert c.tool("UpdateCodeWorldModel", {}).get("accepted")
    exploration(c, (1,))
    result = c.tool("RunController", {})
    assert result["real_actions"] == 1 and c.env.live.last_obs.frame[0] == 4
    assert c.episode == 1


def test_vanilla_has_no_cwm_requirements_or_files(make_rig):
    c = make_rig()
    assert not (c.workdir / "world_model").exists()
    assert not (c.workdir / "simulate.py").exists()
    assert set(c.commands) == {"RunController", "ResetEnvironment"}
    assert c.tool("UpdateCodeWorldModel", {})["error_type"] == "command_unavailable"
    # Repeated observation/action conflicts do not impose the CWM determinism assumption.
    assert c.uses_model is False


@pytest.mark.parametrize("protocol", ["vanilla", "cwm"])
def test_action_cap_survives_multiple_calls(make_rig, protocol):
    c = make_rig(protocol)
    c.config.limits.max_actions_per_episode = 3
    exploration(c, (1,))
    assert c.tool("RunController", {})["real_actions"] == 1
    result = c.tool("RunController", {})
    assert result["real_actions"] == 0 and result["stop_reason"] == "max_actions_per_episode"
    assert c.terminal is None
    c.tool("ResetEnvironment", {})
    assert c.env.live.episode_action_count == 0


@pytest.mark.parametrize("protocol", ["vanilla", "cwm"])
def test_controller_timeout_preserves_episode_and_next_call_gets_fresh_budget(make_rig, protocol):
    c = make_rig(protocol, execution={"max_seconds_per_RunController": 1})
    (c.workdir / "controller.py").write_text(
        '\"\"\"Try one slow action.\"\"\"\n'
        "import time\n"
        "class Controller:\n"
        " def act(self, obs):\n"
        "  time.sleep(3)\n"
        "  return 1\n"
        " def is_done(self, obs): return False\n"
        "def get_controller(): return Controller()\n"
    )
    episode = c.episode
    result = c.tool("RunController", {})
    assert result.get("real_actions") == 0 and result.get("stop_reason") == "controller_call_time_limit", result
    assert result["error_type"] == "operation_timeout"
    feedback = present("RunController", result, c.options, c.config.limits)
    assert feedback["budget"] == {
        "parameter": "protocol.execution.max_seconds_per_RunController",
        "value": 1,
        "unit": "seconds of submitted-code execution per RunController call",
    }
    assert feedback["error"]["budget"] == feedback["budget"]
    assert "reset" not in feedback["message"].lower()
    assert c.terminal is None and c.env.live.last_obs.frame[0] == 2
    # Time consumed by the failed call does not require a reset or reduce the next cap.
    exploration(c, (1,))
    result = c.tool("RunController", {})
    assert result["real_actions"] == 1 and result["stop_reason"] == "plan_exhausted", result
    assert c.episode == episode and c.env.live.last_obs.frame[0] == 3


@pytest.mark.parametrize("protocol", ["vanilla", "cwm"])
def test_controller_budget_charges_submitted_code_not_framework_time(
    make_rig, protocol, monkeypatch
):
    c = make_rig(protocol, execution={"max_seconds_per_RunController": 1})
    exploration(c, (1, 1, 1))
    step = c._step

    def slow_step(action):  # a slow environment or store, i.e. framework time
        time.sleep(0.5)
        return step(action)

    monkeypatch.setattr(c, "_step", slow_step)
    result = c.tool("RunController", {})
    assert result["real_actions"] == 3 and result["stop_reason"] == "plan_exhausted", result
    assert result["timings"]["submitted_code_seconds"] < 1 < result["elapsed_seconds"]


@pytest.mark.parametrize("protocol", ["vanilla", "cwm"])
def test_controller_budget_sums_submitted_code_across_callbacks(make_rig, protocol):
    c = make_rig(protocol, execution={"max_seconds_per_RunController": 1})
    (c.workdir / "controller.py").write_text(
        '\"\"\"Each action is slow, none alone reaches the budget.\"\"\"\n'
        "import time\n"
        "class Controller:\n"
        " def act(self, obs):\n"
        "  time.sleep(0.4)\n"
        "  return 1\n"
        " def is_done(self, obs): return False\n"
        "def get_controller(): return Controller()\n"
    )
    result = c.tool("RunController", {})
    assert result["stop_reason"] == "controller_call_time_limit", result
    assert result["real_actions"] == 2


def test_call_playback_retains_order_and_repeated_ids(make_rig):
    c = make_rig()
    exploration(c, (1, 1))
    first = c.tool("RunController", {})
    player = Playback()
    loaded = player.load(c.output, "controller", first["exploration_id"])
    assert loaded["frames"] == 3
    assert [
        player.frame(c.output, "controller", first["exploration_id"], i)["obs"]["frame"][0]
        for i in range(3)
    ] == [2, 3, 4]
    assert player.frame(c.output, "controller", first["exploration_id"], 1)["action"] == 1


def test_arc_explicit_resets_preserve_or_clear_level_progress():
    from regact.problems.arc_agi.problem import ArcAgiProblem

    problem = ArcAgiProblem(
        environments_dir="/home/tboulet/projects/regact/environnement", operation_mode="offline"
    )
    native = problem.make_env("ls20")
    game = native._env._game
    game.set_level(1)
    game._score = 1
    _, info = native.reset_explicit("level")
    assert info["levels_completed"] == 1
    assert game._current_level_index == 1
    _, info = native.reset_explicit("level")
    assert info["levels_completed"] == 1 and game._current_level_index == 1
    _, info = native.reset_explicit("environment")
    assert info["levels_completed"] == 0 and game._current_level_index == 0
    assert "handle_reset" not in game.__dict__


def test_arc_reset_handler_restored_on_failure():
    from regact.problems.arc_agi.problem import _ArcGymShim

    class Game:
        def handle_reset(self):
            pass

        def level_reset(self):
            pass

    class Wrapper:
        _game = Game()

        def reset(self):
            raise RuntimeError("oops")

    shim = _ArcGymShim(Wrapper())
    with pytest.raises(RuntimeError):
        shim.reset_explicit("level")
    assert "handle_reset" not in shim._env._game.__dict__


@pytest.mark.asyncio
async def test_vanilla_feedback_transport_and_limits(make_rig):
    import json

    from regact.protocols.managed.session import ManagedTool
    from regact.tools.base import ToolContext

    c = make_rig()
    c.options.max_actions_per_RunController = 1
    exploration(c, (1, 1))
    response = await ManagedTool("RunController", c).call({}, ToolContext(cwd=str(c.workdir)))
    value = json.loads(response.data)
    assert value["real_actions"] == 1
    assert value["budget"]["parameter"] == "protocol.max_actions_per_RunController"
    assert value["current_observation_id"] == 4
    reset = await ManagedTool("ResetEnvironment", c).call({}, ToolContext(cwd=str(c.workdir)))
    assert json.loads(reset.data)["current_observation_id"] == 1


@pytest.mark.parametrize("protocol", ["vanilla", "cwm"])
def test_controller_call_reports_where_its_time_went(make_rig, protocol):
    """A call that overran its deadline on a cluster could not say whether the native step,
    the durable store commit or the worker teardown held it; each call now records them."""
    c = make_rig(protocol)
    exploration(c, (1, 1))
    timings = c.tool("RunController", {})["timings"]
    for phase in ("native_step", "record_step"):
        assert timings[f"{phase}_seconds"] >= timings[f"{phase}_max_seconds"] >= 0
    assert timings["controller_worker_close_seconds"] >= 0
