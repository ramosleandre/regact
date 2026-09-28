import concurrent.futures
import json
import time

import pytest
from fastapi.testclient import TestClient

from regact.config.schema import (
    AgentConfig,
    AgentName,
    Lifecycle,
    ProblemConfig,
    ProtocolConfig,
    RunConfig,
    redacted_config_dict,
)
from regact.env.renderer import RawRenderer
from regact.orchestration.task import _build_server
from regact.problems.base import BaseProblem, register_problem
from regact.protocols.cwm.store import ExperienceStore
from regact.protocols.cwm.viewer import Playback
from regact.protocols.registry import build_protocol
from regact.security.runtime import SandboxRuntime, detect
from regact.workspace.bootstrap import Workspace


class Native:
    def reset(self, seed=None):
        self.n = 0
        return [0] * 30, {"available_actions": [1]}

    def step(self, a):
        from regact.envclient.errors import InvalidActionError

        if a != 1:
            raise InvalidActionError("only action 1")
        self.n += 1
        return [self.n] * 30, float(self.n >= 6), self.n >= 6, False, {"available_actions": [1]}

    def close(self):
        pass


class Problem(BaseProblem):
    name = "fake"

    def make_env(self, task):
        return Native()

    def get_task_names(self):
        return ["counter"]

    def obs_renderer(self, *a, **k):
        return RawRenderer()

    def build_prompt(self, *a, **k):
        return "Advance to 6."

    def config_kwargs(self):
        return {}

    def compute_episode_metrics(self, obs, steps):
        return {"success": obs.is_done, "steps": steps}

    def aggregate_episode_metrics(self, e):
        return {"success_rate": sum(x["success"] for x in e) / len(e)}


def model(root, bad=False):
    d = root / "world_model"
    d.mkdir(exist_ok=True, parents=True)
    (d / "model_state.py").write_text(
        "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass S"
        "tate:\n n:int\n"
    )
    (d / "model_parser.py").write_text(
        'from model_state import State\ndef parse(o): return State(o["frame'
        '"][0])\n'
    )
    (d / "model_render.py").write_text(
        'def render(s): return {"frame":[s.n]*30, "reward":float(s.n>=6),'
        '"is_done":s.n>=6,"available_actions":[1],"i'
        'nfo":{"available_actions":[1],"milestones":[]}}\n'
    )
    (d / "model_transition.py").write_text(
        "from model_state import State\ndef step(s,a): return State(s.n+"
        + (" (2 if s.n>=2 else 1)" if bad else "1")
        + ")\n"
    )


pytestmark = pytest.mark.integration


@pytest.fixture
def rig(tmp_path):
    if detect() is SandboxRuntime.NONE:
        pytest.skip("CWM requires an OS sandbox")
    cfg = RunConfig(
        AgentConfig(AgentName.SCRIPTED),
        ProblemConfig("fake", seed=0),
        protocol=ProtocolConfig("cwm", {"n_unique_observations_in_initial_collection": 3}),
    )
    (tmp_path / "config.json").write_text(json.dumps(redacted_config_dict(cfg)))
    register_problem("fake", lambda _: Problem())
    protocol = build_protocol(cfg)
    server = _build_server(
        cfg,
        Problem(),
        "counter",
        protocol=protocol,
        workdir=str(tmp_path / "workdir"),
        output_dir=str(tmp_path),
    )
    Workspace(str(tmp_path / "workdir")).bootstrap(
        [],
        templates=protocol.templates,
        expose_environment=False,
        problem_name="fake",
        task_name="counter",
        env_base_url="http://unused",
        game_id="counter",
        lifecycle=Lifecycle.MULTI_INSTANCE,
    )
    c = protocol.coordinator
    try:
        yield c, server
    finally:
        c.close("test_finished")


def collect(c):
    c.collect_initial()


def accept(c, bad=False):
    collect(c)
    model(c.workdir, bad)
    result = c.tool("UpdateCodeWorldModel", {})
    assert result.get("accepted"), result


def exploration(c, actions=(1, 1, 1, 1), extra=""):
    (c.workdir / "exploration.py").write_text(
        (
            '"""Reach a new position."""\nfrom framework.action_list_controller import Exp'
            "lorationControllerFromListActions\n"
        )
        + extra
        + ("\ndef get_controller(): return ExplorationControllerFromListActions(")
        + repr(list(actions))
        + ")\n"
    )


def test_prefill_and_direct_environment_denial(rig):
    c, server = rig
    for filename in ("make_env.py", "cwm_client.py"):
        assert not (c.workdir / "framework" / filename).exists()
    with TestClient(server.app) as http:
        for _ in range(2):
            assert c.phase == "CWM Modeling"
            for op, body in (("step", {"action": 1}), ("reset", {})):
                reply = http.post(f"/env/counter/{op}", json=body)
                assert reply.status_code == 409
                assert reply.json()["detail"]["code"] == "cwm_direct_environment_disabled"
            collect(c)
    assert c.store.summary()["n_unique_observations"] == 3
    assert c.store.summary()["n_total_transitions"] == 2
    assert c.store.summary()["n_started_episodes"] == 1
    assert c.initial_collection["target_reached"]


def test_concurrent_prefill_executes_once(rig):
    c, _ = rig
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: c.collect_initial(), range(4)))
    assert c.store.summary()["n_total_transitions"] == 2


def test_prefill_action_cap_allows_modelling_with_less_data(rig):
    c, _ = rig
    c.options.max_actions_per_initial_collection = 1
    collect(c)
    assert c.phase == "CWM Modeling" and c.terminal is None
    assert c.initial_collection["stop_reason"] == "action_cap"
    assert not c.initial_collection["target_reached"]
    assert c.store.summary()["n_unique_observations"] == 2


def test_prefill_respects_global_budget(rig):
    c, _ = rig
    c.config.limits.max_actions_per_task = 1
    collect(c)
    assert c.terminal == "real_action_limit"
    assert c.env.live.action_count == 1


def test_prefill_interrupt_before_first_action(rig):
    from regact.orchestration.signals import StopSignal

    c, _ = rig
    stop = StopSignal()
    stop.set()
    c.collect_initial(stop)
    assert c.terminal == "interrupted"
    assert c.store.summary()["n_total_transitions"] == 0


def test_plan_real_exploration_and_replay(rig):
    c, _ = rig
    accept(c)
    (c.workdir / "goal.py").write_text('"""Reach 4."""\ndef achieved(s): return s.n==4\n')
    before = c.store.summary()
    result = c.tool("PlanInCWM", {"request_id": "plan"})
    assert result["achieved"], result
    assert c.store.summary() == before
    assert c.tool("PlanInCWM", {"request_id": "plan"}) == {**result, "replayed": True}
    playback = Playback()
    assert playback.load(c.output, "plan", result["plan_id"])["frames"] == 5
    assert playback.frame(c.output, "plan", result["plan_id"], 4)["obs"]["frame"][0] == 4
    exploration(c)
    result = c.tool("SubmitExplorationController", {})
    assert result["real_actions"] == result["simulation_actions"] == 4, result
    assert result["actual_novel_observations"] == 2
    assert result["objective_reached"] is None
    assert result["stop_reason"] == "plan_exhausted"
    assert playback.load(c.output, "episode", result["episode_id"])["frames"] == 5
    assert playback.frame(c.output, "episode", result["episode_id"], 4)["obs"]["frame"][0] == 4
    assert c.phase == "Active Exploration"


def test_first_mismatch_recorded_then_requires_repair(rig):
    c, _ = rig
    accept(c, bad=True)
    exploration(c)
    result = c.tool("SubmitExplorationController", {})
    assert result["stop_reason"] == "prediction_mismatch", result
    assert result["real_actions"] == 3
    assert c.phase == "CWM Modeling"
    diagnostic = c.store.get_diagnostic(result["diagnostic_id"])
    assert diagnostic["predicted"]["frame"][0] == 4 and diagnostic["observed"]["frame"][0] == 3
    assert "error" in c.tool("PlanInCWM", {})
    model(c.workdir)
    repaired = c.tool("UpdateCodeWorldModel", {})
    assert repaired["accepted"] and repaired["observations_checked"] == 4
    assert repaired["transitions_checked"] == 3


def test_no_novelty_never_resets_or_steps_real_env(rig):
    c, _ = rig
    accept(c)
    exploration(c, (1, 1))
    before = c.store.summary()
    result = c.tool("SubmitExplorationController", {})
    assert result["stop_reason"] == "no_predicted_novelty"
    assert c.store.summary() == before


def test_invalid_controller_action_gets_worst_metrics(rig):
    c, _ = rig
    accept(c)
    exploration(c, (1, 99, 1))
    result = c.tool("SubmitExplorationController", {})
    assert result["stop_reason"] == "controller_error", result
    assert result["real_actions"] == 1
    assert result["metrics"]["success"] is False
    assert result["error_type"] == "InvalidActionError"
    assert c.phase == "Active Exploration"


def test_controller_cannot_monkeypatch_model(rig):
    c, _ = rig
    accept(c)
    exploration(
        c,
        extra=(
            "import model_transition\nmodel_transition.step=lambda s,a: (_ for "
            '_ in ()).throw(RuntimeError("tampered"))\n'
        ),
    )
    result = c.tool("SubmitExplorationController", {})
    assert result["real_actions"] == 4, result


def test_accepted_snapshot_ignores_workdir_edits(rig):
    c, _ = rig
    accept(c)
    (c.workdir / "world_model/model_transition.py").write_text('raise RuntimeError("mutable")')
    exploration(c)
    result = c.tool("SubmitExplorationController", {})
    assert result["real_actions"] == 4, result


def test_callback_timeout_is_recorded_and_not_accepted(rig):
    c, _ = rig
    collect(c)
    model(c.workdir)
    c.options.execution.max_seconds_per_call = 0.15
    (c.workdir / "world_model/model_transition.py").write_text(
        "def step(s,a):\n while True: pass\n"
    )
    started = time.monotonic()
    result = c.tool("UpdateCodeWorldModel", {})
    assert time.monotonic() - started < 4
    assert result["error_type"] == "code_timeout", result
    assert result["observations_checked"] == 3 and not result["complete"]
    assert c.phase == "CWM Modeling" and c.accepted is None
    assert c.store.db.execute("SELECT status FROM records").fetchone()[0] == "rejected"


def test_bad_reconstruction_and_compression_are_rejected(rig):
    c, _ = rig
    collect(c)
    model(c.workdir)
    p = c.workdir / "world_model/model_render.py"
    p.write_text(p.read_text().replace("[s.n]*30", "[0]*30"))
    result = c.tool("UpdateCodeWorldModel", {})
    assert not result["accepted"] and result["complete"]
    assert result["failures"]["reconstruction_mismatch"] == 2
    assert result["counterexamples"][0]["diagnostic_id"]
    model(c.workdir)
    c.options.threshold_max_state_obs_size_ratio = 0.01
    assert "compression_ratio" in c.tool("UpdateCodeWorldModel", {})["failures"]


def test_independent_action_budgets_and_global_real_budget(rig):
    c, _ = rig
    accept(c)
    exploration(c)
    c.options.max_actions_per_exploration = 4
    result = c.tool("SubmitExplorationController", {})
    assert result["simulation_actions"] == result["real_actions"] == 4
    # New proposal predicts position 5, but only two real actions remain.
    exploration(c, (1, 1, 1, 1, 1))
    c.options.max_actions_per_exploration = 5
    c.config.limits.max_actions_per_task = 8
    result = c.tool("SubmitExplorationController", {})
    assert result["simulation_actions"] == 5 and result["real_actions"] == 2, result
    assert c.terminal == "real_action_limit"


def test_observation_determinism_preserves_both_witnesses(tmp_path):
    store = ExperienceStore(tmp_path / "experience.sqlite3")
    try:
        ep, _ = store.start_episode({"x": 0}, "test", {})
        a = store.record_step(ep, {"x": 0}, 1, {"x": 1})
        b = store.record_step(ep, {"x": 0}, 1, {"x": 2})
        assert b["conflicting_witnesses"][0]["event_id"] == a["event_id"]
        assert store.summary()["n_unique_transitions"] == 2
    finally:
        store.close()


def test_worker_cannot_read_database_or_network(rig):
    c, _ = rig
    collect(c)
    model(c.workdir)
    parser = c.workdir / "world_model/model_parser.py"
    parser.write_text(
        "from pathlib import Path\nimport socket\n"
        + parser.read_text()
        + f"\nassert not Path({str(c.root / 'experience.sqlite3')!r}).exists()\n"
        + 's=socket.socket()\ns.settimeout(.1)\ntry:\n s.connect(("127.0.0.1",8030))\n'
        + 'except OSError:\n pass\nelse:\n raise RuntimeError("network reachable")\n'
    )
    result = c.tool("UpdateCodeWorldModel", {})
    assert result["accepted"], result


def test_partial_plan_and_utility_not_goal(rig):
    c, _ = rig
    accept(c)
    c.options.planner.max_depth_per_planner_call = 3
    (c.workdir / "goal.py").write_text(
        '"""Reach 100."""\ndef achieved(s): return s.n==100\ndef utility(s):'
        " return min(s.n/100,1)\n"
    )
    result = c.tool("PlanInCWM", {})
    assert result["candidate_found"] and not result["achieved"]
    assert not result["optimality_proven"]
    assert result["search_stop_reason"] == "max_depth_per_planner_call"


def test_full_solve_closes_run_without_submission(rig):
    c, _ = rig
    accept(c)
    exploration(c, (1,) * 6)
    result = c.tool("SubmitExplorationController", {})
    assert result["exit_reason"] == "solved" and result["metrics"]["success"], result
    assert not (c.workdir / "submissions").exists()


def test_submitted_symlinks_and_plan_output_symlinks_are_rejected(rig, tmp_path):
    c, _ = rig
    collect(c)
    model(c.workdir)
    target = tmp_path / "outside.py"
    target.write_text("SECRET")
    source = c.workdir / "world_model/model_parser.py"
    source.unlink()
    source.symlink_to(target)
    result = c.tool("UpdateCodeWorldModel", {})
    assert "error" in result and not c.accepted
    source.unlink()
    model(c.workdir)
    assert c.tool("UpdateCodeWorldModel", {})["accepted"]
    (c.workdir / "goal.py").write_text('"""Reach 4."""\ndef achieved(s): return s.n==4\n')
    outside = tmp_path / "outside"
    outside.mkdir()
    (c.workdir / "plans").symlink_to(outside, target_is_directory=True)
    result = c.tool("PlanInCWM", {})
    assert "error" in result
    assert not list(outside.iterdir())


def test_storage_failure_blocks_retry_of_uncertain_action(rig, monkeypatch):
    c, _ = rig

    def failed_record(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(c.store, "record_step", failed_record)
    collect(c)
    assert c.terminal == "environment_or_storage_failure"
    assert c.env.live.action_count == 1
    collect(c)
    assert c.env.live.action_count == 1


async def test_shutdown_interrupts_active_worker(rig):
    import asyncio

    from regact.protocols.cwm.session import CwmTool
    from regact.tools.base import ToolContext

    c, _ = rig
    collect(c)
    model(c.workdir)
    (c.workdir / "world_model/model_transition.py").write_text(
        "def step(s,a):\n while True: pass\n"
    )
    job = asyncio.create_task(
        CwmTool("UpdateCodeWorldModel", c).call({}, ToolContext(cwd=str(c.workdir)))
    )
    await asyncio.sleep(0.2)
    start = time.monotonic()
    await c.shutdown("test_abort")
    output = await job
    assert time.monotonic() - start < 2
    assert c.closed and output.is_error


def test_planner_time_budget_returns_best_partial_candidate(rig):
    c, _ = rig
    collect(c)
    model(c.workdir)
    p = c.workdir / "world_model/model_render.py"
    p.write_text(p.read_text().replace("s.n>=6", "s.n>=600"))
    assert c.tool("UpdateCodeWorldModel", {})["accepted"]
    (c.workdir / "goal.py").write_text(
        '"""Reach 100."""\nimport time\ndef achieved(s):\n time.sleep(.1)\n re'
        "turn s.n==100\ndef utility(s): return min(s.n/100,1)\n"
    )
    c.options.planner.max_seconds_per_planner_call = 0.75
    result = c.tool("PlanInCWM", {})
    assert result["candidate_found"] and not result["achieved"], result
    assert result["search_stop_reason"] == "max_seconds_per_planner_call"


def test_simulation_playback_and_frozen_source(rig):
    from regact.protocols.cwm.viewer import source

    c, _ = rig
    accept(c)
    exploration(c)
    result = c.tool("SubmitExplorationController", {})
    playback = Playback()
    assert playback.load(c.output, "simulation", result["exploration_id"])["frames"] == 5
    listing = source(c.output, result["bundle"])
    assert "exploration.py" in listing["files"]
    assert "Reach a new position" in source(c.output, result["bundle"], "exploration.py")["source"]
    with pytest.raises(ValueError):
        source(c.output, result["bundle"], "../config.json")


@pytest.mark.parametrize(
    "options",
    [
        {"n_unique_observations_in_initial_collection": "15"},
        {"max_actions_per_exploration": False},
        {"planner": {"algorithm": "astar"}},
        {"execution": {"max_seconds_per_call": 0}},
    ],
)
def test_invalid_protocol_options_rejected(options):
    from regact.protocols.cwm.config import CwmConfig

    with pytest.raises(ValueError):
        CwmConfig.from_mapping(options)


async def test_cwm_runner_dry_run_has_only_its_own_tools(tmp_path):
    from regact.agent.scripted_agent import ScriptedAgent
    from regact.orchestration.task import run_task

    if detect() is SandboxRuntime.NONE:
        pytest.skip("OS sandbox required")
    cfg = RunConfig(
        AgentConfig(AgentName.SCRIPTED),
        ProblemConfig("fake", seed=0),
        protocol=ProtocolConfig("cwm"),
        dry_run=True,
    )
    agent = ScriptedAgent()
    assert (
        await run_task(cfg, Problem(), "counter", output_dir=str(tmp_path), agent=agent)
        == "dry_run"
    )
    transcript = (tmp_path / "logs/transcript.jsonl").read_text()
    assert "SubmitExplorationController" in transcript and "SubmitSolution" not in transcript
    assert "ExitTask" not in transcript
    assert json.loads((tmp_path / "cwm/status.json").read_text())["exit_reason"] == "dry_run"


def test_prefill_time_cap(rig, monkeypatch):
    from regact.protocols.cwm import session

    c, _ = rig
    clock = iter([0, 0, 2, 2])
    monkeypatch.setattr(session.time, "monotonic", lambda: next(clock))
    c.options.max_seconds_per_initial_collection = 1
    collect(c)
    assert c.initial_collection["stop_reason"] == "time_cap"
    assert c.store.summary()["n_total_transitions"] == 0
    assert c.phase == "CWM Modeling"


def test_prefill_empty_action_space(rig, monkeypatch):
    c, _ = rig
    monkeypatch.setattr(c.problem, "enumerate_actions", lambda obs: iter(()))
    collect(c)
    assert c.initial_collection["stop_reason"] == "no_available_actions"
    assert c.phase == "CWM Modeling"


def test_prefill_episode_limit_reset(rig):
    c, _ = rig
    c.config.limits.max_actions_per_episode = 1
    c.options.max_actions_per_initial_collection = 4
    collect(c)
    assert c.initial_collection["stop_reason"] == "action_cap"
    assert c.store.summary()["n_total_transitions"] == 4
    assert c.store.summary()["n_unique_transitions"] == 1
    assert c.store.summary()["n_started_episodes"] == 4


def test_prediction_replay_with_relative_output_root(rig, monkeypatch):
    from pathlib import Path

    c, _ = rig
    accept(c)
    (c.workdir / "goal.py").write_text('"""Reach 4."""\ndef achieved(s): return s.n==4\n')
    result = c.tool("PlanInCWM", {})
    monkeypatch.chdir(c.output.parent)
    relative = Path(c.output.name)
    playback = Playback()
    assert playback.load(relative, "plan", result["plan_id"])["frames"] == 5
    assert playback.frame(relative, "plan", result["plan_id"], 4)["obs"]["frame"][0] == 4


def test_noop_after_reset_does_not_create_artificial_dataset_novelty(rig, monkeypatch):
    c, _ = rig
    monkeypatch.setattr(
        c.env.live._native,
        "step",
        lambda action: ([0] * 30, 0, False, False, {"available_actions": [1]}),
    )
    c._step(1)
    assert c.current_id == c.initial_id
    assert c.store.summary()["n_unique_observations"] == 1
    assert c.store.summary()["n_total_observations"] == 2
