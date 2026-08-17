"""Integration: run_task ties the whole stack together for one task.

No LLM, no real game: a ScriptedAgent that 'writes' a working controller on
start, against a FakeNativeEnv-backed test problem, driven through the real
session builder (in-process transport). Asserts the canonical artifacts.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from regact.agent.events import TextDelta, ToolCall, TurnComplete
from regact.agent.scripted_agent import ScriptedAgent
from regact.config.schema import (
    AgentConfig,
    AgentName,
    ControllerConfig,
    Lifecycle,
    LimitsConfig,
    ProblemConfig,
    RunConfig,
)
from regact.env.renderer import RawRenderer
from regact.envclient.obs import Obs
from regact.orchestration.task import run_task
from regact.problems.base import BaseProblem
from regact.security.runtime import SandboxRuntime
from regact.testing.fakes import FakeNativeEnv

pytestmark = pytest.mark.integration

# The controller the agent "writes": always step forward -> reaches the goal.
_FORWARD = """\
class Controller:
    def act(self, obs):
        return 1


def get_controller():
    return Controller()
"""


class _FakeProblem(BaseProblem):
    """A minimal problem backed by FakeNativeEnv (no game lib needed)."""

    name = "fake"

    def make_env(self, task_name: str) -> Any:
        return FakeNativeEnv(goal=3)

    def get_task_names(self) -> list[str]:
        return ["corridor"]

    def obs_renderer(self, task_name: str, *, mode: Any) -> RawRenderer:
        return RawRenderer()

    def compute_episode_metrics(self, final_obs: Obs, *, steps: int) -> dict[str, Any]:
        return {"success": final_obs.is_done, "steps": steps}

    def aggregate_episode_metrics(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(episodes) or 1
        return {
            "n_episodes": len(episodes),
            "success_rate": sum(bool(e.get("success")) for e in episodes) / n,
        }

    def build_prompt(self, task_name: str, *, info_mode: Any) -> str:
        return f"# Game: fake ({task_name})\nReach the goal."

    def config_kwargs(self) -> dict[str, Any]:
        return {}


class _WritingAgent(ScriptedAgent):
    """A scripted agent that drops a working controller into the workdir on start."""

    async def start(self, *, cwd: str, **kwargs: Any) -> None:
        await super().start(cwd=cwd, **kwargs)
        Path(cwd, "solution.py").write_text(_FORWARD)

    def session_id(self) -> str | None:
        return "native-123"


def _config() -> RunConfig:
    return RunConfig(
        agent=AgentConfig(name=AgentName.SCRIPTED),
        problem=ProblemConfig(name="fake", kwargs={"env_id": "fake-v0"}),
        controller=ControllerConfig(n_episodes=2, max_moves=10),
        limits=LimitsConfig(max_turns=10),
    )


async def test_run_task_end_to_end(tmp_path: Path) -> None:
    agent = _WritingAgent(
        [
            [TextDelta("Submitting."), ToolCall("c1", "SubmitSolution", {}), TurnComplete()],
            [ToolCall("c2", "ExitTask", {}), TurnComplete()],
        ]
    )
    reason = await run_task(
        _config(), _FakeProblem(), "corridor", output_dir=str(tmp_path), agent=agent
    )

    assert reason == "agent_exit"
    assert agent.started and agent.closed

    logs = tmp_path / "logs"
    workdir = tmp_path / "workdir"
    # Workdir bootstrapped: agnostic base + the always-on controller's templates.
    assert (workdir / "framework" / "make_env.py").exists()
    assert (workdir / "code_library" / "base_controller.py").exists()
    # Canonical artifacts written.
    assert (logs / "transcript.jsonl").exists()
    assert (logs / "experiment_state.json").exists()
    types = [
        json.loads(line)["type"] for line in (logs / "transcript.jsonl").read_text().splitlines()
    ]
    assert "ToolResult" in types
    # The controller was scored (submission + final), success_rate 1.0.
    submitted = json.loads((workdir / "submissions" / "0" / "results.json").read_text())
    assert submitted["aggregate"]["success_rate"] == 1.0
    final = json.loads((workdir / "submissions" / "final" / "results.json").read_text())
    assert final["aggregate"]["success_rate"] == 1.0
    # The persisted state reflects the run's config and telemetry, not placeholders.
    state = json.loads((logs / "experiment_state.json").read_text())
    assert state["n_eval_episodes"] == 2
    assert state["n_videos"] == 2  # record_video defaults to True
    assert state["problem_kwargs"] == {"env_id": "fake-v0"}
    assert state["agent_session_id"] == "native-123"
    assert state["env_moves"] > 0  # eval episodes stepped the env
    # Framework actions and finalization land in the operational log.
    events = [json.loads(line) for line in (logs / "events.jsonl").read_text().splitlines()]
    executed = [e["detail"]["tool"] for e in events if e["event"] == "tool_executed"]
    assert "SubmitSolution" in executed and "ExitTask" in executed
    assert any(e["event"] == "hook_executed" for e in events)


def test_network_isolation_knob() -> None:
    """`sandbox_opts.network_isolation=false` keeps the fs sandbox but drops the net namespace;
    it is meaningless without sandbox. (The interim for an offline node while the loopback bridge
    is debugged.)"""
    from regact.orchestration.task import _network_isolation

    def cfg(**over: Any) -> RunConfig:
        base: dict[str, Any] = {
            "agent": AgentConfig(name=AgentName.ALAN),
            "problem": ProblemConfig(name="minigrid"),
        }
        return RunConfig(**{**base, **over})

    assert _network_isolation(cfg(sandbox=True)) is True  # default: isolate under sandbox
    assert _network_isolation(cfg(sandbox=False)) is False  # no sandbox -> nothing to isolate
    assert (
        _network_isolation(cfg(sandbox=True, sandbox_opts={"network_isolation": False})) is False
    )
    assert (
        _network_isolation(cfg(sandbox=True, sandbox_opts={"network_isolation": True})) is True
    )


async def test_run_task_sandbox_true_fails_when_no_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from regact.orchestration import task as task_module

    monkeypatch.setattr(task_module, "resolve", lambda _: SandboxRuntime.NONE)
    config = _config()
    config.sandbox = True
    with pytest.raises(RuntimeError, match="sandbox"):
        await run_task(config, _FakeProblem(), "corridor", output_dir=str(tmp_path))
    assert not (tmp_path / "logs").exists()  # refused before writing any artifact


async def test_run_task_refuses_single_instance_with_on_env_eval(tmp_path: Path) -> None:
    # The always-on controller scores on the env, so single-instance is refused (its eval
    # would share the exploration env).
    config = _config()
    config.problem.lifecycle = Lifecycle.SINGLE_INSTANCE
    with pytest.raises(RuntimeError, match="single-instance"):
        await run_task(config, _FakeProblem(), "corridor", output_dir=str(tmp_path))
    assert not (tmp_path / "logs").exists()  # refused before writing any artifact


async def test_run_task_builds_agent_from_config_when_none(tmp_path: Path) -> None:
    """With no injected agent, build_agent(scripted) runs (default turns -> exits on limit)."""
    config = _config()
    config.limits = LimitsConfig(max_turns=1)
    reason = await run_task(config, _FakeProblem(), "corridor", output_dir=str(tmp_path))
    assert reason == "loop_limit"  # the default scripted agent never submits/exits


def test_workdir_on_pythonpath_lets_subdir_scripts_import_framework(tmp_path: Path) -> None:
    """The scaffolding contract: ``import framework`` must resolve even when the agent
    runs a script from a SUBDIR (``python code_library/probe.py``). Python only puts the
    script's own dir on sys.path, so this works only with the workdir root on PYTHONPATH -
    which run_task sets (see task.py agent_env). Guards that papercut from regressing."""
    import os
    import subprocess
    import sys

    workdir = tmp_path / "workdir"
    (workdir / "framework").mkdir(parents=True)
    (workdir / "framework" / "__init__.py").write_text("MARKER = 'ok'\n")
    (workdir / "code_library").mkdir()
    probe = workdir / "code_library" / "probe.py"
    probe.write_text("import framework; print(framework.MARKER)\n")
    bare = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    # Without the workdir on PYTHONPATH: the documented import fails (the papercut).
    missing = subprocess.run(
        [sys.executable, "code_library/probe.py"],
        cwd=workdir,
        env=bare,
        capture_output=True,
        text=True,
    )
    assert missing.returncode != 0 and "ModuleNotFoundError" in missing.stderr

    # With it (what run_task now sets): the import resolves from the subdir.
    ok = subprocess.run(
        [sys.executable, "code_library/probe.py"],
        cwd=workdir,
        env={**bare, "PYTHONPATH": str(workdir)},
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0 and ok.stdout.strip() == "ok"
