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
    LimitsConfig,
    ProblemConfig,
    RunConfig,
)
from regact.env.renderer import RawRenderer
from regact.envclient.obs import Obs
from regact.orchestration.task import run_task
from regact.problems.base import BaseProblem
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
        return {"n_episodes": len(episodes)}

    def build_prompt(self, task_name: str, *, info_mode: Any) -> str:
        return f"# Game: fake ({task_name})\nReach the goal."

    def config_kwargs(self) -> dict[str, Any]:
        return {}


class _WritingAgent(ScriptedAgent):
    """A scripted agent that drops a working controller into the workdir on start."""

    async def start(self, *, cwd: str, **kwargs: Any) -> None:
        await super().start(cwd=cwd, **kwargs)
        Path(cwd, "solution.py").write_text(_FORWARD)


def _config() -> RunConfig:
    return RunConfig(
        agent=AgentConfig(name=AgentName.SCRIPTED),
        problem=ProblemConfig(name="fake"),
        features=["controller"],
        limits=LimitsConfig(keep_alive=10, max_moves=10),
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
    # Workdir bootstrapped: agnostic base + controller-feature templates.
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


async def test_run_task_builds_agent_from_config_when_none(tmp_path: Path) -> None:
    """With no injected agent, build_agent(scripted) runs (default turns -> exits on limit)."""
    config = _config()
    config.limits = LimitsConfig(keep_alive=1, max_moves=10)
    reason = await run_task(config, _FakeProblem(), "corridor", output_dir=str(tmp_path))
    assert reason == "loop_limit"  # the default scripted agent never submits/exits
