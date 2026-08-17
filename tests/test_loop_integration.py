"""Integration: the whole pipeline on doubles (ScriptedAgent + FakeNativeEnv).

No LLM, no real game. Builds the full stack — env server behind a TestClient, an
ControllerExecutor, the always-on controller's tools - drives ``run_session`` with a
scripted agent, and checks the on-disk artifacts (transcript.jsonl, experiment_state.json,
results.json) plus the error-path exits.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from regact.agent.events import AgentError, TextDelta, ToolCall, TurnComplete
from regact.agent.scripted_agent import ScriptedAgent
from regact.config.schema import Lifecycle, LimitsConfig
from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.envclient.client import EnvClient
from regact.features.base import RunDeps
from regact.features.controller import Controller
from regact.obs.errors import ErrorCategory
from regact.obs.logger import RunLogger
from regact.obs.transcript import TranscriptWriter
from regact.orchestration.loop import run_session
from regact.orchestration.signals import StopSignal
from regact.session.state import ExperimentState
from regact.testing.fakes import FakeNativeEnv
from regact.tools.base import Tool, ToolContext, ToolOutput

pytestmark = pytest.mark.integration

# A controller that always steps forward reaches the corridor goal in 3 moves.
_FORWARD = """\
class Controller:
    def act(self, obs):
        return 1

def get_controller():
    return Controller()
"""


class _Stack:
    """The wired pipeline + the kwargs for run_session (minus agent/first_message)."""

    def __init__(self, tmp_path: Path, *, tools: list[Tool] | None = None) -> None:
        self.workdir = tmp_path / "wd"
        self.workdir.mkdir()
        (self.workdir / "solution.py").write_text(_FORWARD)
        self.logs = tmp_path / "logs"
        self.logs.mkdir()

        server = EnvServer()
        server.register(
            "g",
            EnvSession(
                make_native=lambda: FakeNativeEnv(goal=3),
                key="g",
                renderer=RawRenderer(),
                lifecycle=MultiInstancePolicy(),
            ),
        )
        client = EnvClient(TestClient(server.app), "g")
        self.experiment = ExperimentState(
            problem_name="p", task_name="g", n_eval_episodes=1, n_videos=0
        )
        self.hooks: list = []
        if tools is None:
            deps = RunDeps(
                experiment=self.experiment,
                env_client=client,
                lifecycle=Lifecycle.MULTI_INSTANCE,
                solution_path=str(self.workdir / "solution.py"),
                submissions_dir=str(self.workdir / "submissions"),
            )
            feature = Controller(n_episodes=1, max_moves=10)
            tools = feature.tools(deps)
            self.hooks = feature.hooks(deps)
        self.tools = tools
        self.transcript = TranscriptWriter(str(self.logs / "transcript.jsonl"))
        self.logger = RunLogger(str(self.logs), task="g")
        self.state_path = str(self.logs / "experiment_state.json")
        self.limits = LimitsConfig(max_turns=10)

    async def run(self, agent: ScriptedAgent, *, stop: StopSignal | None = None) -> str:
        try:
            return await run_session(
                agent,
                first_message="Start the task.",
                experiment=self.experiment,
                tools=self.tools,
                transcript=self.transcript,
                logger=self.logger,
                limits=self.limits,
                state_path=self.state_path,
                cwd=str(self.workdir),
                hooks=self.hooks,
                stop=stop,
            )
        finally:
            self.transcript.close()
            self.logger.close()

    def transcript_types(self) -> list[str]:
        lines = (self.logs / "transcript.jsonl").read_text().splitlines()
        return [json.loads(line)["type"] for line in lines]


async def test_full_pipeline_submit_then_exit(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    agent = ScriptedAgent(
        [
            [TextDelta("Submitting."), ToolCall("c1", "SubmitSolution", {}), TurnComplete()],
            [ToolCall("c2", "ExitTask", {}), TurnComplete()],
        ]
    )
    reason = await stack.run(agent)

    assert reason == "agent_exit"
    assert stack.experiment.exit_requested is True
    assert stack.experiment.submission_count == 1

    # All three canonical artifacts on disk.
    assert Path(stack.state_path).exists()
    types = stack.transcript_types()
    assert "ToolCall" in types and "ToolResult" in types
    results = json.loads((stack.workdir / "submissions" / "0" / "results.json").read_text())
    assert results["aggregate"]["success_rate"] == 1.0
    # The teardown hook also re-scored the final solution.
    final = json.loads((stack.workdir / "submissions" / "final" / "results.json").read_text())
    assert final["aggregate"]["success_rate"] == 1.0


async def test_teardown_finalizes_when_agent_exits_without_resubmitting(tmp_path: Path) -> None:
    """The agent exits having never called SubmitSolution; finalize still scores solution.py."""
    stack = _Stack(tmp_path)
    agent = ScriptedAgent([[ToolCall("c1", "ExitTask", {}), TurnComplete()]])
    reason = await stack.run(agent)

    assert reason == "agent_exit"
    assert stack.experiment.submission_count == 0  # never submitted during the run
    # ...yet the official final result exists from the teardown hook.
    final = json.loads((stack.workdir / "submissions" / "final" / "results.json").read_text())
    assert final["aggregate"]["success_rate"] == 1.0


async def test_graceful_stop_still_finalizes(tmp_path: Path) -> None:
    """A stop signal (Ctrl+C) ends the run but still runs teardown — unlike a crash."""
    stack = _Stack(tmp_path)
    stop = StopSignal()
    stop.set()  # pre-armed: the loop stops at the first safe point
    agent = ScriptedAgent([[ToolCall("c1", "ExitTask", {}), TurnComplete()]])
    reason = await stack.run(agent, stop=stop)

    assert reason == "interrupted"
    # teardown was NOT skipped (it would be on a crash) -> the final result is written.
    assert (stack.workdir / "submissions" / "final" / "results.json").is_file()


async def test_pipeline_stops_on_persistent_backend_error(tmp_path: Path) -> None:
    """Only a wall of consecutive backend errors ends the run (transient ones retry)."""
    stack = _Stack(tmp_path)
    error_turn = [AgentError(ErrorCategory.AGENT_API, "429"), TurnComplete()]
    agent = ScriptedAgent([list(error_turn), list(error_turn), list(error_turn)])
    reason = await stack.run(agent)

    assert reason == "agent_api"
    assert stack.experiment.last_error_category == "agent_api"
    assert Path(stack.state_path).exists()  # artifacts still written on error


async def test_pipeline_survives_a_transient_backend_error(tmp_path: Path) -> None:
    """One failed turn (e.g. a 500 from a slow local server) must not kill the session."""
    stack = _Stack(tmp_path)
    agent = ScriptedAgent(
        [
            [AgentError(ErrorCategory.AGENT_API, "500"), TurnComplete()],
            [ToolCall("c1", "ExitTask", {}), TurnComplete()],
        ]
    )
    reason = await stack.run(agent)

    assert reason == "agent_exit"  # the error was retried, then the agent finished normally
    assert stack.experiment.last_error_category == "agent_api"  # ...but it stays on record
    assert "agent_error_retry" in (stack.logs / "events.jsonl").read_text()


async def test_pipeline_stops_on_keep_alive_limit(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    stack.limits = LimitsConfig(max_turns=2)
    agent = ScriptedAgent([])  # never calls ExitTask: each turn defaults to TurnComplete
    reason = await stack.run(agent)

    assert reason == "loop_limit"
    assert stack.experiment.submission_count == 0


async def test_pipeline_is_stable_over_many_turns(tmp_path: Path) -> None:
    """A long run must terminate cleanly on the turn limit, not crash or hang: the loop
    streams the transcript to disk and offloads the growing conversation to the agent,
    so it stays stable across many turns rather than accumulating state to failure."""
    stack = _Stack(tmp_path)
    stack.limits = LimitsConfig(max_turns=200)
    agent = ScriptedAgent([])  # never exits: drives straight to the turn limit
    reason = await stack.run(agent)

    assert reason == "loop_limit"  # clean stop, not loop_crash
    assert len(agent.sent) == 200  # every turn actually ran
    assert Path(stack.state_path).exists()


async def test_pipeline_survives_tool_crash(tmp_path: Path) -> None:
    class _BoomTool(Tool):
        @property
        def name(self) -> str:
            return "Boom"

        @property
        def description(self) -> str:
            return "raises"

        @property
        def input_schema(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def call(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
            raise RuntimeError("kaboom")

    stack = _Stack(tmp_path, tools=[_BoomTool()])
    agent = ScriptedAgent([[ToolCall("c1", "Boom", {}), TurnComplete()]])
    reason = await stack.run(agent)

    assert reason == "loop_crash"
    assert stack.experiment.last_error_category == "loop_crash"
    assert "turn_crash" in (stack.logs / "events.jsonl").read_text()


async def test_pipeline_stops_on_interrupt(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    stop = StopSignal()
    stop.set()  # interrupted before the first turn
    agent = ScriptedAgent([[ToolCall("c1", "SubmitSolution", {}), TurnComplete()]])
    reason = await stack.run(agent, stop=stop)

    assert reason == "interrupted"
    assert stack.experiment.submission_count == 0  # no turn ran
