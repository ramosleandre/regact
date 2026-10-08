"""Integration: the whole pipeline on doubles (ScriptedAgent + FakeNativeEnv).

No LLM, no real game. Builds the full stack — env server behind a TestClient, an
ControllerExecutor, the always-on controller's tools - drives ``run_session`` with a
scripted agent, and checks the on-disk artifacts (transcript.jsonl, experiment_state.json,
results.json) plus the error-path exits.
"""

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from regact.agent.events import (
    AgentError,
    IterationComplete,
    TextDelta,
    ToolCall,
    ToolResult,
)
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
from regact.protocols.policy_search import PolicySearchSession
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
        self.limits = LimitsConfig(max_turns_per_task=10)

    async def run(
        self, agent: ScriptedAgent, *, stop: StopSignal | None = None, is_perfect=None
    ) -> str:  # type: ignore[no-untyped-def]
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
                protocol=PolicySearchSession(
                    experiment=self.experiment, hooks=self.hooks, is_perfect=is_perfect
                ),
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
            [
                TextDelta("Submitting."),
                ToolCall("c1", "SubmitSolution", {}),
                IterationComplete(),
            ],
            [ToolCall("c2", "ExitTask", {}), IterationComplete()],
        ]
    )
    reason = await stack.run(agent)

    assert reason == "agent_exit"
    assert stack.experiment.exit_requested is True
    assert stack.experiment.submission_count == 1
    assert stack.experiment.tool_calls_total == 2  # SubmitSolution + ExitTask
    assert stack.experiment.turn == 2  # both turns ran

    # All three canonical artifacts on disk.
    assert Path(stack.state_path).exists()
    types = stack.transcript_types()
    assert "ToolCall" in types and "ToolResult" in types
    results = json.loads((stack.workdir / "submissions" / "0" / "results.json").read_text())
    assert results["aggregate"]["success_rate"] == 1.0
    # The teardown hook also re-scored the final solution.
    final = json.loads((stack.workdir / "submissions" / "final" / "results.json").read_text())
    assert final["aggregate"]["success_rate"] == 1.0


async def test_teardown_finalizes_when_agent_exits_without_resubmitting(
    tmp_path: Path,
) -> None:
    """The agent exits having never called SubmitSolution; finalize still scores solution.py."""
    stack = _Stack(tmp_path)
    agent = ScriptedAgent([[ToolCall("c1", "ExitTask", {}), IterationComplete()]])
    reason = await stack.run(agent)

    assert reason == "agent_exit"
    assert stack.experiment.submission_count == 0  # never submitted during the run
    # ...yet the official final result exists from the teardown hook.
    final = json.loads((stack.workdir / "submissions" / "final" / "results.json").read_text())
    assert final["aggregate"]["success_rate"] == 1.0


async def test_exit_mid_turn_stops_before_later_calls(tmp_path: Path) -> None:
    """ExitTask fired MID-turn ends the run at once. alancode runs its whole loop inside one send(),
    so a later call in the SAME turn (a SubmitSolution after ExitTask) must NOT run - the loop can't
    wait for the next send to honor the exit. Guards the ARC hang (28min post-ExitTask)."""
    stack = _Stack(tmp_path)
    agent = ScriptedAgent(
        [
            [
                ToolCall("c1", "ExitTask", {}),
                ToolCall("c2", "SubmitSolution", {}),
                IterationComplete(),
            ]
        ]
    )
    reason = await stack.run(agent)

    assert reason == "agent_exit"
    assert stack.experiment.submission_count == 0  # the post-ExitTask submit did NOT run


async def test_graceful_stop_still_finalizes(tmp_path: Path) -> None:
    """A stop signal (Ctrl+C) ends the run but still runs teardown — unlike a crash."""
    stack = _Stack(tmp_path)
    stop = StopSignal()
    stop.set()  # pre-armed: the loop stops at the first safe point
    agent = ScriptedAgent([[ToolCall("c1", "ExitTask", {}), IterationComplete()]])
    reason = await stack.run(agent, stop=stop)

    assert reason == "interrupted"
    # teardown was NOT skipped (it would be on a crash) -> the final result is written.
    assert (stack.workdir / "submissions" / "final" / "results.json").is_file()


async def test_pipeline_stops_on_persistent_backend_error(tmp_path: Path) -> None:
    """Only a wall of consecutive backend errors ends the run (transient ones retry)."""
    stack = _Stack(tmp_path)
    error_turn = [AgentError(ErrorCategory.AGENT_API, "429"), IterationComplete()]
    agent = ScriptedAgent([list(error_turn), list(error_turn), list(error_turn)])
    reason = await stack.run(agent)

    assert reason == "agent_api"
    assert stack.experiment.last_error_category == "agent_api"
    assert Path(stack.state_path).exists()  # artifacts still written on error


async def test_errors_at_the_end_of_turns_that_worked_do_not_add_up(tmp_path: Path) -> None:
    """A CLI agent's turn can hold many tool calls and still end on a backend error (an answer
    over the output cap, say). Such turns are progress, not a dead backend."""
    stack = _Stack(tmp_path)

    def working_then_failing(i: int) -> list:
        return [
            ToolCall(f"c{i}", "Bash", {}),
            ToolResult(f"c{i}", "done"),
            AgentError(ErrorCategory.AGENT_API, "response exceeded the output token maximum"),
            IterationComplete(),
        ]

    turns = [working_then_failing(i) for i in range(5)]
    agent = ScriptedAgent([*turns, [ToolCall("x", "ExitTask", {}), IterationComplete()]])
    reason = await stack.run(agent)

    assert reason == "agent_exit" and stack.experiment.tool_calls_total == 6


async def test_pipeline_survives_a_transient_backend_error(tmp_path: Path) -> None:
    """One failed turn (e.g. a 500 from a slow local server) must not kill the session."""
    stack = _Stack(tmp_path)
    agent = ScriptedAgent(
        [
            [AgentError(ErrorCategory.AGENT_API, "500"), IterationComplete()],
            [ToolCall("c1", "ExitTask", {}), IterationComplete()],
        ]
    )
    reason = await stack.run(agent)

    assert reason == "agent_exit"  # the error was retried, then the agent finished normally
    assert stack.experiment.last_error_category == "agent_api"  # ...but it stays on record
    assert "agent_error_retry" in (stack.logs / "events.jsonl").read_text()


async def test_a_usage_limit_is_waited_out_with_the_task_clock_paused(
    tmp_path: Path, monkeypatch
) -> None:
    from regact.orchestration import loop

    monkeypatch.setattr(loop, "_USAGE_LIMIT_MARGIN_S", 0)

    class LimitedAgent(ScriptedAgent):
        def usage_limit_reset(self, message: str) -> float | None:
            return time.time() + 1.2 if "session limit" in message else None

    stack = _Stack(tmp_path)
    # A task budget shorter than the wait itself.
    stack.limits = LimitsConfig(max_seconds_per_task=1, wait_for_usage_limit=True)
    limit = AgentError(ErrorCategory.AGENT_API, "You've hit your session limit")
    limited = [[limit, IterationComplete()] for _ in range(4)]
    agent = LimitedAgent([*limited, [ToolCall("c1", "ExitTask", {}), IterationComplete()]])
    reason = await stack.run(agent)

    # Four limits in a row: more than the error-retry allowance, and 4.8 s of waiting against
    # a 1 s task budget. Neither ended the run.
    assert reason == "agent_exit"
    assert stack.experiment.usage_limit_waits == 4
    assert stack.experiment.usage_limit_waited_s >= 4.0
    assert stack.experiment.duration_s < 1.0
    events = (stack.logs / "events.jsonl").read_text()
    assert events.count('"usage_limit_wait"') == 4 and "usage_limit_resumed" in events


async def test_a_usage_limit_ends_the_task_at_once_by_default(tmp_path: Path) -> None:
    class LimitedAgent(ScriptedAgent):
        def usage_limit_reset(self, message: str) -> float | None:
            return 1_800_000_000.0 if "session limit" in message else None

    stack = _Stack(tmp_path)
    limit = AgentError(ErrorCategory.AGENT_API, "You've hit your session limit")
    agent = LimitedAgent([[limit, IterationComplete()] for _ in range(3)])
    reason = await stack.run(agent)

    assert reason == "usage_limit" and len(agent.sent) == 1  # no retry against a closed window
    assert stack.experiment.exit_detail == "resets 1800000000"
    assert stack.experiment.resumable() and stack.experiment.usage_limit_waits == 0


async def test_a_usage_limit_that_resets_too_late_ends_the_run(tmp_path: Path) -> None:
    class LimitedAgent(ScriptedAgent):
        def usage_limit_reset(self, message: str) -> float | None:
            return time.time() + 3600

    stack = _Stack(tmp_path)
    stack.limits = LimitsConfig(
        experiment_deadline_unix=int(time.time()) + 600, wait_for_usage_limit=True
    )
    limit = AgentError(ErrorCategory.AGENT_API, "You've hit your session limit")
    reason = await stack.run(LimitedAgent([[limit, IterationComplete()]] * 5))

    assert reason == "agent_api" and stack.experiment.usage_limit_waits == 0


async def test_a_resumed_task_gets_what_the_new_deadline_leaves(tmp_path: Path) -> None:
    """A task that already ran 1,000 s, resumed in a job that ends in 500 s, has about 500 s."""
    stack = _Stack(tmp_path)
    stack.experiment.duration_s = 1000.0
    stack.limits = LimitsConfig(experiment_deadline_unix=int(time.time()) + 500)
    agent = ScriptedAgent([[ToolCall("c1", "ExitTask", {}), IterationComplete()]])
    reason = await stack.run(agent)

    assert reason == "agent_exit"  # not walltime_limit at the first check
    assert 1000.0 <= stack.experiment.duration_s < 1010.0


@pytest.mark.parametrize(
    "limits,detail",
    [
        (LimitsConfig(max_seconds_per_task=0), "task_budget"),
        (LimitsConfig(experiment_deadline_unix=1), "experiment_deadline"),
    ],
)
async def test_a_walltime_end_records_which_limit_caused_it(
    tmp_path: Path, limits: LimitsConfig, detail: str
) -> None:
    stack = _Stack(tmp_path)
    stack.limits = limits
    reason = await stack.run(ScriptedAgent([]))

    assert reason == "walltime_limit" and stack.experiment.exit_detail == detail


async def test_waits_for_usage_limits_are_capped_over_the_task(tmp_path: Path, monkeypatch) -> None:
    from regact.orchestration import loop

    monkeypatch.setattr(loop, "_USAGE_LIMIT_MARGIN_S", 0)
    monkeypatch.setattr(loop, "_MAX_USAGE_LIMIT_TOTAL_WAIT_S", 1.0)

    class LimitedAgent(ScriptedAgent):
        def usage_limit_reset(self, message: str) -> float | None:
            return time.time() + 0.6

    stack = _Stack(tmp_path)
    stack.limits = LimitsConfig(wait_for_usage_limit=True)
    limit = AgentError(ErrorCategory.AGENT_API, "You've hit your session limit")
    reason = await stack.run(LimitedAgent([[limit, IterationComplete()] for _ in range(8)]))

    assert reason == "agent_api" and stack.experiment.usage_limit_waits == 1


async def test_pipeline_stops_on_keep_alive_limit(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    stack.limits = LimitsConfig(max_turns_per_task=2)
    agent = ScriptedAgent([])  # never calls ExitTask: each turn defaults to IterationComplete
    reason = await stack.run(agent)

    assert reason == "loop_limit"
    assert stack.experiment.submission_count == 0


async def test_pipeline_stops_on_tool_call_limit(tmp_path: Path) -> None:
    """max_tool_calls caps TOTAL tool calls across the run - agent-agnostic, turn-independent."""
    stack = _Stack(tmp_path)
    stack.limits = LimitsConfig(max_turns_per_task=100, max_tool_calls=3)
    # One Bash call per turn (never submits/exits); the loop counts every ToolCall event.
    agent = ScriptedAgent(
        [
            [ToolCall("c", "Bash", {}), ToolResult("c", "done"), IterationComplete()]
            for _ in range(6)
        ]
    )
    reason = await stack.run(agent)

    assert reason == "tool_call_limit"
    assert stack.experiment.tool_calls_total == 3  # stopped exactly at the budget
    assert len(agent.sent) == 3  # three turns ran; the cap fired before the fourth


async def test_pipeline_aborts_mid_send_at_the_tool_call_budget(tmp_path: Path) -> None:
    """The budget is enforced MID-send: one send() emitting more calls than the budget is cut off at
    it, not after (the CLI agents run their whole loop in one send() with no inner tool knob)."""
    stack = _Stack(tmp_path)
    stack.limits = LimitsConfig(max_turns_per_task=100, max_tool_calls=3)
    # A single send() emitting five Bash calls: the loop must stop after the third, not run all 5.
    calls = [
        event
        for i in range(5)
        for event in (ToolCall(f"c{i}", "Bash", {}), ToolResult(f"c{i}", "done"))
    ]
    agent = ScriptedAgent([[*calls, IterationComplete()]])
    reason = await stack.run(agent)

    assert reason == "tool_call_limit"
    assert stack.experiment.tool_calls_total == 3  # cut off mid-send, not the 5 it would have run
    assert len(agent.sent) == 1  # it all happened inside the first (and only) send


async def test_pipeline_stops_when_a_submission_is_perfect(tmp_path: Path) -> None:
    """With is_perfect, a perfect submission ends the run at once - no ExitTask, budget to spare."""
    stack = _Stack(tmp_path)
    agent = ScriptedAgent(
        [
            [
                ToolCall("c1", "SubmitSolution", {}),
                IterationComplete(),
            ],  # _FORWARD solves -> 1.0
            [
                ToolCall("c2", "Bash", {}),
                IterationComplete(),
            ],  # must NOT run: the run stops first
        ]
    )
    reason = await stack.run(agent, is_perfect=lambda agg: agg.get("success_rate", 0) >= 1.0)

    assert reason == "solved"
    assert stack.experiment.submission_count == 1
    assert len(agent.sent) == 1  # stopped right after the perfect submission


async def test_pipeline_stops_at_the_perfect_submission_not_the_turn_end(
    tmp_path: Path,
) -> None:
    """Solved must stop at the SUBMISSION, not when the turn ends. alancode submits many times
    inside one send(), so a turn-granular check keeps re-scoring the winning controller: a live run
    paid for six identical perfect submissions before its turn closed."""
    stack = _Stack(tmp_path)
    # ONE send: a perfect submission followed by more calls that must never be dispatched.
    rest = [ToolCall(f"x{i}", "Bash", {}) for i in range(5)]
    agent = ScriptedAgent([[ToolCall("c1", "SubmitSolution", {}), *rest, IterationComplete()]])
    reason = await stack.run(agent, is_perfect=lambda agg: agg.get("success_rate", 0) >= 1.0)

    assert reason == "solved"
    assert stack.experiment.submission_count == 1  # not re-scored for the rest of the turn
    assert stack.experiment.tool_calls_total == 1  # aborted at the submission; the 5 never ran


async def test_doom_loop_breaker_stops_a_no_tool_agent(tmp_path: Path) -> None:
    """A degenerate agent that never calls a tool (e.g. a temp0 model emitting the same
    unparseable garbage each turn) is cut off after max_consecutive_no_tool_turns, not left to
    burn the full budget. (Off by default -> test_pipeline_is_stable_over_many_turns runs 200
    no-tool turns to loop_limit, guarding the default.)"""
    stack = _Stack(tmp_path)
    stack.limits = LimitsConfig(max_turns_per_task=50, max_consecutive_no_tool_turns=3)
    agent = ScriptedAgent([])  # every turn is a no-tool turn
    reason = await stack.run(agent)

    assert reason == "no_tool_progress"
    assert len(agent.sent) == 3  # gave up at the breaker, not at max_turns_per_task=50


async def test_pipeline_is_stable_over_many_turns(tmp_path: Path) -> None:
    """A long run must terminate cleanly on the turn limit, not crash or hang: the loop
    streams the transcript to disk and offloads the growing conversation to the agent,
    so it stays stable across many turns rather than accumulating state to failure."""
    stack = _Stack(tmp_path)
    stack.limits = LimitsConfig(max_turns_per_task=200)
    agent = ScriptedAgent([])  # never exits: drives straight to the turn limit
    reason = await stack.run(agent)

    assert reason == "loop_limit"  # clean stop, not loop_crash
    assert len(agent.sent) == 200  # every turn actually ran
    assert Path(stack.state_path).exists()
    # Spinning detector: 200 turns, zero tool calls -> tool_calls_total << turn.
    assert stack.experiment.turn == 200
    assert stack.experiment.tool_calls_total == 0


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
    agent = ScriptedAgent([[ToolCall("c1", "Boom", {}), IterationComplete()]])
    reason = await stack.run(agent)

    assert reason == "loop_crash"
    assert stack.experiment.last_error_category == "loop_crash"
    assert "turn_crash" in (stack.logs / "events.jsonl").read_text()


async def test_pipeline_stops_on_interrupt(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    stop = StopSignal()
    stop.set()  # interrupted before the first turn
    agent = ScriptedAgent([[ToolCall("c1", "SubmitSolution", {}), IterationComplete()]])
    reason = await stack.run(agent, stop=stop)

    assert reason == "interrupted"
    assert stack.experiment.submission_count == 0  # no turn ran


async def test_the_verdict_is_on_disk_before_teardown_runs(tmp_path: Path) -> None:
    """Teardown re-scores the controller, which on a slow serve outlasts what is left of an
    exhausted budget. bench-04 job 5418021 decided walltime_limit correctly and was then SIGKILLed
    two minutes into that re-score, leaving exit_reason=None - the run read as "still running"
    forever. A hook that inspects the state file mid-teardown must already see the verdict."""
    from regact.features.base import Hook, HookPhase

    seen: dict[str, Any] = {}
    stack = _Stack(tmp_path)

    class _ReadStateMidTeardown(Hook):
        phase = HookPhase.TEARDOWN

        async def run(self) -> None:
            state = json.loads(Path(stack.state_path).read_text())
            seen["exit_reason"] = state.get("exit_reason")

    stack.hooks = [*stack.hooks, _ReadStateMidTeardown()]
    reason = await stack.run(ScriptedAgent([[ToolCall("c1", "ExitTask", {}), IterationComplete()]]))

    assert reason == "agent_exit"
    assert seen["exit_reason"] == "agent_exit"  # already persisted, not written after teardown


@pytest.mark.parametrize("stop_kind", ["budget", "solved", "exit"])
async def test_stop_waits_for_enclosing_shell_result(tmp_path: Path, stop_kind: str) -> None:
    """A submission/exit inside a shell command must not interrupt its remaining writes."""
    stack = _Stack(tmp_path)
    stack.limits = LimitsConfig(
        max_turns_per_task=10, max_tool_calls=1 if stop_kind == "budget" else None
    )
    solution = stack.workdir / "solution.py"

    class WritingAgent(ScriptedAgent):
        async def send(self, message):
            yield ToolCall("shell", "Bash", {"command": "write; SubmitSolution; write"})
            solution.write_text("")  # reproduce the truncation window at call start
            if stop_kind == "exit":
                stack.experiment.exit_requested = True
            else:
                # Execute the real tool as the HTTP bridge would, inside the shell call.
                submit = next(t for t in stack.tools if t.name == "SubmitSolution")
                solution.write_text(_FORWARD)
                await submit.call({}, ToolContext(cwd=str(stack.workdir)))
                solution.write_text("")
            yield TextDelta("Intermediate event while the command is still running")
            assert not self.aborted
            solution.write_text(_FORWARD)
            yield ToolResult("shell", "finished")
            pytest.fail("A new command must not start after the completed-call stop")

    agent = WritingAgent()
    reason = await stack.run(
        agent,
        is_perfect=(lambda a: a.get("success_rate") == 1) if stop_kind == "solved" else None,
    )
    assert (
        reason == {"budget": "tool_call_limit", "solved": "solved", "exit": "agent_exit"}[stop_kind]
    )
    assert agent.aborted
    assert solution.read_text() == _FORWARD
    result = json.loads((stack.workdir / "submissions/final/results.json").read_text())
    assert result["aggregate"]["success_rate"] == 1
    assert stack.transcript_types().count("ToolResult") == 1


async def test_tool_budget_waits_for_all_started_calls(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    stack.limits = LimitsConfig(max_turns_per_task=10, max_tool_calls=2)

    class ParallelAgent(ScriptedAgent):
        async def send(self, message):
            yield ToolCall("a", "Bash", {})
            yield ToolCall("b", "Bash", {})
            yield ToolResult("b", "failed", is_error=True)
            yield TextDelta("a is still writing")
            assert not self.aborted
            yield ToolResult("a", "finished")
            pytest.fail("Budget must stop after both results")

    agent = ParallelAgent()
    assert await stack.run(agent) == "tool_call_limit"
    assert stack.transcript_types().count("ToolResult") == 2
    assert agent.aborted


async def test_inline_framework_result_satisfies_tool_budget(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    stack.limits = LimitsConfig(max_turns_per_task=10, max_tool_calls=1)
    agent = ScriptedAgent(
        [[ToolCall("submit", "SubmitSolution", {}), ToolCall("extra", "Bash", {})]]
    )
    assert await stack.run(agent) == "tool_call_limit"
    assert stack.experiment.submission_count == 1
    assert stack.experiment.tool_calls_total == 1
    assert stack.transcript_types().count("ToolResult") == 1


async def test_codex_completion_only_patch_counts_and_finishes_before_stop(
    tmp_path: Path,
) -> None:
    from regact.agent.codex_adapter import CodexAgent

    stack = _Stack(tmp_path)
    stack.limits = LimitsConfig(max_turns_per_task=10, max_tool_calls=1)
    parser = CodexAgent()

    class PatchAgent(ScriptedAgent):
        async def send(self, message):
            # CLI reports this patch only AFTER it has written both files.
            item = {
                "id": "item_0",
                "type": "file_change",
                "changes": [
                    {"path": "solution.py", "kind": "update"},
                    {"path": "helper.py", "kind": "add"},
                ],
                "status": "completed",
            }
            for event in parser._parse_events({"type": "item.completed", "item": item}):
                assert not self.aborted
                yield event
            pytest.fail("Budget must stop before another operation")

    agent = PatchAgent()
    assert await stack.run(agent) == "tool_call_limit"
    assert stack.experiment.tool_calls_total == 1
    assert stack.transcript_types().count("ToolResult") == 1
    assert agent.aborted


@pytest.mark.parametrize("pending, force", [(False, False), (True, False), (True, True)])
async def test_interrupt_during_long_turn(tmp_path, pending, force):
    import asyncio

    stack = _Stack(tmp_path, tools=[])
    stop = StopSignal()
    completed = False

    class WaitingAgent(ScriptedAgent):
        async def send(self, message):
            nonlocal completed
            if pending:
                yield ToolCall("write", "Bash", {"command": "write file"})
            stop.set()
            if force:
                stop.set()
            await asyncio.sleep(0.15)
            if pending and not force:
                assert not self.aborted
                completed = True
                yield ToolResult("write", "saved")
                pytest.fail("Must stop after the pending result")
            await asyncio.Event().wait()

    agent = WaitingAgent()
    assert await asyncio.wait_for(stack.run(agent, stop=stop), 2) == "interrupted"
    assert agent.aborted
    assert completed == (pending and not force)
    assert "agent_error" not in (stack.logs / "events.jsonl").read_text()


async def test_absolute_deadline_caps_a_budget_that_counts_from_session_start(
    tmp_path: Path,
) -> None:
    """GLM-5.2 runs started late enough that start + max_seconds_per_task fell after the Slurm
    kill, so they were SIGKILLed with no verdict. A deadline already past ends the run on
    walltime_limit before any turn, however generous the relative budget."""
    stack = _Stack(tmp_path)
    stack.limits = LimitsConfig(
        max_turns_per_task=10, max_seconds_per_task=36000, experiment_deadline_unix=int(time.time()) - 1
    )
    agent = ScriptedAgent([[TextDelta("never sent"), IterationComplete()]])
    assert await stack.run(agent) == "walltime_limit"
    assert stack.experiment.turn == 0


def test_seconds_left_takes_the_tighter_of_budget_and_deadline() -> None:
    now = int(time.time())
    assert LimitsConfig(max_seconds_per_task=100).seconds_left() == 100
    assert 0 < LimitsConfig(max_seconds_per_task=100, experiment_deadline_unix=now + 50).seconds_left() <= 50
    assert LimitsConfig(max_seconds_per_task=10, experiment_deadline_unix=now + 500).seconds_left() == 10
    assert 400 < LimitsConfig(experiment_deadline_unix=now + 500).seconds_left() <= 500
