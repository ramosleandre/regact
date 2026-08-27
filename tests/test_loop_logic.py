"""Unit tests: the loop's atomic helpers in isolation (no agent, no env).

Covers the pure stop decision and the single-tool execution helper.
"""

from typing import Any

from regact.agent.events import ToolCall
from regact.config.schema import LimitsConfig
from regact.orchestration.loop import _decide_stop, _execute_framework_tool, _LoopContext
from regact.security.policy import default_policy
from regact.tools.base import Tool, ToolContext, ToolOutput

_LIMITS = LimitsConfig(max_turns=3, max_seconds_per_task=None)


def test_decide_stop_continues_by_default() -> None:
    assert (
        _decide_stop(
            exit_requested=False, interrupted=False, turns=0, elapsed_s=0.0, limits=_LIMITS
        )
        is None
    )


def test_decide_stop_interrupt_wins_over_everything() -> None:
    reason = _decide_stop(
        exit_requested=True, interrupted=True, turns=99, elapsed_s=999.0, limits=_LIMITS
    )
    assert reason == "interrupted"


def test_decide_stop_agent_exit() -> None:
    reason = _decide_stop(
        exit_requested=True, interrupted=False, turns=0, elapsed_s=0.0, limits=_LIMITS
    )
    assert reason == "agent_exit"


def test_decide_stop_keep_alive_limit() -> None:
    reason = _decide_stop(
        exit_requested=False, interrupted=False, turns=3, elapsed_s=0.0, limits=_LIMITS
    )
    assert reason == "loop_limit"


def test_decide_stop_walltime_limit() -> None:
    limits = LimitsConfig(max_turns=100, max_seconds_per_task=5)
    reason = _decide_stop(
        exit_requested=False, interrupted=False, turns=0, elapsed_s=6.0, limits=limits
    )
    assert reason == "walltime_limit"


class _OkTool(Tool):
    @property
    def name(self) -> str:
        return "Ok"

    @property
    def description(self) -> str:
        return "ok"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def call(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        return ToolOutput(data={"v": 1}, is_error=False)


class _FakeLogger:
    def __init__(self) -> None:
        self.logged: list[tuple[Any, Any]] = []

    def log(self, *args: Any, **kwargs: Any) -> None:
        self.logged.append((args, kwargs))


def _ctx(logger: Any) -> _LoopContext:
    return _LoopContext(
        agent=None,  # type: ignore[arg-type]
        experiment=None,  # type: ignore[arg-type]
        tools_by_name={},
        transcript=None,  # type: ignore[arg-type]
        logger=logger,
        cwd="/tmp",
        policy=default_policy(),
    )


async def test_execute_framework_tool_normalizes_output() -> None:
    logger = _FakeLogger()
    result = await _execute_framework_tool(_OkTool(), ToolCall("c1", "Ok", {}), _ctx(logger))
    assert result.id == "c1"
    assert result.is_error is False
    assert "v" in result.output


async def test_logging_tool_logs_every_call() -> None:
    # Execution logging lives on the tool wrapper, shared by every dispatch path
    # (loop, HTTP control channel, backend-executed native tools).
    from regact.tools.base import LoggingTool, ToolContext

    logger = _FakeLogger()
    wrapped = LoggingTool(_OkTool(), logger)  # type: ignore[arg-type]
    assert wrapped.name == _OkTool().name
    output = await wrapped.call({}, ToolContext(cwd="/tmp"))
    assert output.is_error is False
    assert logger.logged
    _, kwargs = logger.logged[0]
    assert kwargs["tool"] == "Ok"


async def test_flagging_warning_injected_up_to_cap() -> None:
    """A flagged call injects the sandbox warning on the agent's next turn, capped at
    flagging_warning_cap; 0 disables it; a clean call injects nothing (but every flag
    is counted)."""
    from regact.orchestration.loop import _FLAGGING_WARNING, _flag_suspicious_call
    from regact.session.state import ExperimentState

    class _RecordingAgent:
        def __init__(self) -> None:
            self.injected: list[str] = []

        async def inject(self, message: str) -> None:
            self.injected.append(message)

    def make(cap: int) -> tuple[_LoopContext, _RecordingAgent]:
        agent = _RecordingAgent()
        ctx = _LoopContext(
            agent=agent,  # type: ignore[arg-type]
            experiment=ExperimentState(problem_name="p", task_name="t"),
            tools_by_name={},
            transcript=None,  # type: ignore[arg-type]
            logger=_FakeLogger(),
            cwd="/tmp",
            policy=default_policy(),
            flagging_warning_cap=cap,
        )
        return ctx, agent

    bad = ToolCall("c", "Bash", {"command": "python -c 'import minigrid'"})  # forbidden import
    clean = ToolCall("c", "Bash", {"command": "ls code_library"})

    capped, agent = make(2)
    for _ in range(3):
        await _flag_suspicious_call(bad, capped)
    assert agent.injected == [_FLAGGING_WARNING, _FLAGGING_WARNING]  # capped at 2
    assert capped.experiment.flagged_tool_calls == 3  # but every flag is still COUNTED

    off, off_agent = make(0)
    await _flag_suspicious_call(bad, off)
    assert off_agent.injected == []  # 0 = never inject

    ok, ok_agent = make(3)
    await _flag_suspicious_call(clean, ok)
    assert ok_agent.injected == []  # a clean call is not flagged -> no warning
