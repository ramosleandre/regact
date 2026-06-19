"""Unit tests: the loop's atomic helpers in isolation (no agent, no env).

Covers the pure stop decision and the single-tool execution helper.
"""

from typing import Any

from regact.agent.events import ToolCall
from regact.config.schema import LimitsConfig
from regact.orchestration.loop import _decide_stop, _execute_framework_tool, _LoopContext
from regact.security.policy import default_policy
from regact.tools.base import Tool, ToolContext, ToolOutput

_LIMITS = LimitsConfig(keep_alive=3, max_moves=10, walltime_s=None)


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
    limits = LimitsConfig(keep_alive=100, walltime_s=5)
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
    assert logger.logged  # the execution was logged
