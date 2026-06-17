"""The single keep-alive loop.

ONE loop over the normalized ``AgentEvent`` stream, replacing the two divergent
loops in GameAgents. It is provider-independent: it sends a message, consumes the
agent's event stream, executes the *framework* tools it recognizes (submit/exit),
mirrors everything to the canonical ``transcript.jsonl``, and stops on the agent's
request, a limit, an interrupt, a backend error, or a crash.

It is deliberately agnostic of controllers/games/eval: it only knows agents,
framework tools, limits, and writers. A controller's final evaluation is a
caller concern (it happens through the submit tool during the run, or via the
entry-point wiring), so this module imports neither the executor nor a problem.

The function stays short; each responsibility is its own helper:
  ``_decide_stop`` (pure) · ``_run_turn`` · ``_dispatch_event`` · ``_execute_framework_tool``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from regact.agent.base import CodeAgent
from regact.agent.events import AgentError, AgentEvent, ToolCall, ToolResult
from regact.config.schema import LimitsConfig
from regact.obs.errors import ErrorCategory, LogComponent
from regact.obs.logger import RunLogger
from regact.obs.transcript import TranscriptWriter
from regact.orchestration.signals import StopSignal
from regact.session.state import ExperimentState
from regact.tools.base import Tool, ToolContext

_KEEP_ALIVE_MESSAGE = (
    "Continue working. Submit again to re-measure your approach to win the game, "
    "or call ExitTask when you are done."
)


@dataclass
class _LoopContext:
    """Everything the per-turn helpers need, bundled once."""

    agent: CodeAgent
    experiment: ExperimentState
    tools_by_name: dict[str, Tool]
    transcript: TranscriptWriter
    logger: RunLogger
    cwd: str


@dataclass
class _TurnOutcome:
    """What one turn produced, so the loop can decide whether to continue."""

    saw_tool_call: bool = False
    error_category: ErrorCategory | None = None  # a backend error in the stream
    crashed: bool = False  # an unexpected exception escaped the turn


async def run_session(
    agent: CodeAgent,
    *,
    experiment: ExperimentState,
    first_message: str,
    tools: list[Tool],
    transcript: TranscriptWriter,
    logger: RunLogger,
    limits: LimitsConfig,
    state_path: str,
    cwd: str,
    stop: StopSignal | None = None,
) -> str:
    """Drive one task to completion; return the exit reason."""
    ctx = _LoopContext(
        agent=agent,
        experiment=experiment,
        tools_by_name={tool.name: tool for tool in tools},
        transcript=transcript,
        logger=logger,
        cwd=cwd,
    )
    logger.log(LogComponent.ORCHESTRATOR, "INFO", "session_start", phase="bootstrap")
    experiment.save(state_path)

    message = first_message
    turns = 0
    start = time.monotonic()

    while True:
        reason = _decide_stop(
            exit_requested=experiment.exit_requested,
            interrupted=stop.is_set() if stop is not None else False,
            turns=turns,
            elapsed_s=time.monotonic() - start,
            limits=limits,
        )
        if reason is not None:
            break

        outcome = await _run_turn(message, ctx)
        experiment.save(state_path)

        if outcome.crashed:
            experiment.last_error_category = ErrorCategory.LOOP_CRASH.value
            reason = "loop_crash"
            break
        if outcome.error_category is not None:
            experiment.last_error_category = outcome.error_category.value
            reason = outcome.error_category.value
            break

        turns += 1
        message = _KEEP_ALIVE_MESSAGE

    logger.log(LogComponent.ORCHESTRATOR, "INFO", "session_end", phase="teardown", reason=reason)
    experiment.save(state_path)
    return reason


def _decide_stop(
    *,
    exit_requested: bool,
    interrupted: bool,
    turns: int,
    elapsed_s: float,
    limits: LimitsConfig,
) -> str | None:
    """Pure stop decision, checked before each turn. ``None`` means keep going."""
    if interrupted:
        return "interrupted"
    if exit_requested:
        return "agent_exit"
    if turns >= limits.keep_alive:
        return "loop_limit"
    if limits.walltime_s is not None and elapsed_s >= limits.walltime_s:
        return "walltime_limit"
    return None


async def _run_turn(message: str, ctx: _LoopContext) -> _TurnOutcome:
    """Send one message, consume the event stream, dispatch each event."""
    outcome = _TurnOutcome()
    try:
        async for event in ctx.agent.send(message):
            ctx.transcript.write(event)
            await _dispatch_event(event, ctx, outcome)
            if outcome.error_category is not None:
                break  # backend error: stop consuming this turn
    except Exception as exc:  # an unexpected fault in a tool or the adapter
        ctx.logger.log(
            LogComponent.LOOP,
            "ERROR",
            "turn_crash",
            error_category=ErrorCategory.LOOP_CRASH,
            error=f"{type(exc).__name__}: {exc}",
        )
        outcome.crashed = True
    return outcome


async def _dispatch_event(event: AgentEvent, ctx: _LoopContext, outcome: _TurnOutcome) -> None:
    """Route one event: execute framework tools, record backend errors, else observe."""
    if isinstance(event, ToolCall):
        tool = ctx.tools_by_name.get(event.name)
        if tool is not None:  # a framework tool: the loop owns its execution
            result = await _execute_framework_tool(tool, event, ctx)
            ctx.transcript.write(result)
            await ctx.agent.inject(result.output)
            outcome.saw_tool_call = True
    elif isinstance(event, AgentError):
        ctx.logger.log(
            LogComponent.AGENT,
            "ERROR",
            "agent_error",
            error_category=event.category,
            message=event.message,
        )
        outcome.error_category = event.category


async def _execute_framework_tool(tool: Tool, call: ToolCall, ctx: _LoopContext) -> ToolResult:
    """Run one framework tool and normalize its result (controlled failures stay results)."""
    output = await tool.call(call.input, ToolContext(cwd=ctx.cwd))
    ctx.logger.log(
        LogComponent.ORCHESTRATOR,
        "INFO",
        "tool_executed",
        phase="submit",
        tool=call.name,
        is_error=output.is_error,
    )
    return ToolResult(id=call.id, output=str(output.data), is_error=output.is_error)
