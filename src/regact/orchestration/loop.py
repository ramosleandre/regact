"""The single keep-alive loop.

ONE loop over the normalized ``AgentEvent`` stream, replacing the two divergent
loops in GameAgents. It is provider-independent: it sends a message, consumes the
agent's event stream, executes the *framework* tools it recognizes (submit/exit),
mirrors everything to the canonical ``transcript.jsonl``, and stops on the agent's
request, a limit, an interrupt, a persistent backend error, or a crash.

It is deliberately agnostic of controllers/games/eval: it only knows agents,
framework tools, hooks, limits, and writers — all generic interfaces. It imports
neither the executor nor a problem. Feature-specific teardown work (e.g. re-scoring
the final solution) arrives as :class:`Hook` objects it fires by phase, the same
way feature ``tools`` arrive as :class:`Tool` objects it executes on demand.

The function stays short; each responsibility is its own helper:
  ``_decide_stop`` (pure) · ``_run_turn`` · ``_dispatch_event`` ·
  ``_execute_framework_tool`` · ``_run_teardown_hooks``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass

from regact.agent.base import CodeAgent
from regact.agent.events import (
    AgentError,
    AgentEvent,
    SystemPrompt,
    ToolCall,
    ToolResult,
    UserMessage,
)
from regact.config.schema import LimitsConfig
from regact.features.base import Hook, HookPhase
from regact.obs.errors import ErrorCategory, LogComponent
from regact.obs.logger import RunLogger
from regact.obs.transcript import TranscriptWriter
from regact.orchestration.signals import StopSignal
from regact.security.detection import flag_os_denial, flag_tool_call
from regact.security.policy import SecurityPolicy, default_policy
from regact.session.state import ExperimentState
from regact.tools.base import Tool, ToolContext

_ABORTED_REASONS = frozenset({"loop_crash"})

_KEEP_ALIVE_MESSAGE = (
    "Keep-alive reminder - Continue working or finish your work : 1) Produce a "
    "controller in solution.py, 2) Submit this solution by calling SubmitSolution, "
    "3) Exit your task through ExitTask if you are satisfied with your solution."
)

# A single backend error (one 500/timeout from a slow local server) must not end the
# session; only a wall of them means the backend is really gone.
_MAX_CONSECUTIVE_ERROR_TURNS = 3
_ERROR_RETRY_MESSAGE = (
    "Your previous turn was interrupted by a backend error. "
    "Continue working from where you left off."
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
    policy: SecurityPolicy  # for flagging (not blocking) suspicious tool calls
    state_path: str = ""  # where to persist ExperimentState (saved live, per event)
    start: float = 0.0  # time.monotonic() at the run's start, for the live duration
    move_count: Callable[[], int] | None = None  # polls the env's step count, for the live state


@dataclass
class _TurnOutcome:
    """What one turn produced, so the loop can decide whether to continue."""

    saw_tool_call: bool = False  # agent emitted >=1 tool call (framework, bash, or native)
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
    system_prompt: str | None = None,
    hooks: list[Hook] | None = None,
    stop: StopSignal | None = None,
    move_count: Callable[[], int] | None = None,
) -> str:
    """Drive one task to completion; return the exit reason."""
    start = time.monotonic()
    ctx = _LoopContext(
        agent=agent,
        experiment=experiment,
        tools_by_name={tool.name: tool for tool in tools},
        transcript=transcript,
        logger=logger,
        cwd=cwd,
        policy=default_policy(),
        state_path=state_path,
        start=start,
        move_count=move_count,
    )
    logger.log(LogComponent.ORCHESTRATOR, "INFO", "session_start", phase="bootstrap")
    experiment.save(state_path)
    if system_prompt:  # record the inputs so the viewer shows the whole session, not just replies
        transcript.write(SystemPrompt(system_prompt))

    message = first_message
    turns = 0
    error_turns = 0  # consecutive turns that ended in a backend error
    no_tool_turns = 0  # consecutive turns that produced no tool call (doom-loop breaker)
    watchdog = _spawn_walltime_watchdog(agent, start, limits.max_seconds_per_task)
    try:
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

            budget = limits.max_seconds_per_task
            if budget is not None and time.monotonic() - start >= budget:
                reason = "walltime_limit"  # the watchdog aborted a long turn; this is not an error
                break
            if outcome.crashed:
                experiment.last_error_category = ErrorCategory.LOOP_CRASH.value
                reason = "loop_crash"
                break
            if outcome.error_category is not None:
                experiment.last_error_category = outcome.error_category.value
                error_turns += 1
                if error_turns >= _MAX_CONSECUTIVE_ERROR_TURNS:
                    reason = outcome.error_category.value
                    break
                logger.log(
                    LogComponent.ORCHESTRATOR,
                    "WARNING",
                    "agent_error_retry",
                    attempt=error_turns,
                    max_attempts=_MAX_CONSECUTIVE_ERROR_TURNS,
                )
                turns += 1
                message = _ERROR_RETRY_MESSAGE
                continue

            error_turns = 0
            turns += 1
            # Doom-loop breaker: a degenerate model that makes no tool call just burns walltime.
            no_tool_turns = 0 if outcome.saw_tool_call else no_tool_turns + 1
            if 0 < limits.max_consecutive_no_tool_turns <= no_tool_turns:
                reason = "no_tool_progress"
                break
            message = _KEEP_ALIVE_MESSAGE
    finally:
        if watchdog is not None:
            watchdog.cancel()

    await _run_teardown_hooks(hooks or [], reason, ctx)
    experiment.exit_reason = reason  # "running" until set; the viewer shows it as the status
    logger.log(LogComponent.ORCHESTRATOR, "INFO", "session_end", phase="teardown", reason=reason)
    _save_state(ctx)
    return reason


def _save_state(ctx: _LoopContext) -> None:
    """Persist the run state with the live duration (called per event, so the viewer
    reflects a long single turn — e.g. a codex ``exec`` — as it happens, not only at its end)."""
    ctx.experiment.duration_s = round(time.monotonic() - ctx.start, 1)
    if ctx.move_count is not None:
        ctx.experiment.env_moves = ctx.move_count()
    if ctx.experiment.agent_session_id is None:
        ctx.experiment.agent_session_id = ctx.agent.session_id()
    ctx.experiment.save(ctx.state_path)


async def _run_teardown_hooks(hooks: list[Hook], reason: str, ctx: _LoopContext) -> None:
    """Fire TEARDOWN hooks unless the run was aborted; a hook fault never aborts teardown."""
    if reason in _ABORTED_REASONS:
        return
    for hook in hooks:
        if hook.phase is not HookPhase.TEARDOWN:
            continue
        try:
            await hook.run()
        except Exception as exc:
            ctx.logger.log(
                LogComponent.EVAL,
                "ERROR",
                "hook_failed",
                phase="teardown",
                error_category=ErrorCategory.EVAL_HARNESS,
                hook=type(hook).__name__,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            ctx.logger.log(
                LogComponent.EVAL,
                "INFO",
                "hook_executed",
                phase="teardown",
                hook=type(hook).__name__,
            )


def _spawn_walltime_watchdog(
    agent: CodeAgent, start: float, max_seconds_per_task: int | None
) -> asyncio.Task[None] | None:
    """A task that aborts ``agent`` once the budget elapses (None = no budget)."""
    if max_seconds_per_task is None:
        return None

    async def _watch() -> None:
        remaining = max_seconds_per_task - (time.monotonic() - start)
        if remaining > 0:
            await asyncio.sleep(remaining)
        with contextlib.suppress(Exception):
            await agent.abort()

    return asyncio.create_task(_watch())


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
    if turns >= limits.max_turns:
        return "loop_limit"
    if limits.max_seconds_per_task is not None and elapsed_s >= limits.max_seconds_per_task:
        return "walltime_limit"
    return None


async def _run_turn(message: str, ctx: _LoopContext) -> _TurnOutcome:
    """Send one message, consume the event stream, dispatch each event."""
    outcome = _TurnOutcome()
    ctx.transcript.write(UserMessage(message))  # record what was sent before the reply
    _save_state(ctx)
    try:
        async for event in ctx.agent.send(message):
            ctx.transcript.write(event)
            await _dispatch_event(event, ctx, outcome)
            _save_state(ctx)  # live: duration + cheat counter update during a long turn
            if outcome.error_category is not None:
                break  # backend error: stop consuming this turn
            if ctx.experiment.exit_requested:
                # ExitTask fired MID-turn: alancode runs its whole loop inside one send(), so the
                # between-sends _decide_stop would not see the exit until the agent ends the turn
                # itself - which it may never do before walltime (an ARC run spun 28min post-exit).
                # abort() ends the send cleanly (backend synthesizes error results, transcript stays
                # valid); the loop's next _decide_stop returns agent_exit.
                await ctx.agent.abort()
                break
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
        outcome.saw_tool_call = True  # any call = progress (feeds the doom-loop breaker)
        _flag_suspicious_call(event, ctx)  # observe-and-log every call (never blocks)
        tool = ctx.tools_by_name.get(event.name)
        if tool is not None:  # a framework tool: the loop owns its execution
            result = await _execute_framework_tool(tool, event, ctx)
            ctx.transcript.write(result)
            await ctx.agent.inject(result.output)
    elif isinstance(event, ToolResult):
        _flag_blocked_result(event, ctx)  # the OS sandbox denied an op (file/network)
    elif isinstance(event, AgentError):
        ctx.logger.log(
            LogComponent.AGENT,
            "ERROR",
            "agent_error",
            error_category=event.category,
            message=event.message,
        )
        outcome.error_category = event.category


def _flag_suspicious_call(call: ToolCall, ctx: _LoopContext) -> None:
    """Keyword camera: flag a call whose arguments reach for a forbidden path/module.

    Precise intent detection (the on-disk game data, escape modules); pairs with
    :func:`_flag_blocked_result`, which catches egress the keyword list cannot enumerate.
    Never blocks — it only records a forensic count + WARNING for the analyst.
    """
    flags = flag_tool_call(call.name, call.input, ctx.policy)
    if not flags:
        return
    ctx.experiment.flagged_tool_calls += len(flags)
    ctx.logger.log(
        LogComponent.AGENT,
        "WARNING",
        "flagged_tool_call",
        tool=call.name,
        flags=flags,
    )


def _flag_blocked_result(result: ToolResult, ctx: _LoopContext) -> None:
    """Egress camera: count an errored result where the sandbox/proxy blocked an external host.

    A blocked curl (DNS failure) or the egress proxy's 403 is real evidence the agent tried
    to leave its box for the internet — no need to guess intent from the command, and it
    covers hosts the keyword list cannot enumerate. Never blocks.
    """
    if not result.is_error or not flag_os_denial(result.output):
        return
    ctx.experiment.flagged_tool_calls += 1
    ctx.logger.log(LogComponent.AGENT, "WARNING", "flagged_tool_call", reason="egress_denied")


async def _execute_framework_tool(tool: Tool, call: ToolCall, ctx: _LoopContext) -> ToolResult:
    """Run one framework tool and normalize its result (controlled failures stay results).

    Execution logging lives on the tool itself (``LoggingTool``), shared with the
    HTTP control channel and backend-executed dispatch paths.
    """
    output = await tool.call(call.input, ToolContext(cwd=ctx.cwd))
    return ToolResult(id=call.id, output=str(output.data), is_error=output.is_error)
