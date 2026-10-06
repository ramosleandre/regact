"""The single keep-alive loop.

ONE loop over the normalized ``AgentEvent`` stream, replacing the two divergent
loops in GameAgents. It is provider-independent: it sends a message, consumes the
agent's event stream, executes the *framework* tools it recognizes (submit/exit),
mirrors everything to the canonical ``transcript.jsonl``, and stops on the agent's
request, a limit, an interrupt, a persistent backend error, or a crash.

It is deliberately agnostic of controllers/games/eval: it only knows agents,
framework tools, protocol sessions, hooks, limits, and writers — all generic interfaces. It imports
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
import dataclasses
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

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
from regact.protocols.base import ProtocolSession
from regact.security.detection import flag_os_denial, flag_tool_call
from regact.security.policy import SecurityPolicy, default_policy
from regact.session.state import ExperimentState
from regact.tools.base import Tool, ToolContext

_ABORTED_REASONS = frozenset({"loop_crash"})

# Delivered after completed tools (CLI adapters: next outer turn), subject to flagging_warning_cap,
# so a model reaching for a sandboxed action learns why it failed and stops wasting budget on it.
_FLAGGING_WARNING = (
    "WARNING: a command was flagged by the environment's monitoring as conflicting with "
    "this benchmark's isolation and security rules. Those rules block, among others: installing "
    "packages (pip / uv / apt / npm ...); any internet or outbound network access (only the model "
    "host is reachable); reading or writing files outside your working directory; reading the "
    "framework's own source, or importing/reading the game's code or answer data; spawning "
    "processes, changing permissions, or otherwise trying to escape the sandbox; and dynamic "
    "tricks (exec/eval, ctypes, importlib) to bypass the above.\n"
    "If that's what happened, please stop: (a) it isn't the purpose of this benchmark, and (b) the "
    "environment is strongly sandboxed, so these attempts simply fail and waste your budget. "
)

# A single backend error (one 500/timeout from a slow local server) must not end the
# session; only a wall of them means the backend is really gone.
_MAX_CONSECUTIVE_ERROR_TURNS = 3
# A backend usage limit (a subscription window) is waited out instead of ending the run, when it
# resets within this long. The margin covers a reset that lands slightly late.
_MAX_USAGE_LIMIT_WAIT_S = 6 * 3600
_MAX_USAGE_LIMIT_TOTAL_WAIT_S = 24 * 3600  # over the whole task
_USAGE_LIMIT_MARGIN_S = 60
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
    protocol: ProtocolSession
    state_path: str = ""  # where to persist ExperimentState (saved live, per event)
    start: float = 0.0  # time.monotonic() at the run's start, for the live duration
    move_count: Callable[[], int] | None = None  # polls the env's step count, for the live state
    flagging_warning_cap: int = 0  # max flagging warnings to inject this task (0 = never)
    warnings_injected: int = 0  # warning allowance consumed
    flags_observed: int = 0  # flags already considered at a completed-tool boundary
    pending_warnings: list[str] = field(default_factory=list)
    # Keep bounded excerpts until a parallel group has completed. CLI warnings may
    # be delivered much later, so never describe the command as "your last command".
    warning_calls: dict[str, str] = field(default_factory=dict)
    flagged_call_ids: set[str] = field(default_factory=set)
    attributed_flags: int = 0
    max_tool_calls: int | None = None  # hard tool-call budget, enforced mid-send (None = off)
    stop: StopSignal | None = None
    interrupted: bool = False


@dataclass
class _TurnOutcome:
    """What one turn produced, so the loop can decide whether to continue."""

    saw_tool_call: bool = False  # agent emitted >=1 tool call (framework, bash, or native)
    error_category: ErrorCategory | None = None  # a backend error in the stream
    error_message: str = ""
    crashed: bool = False  # an unexpected exception escaped the turn
    pending_tools: set[str] = field(default_factory=set)  # calls awaiting their matching result


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
    protocol: ProtocolSession,
    stop: StopSignal | None = None,
    move_count: Callable[[], int] | None = None,
    flagging_warning_cap: int = 0,
) -> str:
    """Drive one task to completion; return the exit reason."""
    start = time.monotonic() - experiment.duration_s  # a resumed task keeps its elapsed time
    initial_limits = limits
    budget = limits.seconds_left()
    if budget is not None and experiment.duration_s:  # counted from the task's first start
        cap = limits.max_seconds_per_task
        budget += experiment.duration_s
        budget = budget if cap is None else min(budget, cap)
    limits = dataclasses.replace(limits, max_seconds_per_task=budget)
    protocol.on_start(start, limits.max_seconds_per_task)
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
        flagging_warning_cap=flagging_warning_cap,
        max_tool_calls=limits.max_tool_calls,
        protocol=protocol,
        stop=stop,
    )
    logger.log(LogComponent.ORCHESTRATOR, "INFO", "session_start", phase="bootstrap")
    experiment.save(state_path)
    # Record the inputs so the viewer shows the whole session, not just replies (a resumed task
    # already has them).
    if system_prompt and not experiment.resumed_at:
        transcript.write(SystemPrompt(agent.prompt_for_transcript(system_prompt)))

    message = first_message
    turns = experiment.turn
    error_turns = 0  # consecutive turns that ended in a backend error
    no_tool_turns = 0  # consecutive turns that produced no tool call (doom-loop breaker)
    reminders = 0
    watchdog = _spawn_walltime_watchdog(agent, start, limits.max_seconds_per_task)
    try:
        while True:
            reason = _decide_stop(
                interrupted=stop.is_set() if stop is not None else False,
                turns=turns,
                tool_calls_total=experiment.tool_calls_total,
                elapsed_s=time.monotonic() - start,
                limits=limits,
                protocol_reason=protocol.stop_reason(),
            )
            if reason is not None:
                break

            experiment.turn = turns + 1  # 1-indexed: the turn now starting
            outcome = await _run_turn(message, ctx)
            if ctx.interrupted or (stop is not None and stop.is_set()):
                reason = "interrupted"
                break

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
                wait = _usage_limit_wait(
                    agent.usage_limit_reset(outcome.error_message),
                    limits,
                    experiment.usage_limit_waited_s,
                )
                if wait is not None:
                    if watchdog is not None:
                        watchdog.cancel()
                    protocol.on_start(start, None)  # no deadline while the clock is paused
                    waited = await _sleep_through_usage_limit(wait, ctx)
                    if waited is None:
                        reason = "interrupted"
                        break
                    # The task clock does not run while the backend refuses work.
                    start += waited
                    ctx.start = start
                    limits = _budget_after_pause(limits, start)
                    protocol.on_start(start, limits.max_seconds_per_task)
                    watchdog = _spawn_walltime_watchdog(agent, start, limits.max_seconds_per_task)
                    error_turns = 0
                    turns += 1
                    message = _ERROR_RETRY_MESSAGE
                    continue
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
            if (
                limits.max_consecutive_no_tool_turns is not None
                and 0 < limits.max_consecutive_no_tool_turns <= no_tool_turns
            ):
                reason = "no_tool_progress"
                break
            reminders += 1
            message = protocol.reminder(reminders)
    finally:
        if watchdog is not None:
            watchdog.cancel()

    # Recorded BEFORE teardown, which re-scores the controller and on a slow serve outlasts what
    # is left of an exhausted budget: the verdict is known the moment the loop breaks, and nothing
    # teardown does can change it. A run killed in that window used to keep exit_reason=None and
    # read as "still running" forever - bench-04 job 5418021 was SIGKILLed two minutes into its
    # final re-score, having correctly decided walltime_limit.
    experiment.exit_reason = reason  # "running" until set; the viewer shows it as the status
    if reason == "walltime_limit":
        own = initial_limits.max_seconds_per_task
        spent = time.monotonic() - start
        experiment.exit_detail = (
            "task_budget" if own is not None and spent >= own else "experiment_deadline"
        )
    _save_state(ctx)
    await _run_teardown_hooks(protocol.hooks, reason, ctx)
    protocol.after_teardown(reason, logger)
    experiment.main_metrics = protocol.task_metrics()
    logger.log(
        LogComponent.ORCHESTRATOR,
        "INFO",
        "session_end",
        phase="teardown",
        reason=reason,
        main_metrics=experiment.main_metrics,
    )
    _save_state(ctx)
    return reason


def _usage_limit_wait(
    reset_unix: float | None, limits: LimitsConfig, already_waited: float
) -> float | None:
    """Seconds to sleep until a backend usage limit resets, or None when the error is not such a
    limit, or waiting is refused (too long, in one go or over the task, or past the experiment's
    own deadline)."""
    if reset_unix is None:
        return None
    resume_unix = reset_unix + _USAGE_LIMIT_MARGIN_S
    wait = max(resume_unix - time.time(), 0.0)
    deadline = limits.experiment_deadline_unix
    if (
        wait > _MAX_USAGE_LIMIT_WAIT_S
        or already_waited + wait > _MAX_USAGE_LIMIT_TOTAL_WAIT_S
        or (deadline is not None and resume_unix >= deadline)
    ):
        return None
    return wait


async def _sleep_through_usage_limit(seconds: float, ctx: _LoopContext) -> float | None:
    """Sleep until the limit resets, recording the wait; None when the run was interrupted."""
    ctx.logger.log(
        LogComponent.ORCHESTRATOR, "WARNING", "usage_limit_wait", seconds=round(seconds, 1)
    )
    began = time.monotonic()
    while (left := seconds - (time.monotonic() - began)) > 0:
        if ctx.interrupted or (ctx.stop is not None and ctx.stop.is_set()):
            return None
        await asyncio.sleep(min(left, 5.0))
    waited = time.monotonic() - began
    ctx.experiment.usage_limit_waits += 1
    ctx.experiment.usage_limit_waited_s = round(ctx.experiment.usage_limit_waited_s + waited, 1)
    ctx.logger.log(
        LogComponent.ORCHESTRATOR, "INFO", "usage_limit_resumed", waited_seconds=round(waited, 1)
    )
    return waited


def _budget_after_pause(limits: LimitsConfig, start: float) -> LimitsConfig:
    """The pause left the task's time budget untouched, but the experiment's absolute deadline
    kept running: cut the budget to what that deadline still allows."""
    budget, deadline = limits.max_seconds_per_task, limits.experiment_deadline_unix
    if budget is None or deadline is None:
        return limits
    allowed = int(time.monotonic() - start + max(0.0, deadline - time.time()))
    return dataclasses.replace(limits, max_seconds_per_task=min(budget, allowed))


def _save_state(ctx: _LoopContext) -> None:
    """Persist the run state with the live duration (called per event, so the viewer
    reflects a long single turn — e.g. a codex ``exec`` — as it happens, not only at its end)."""
    ctx.experiment.duration_s = round(time.monotonic() - ctx.start, 1)
    if ctx.move_count is not None:
        ctx.experiment.env_moves = ctx.move_count()
    if ctx.experiment.agent_session_id is None:
        ctx.experiment.agent_session_id = ctx.agent.session_id()
    ctx.experiment.agent_resume = ctx.agent.resume_token() or ctx.experiment.agent_resume
    info = ctx.agent.resolved_model_info()
    resolved = info.get("context_window") if info else None
    if info is not None and resolved is not None:  # reported window wins over the baseline
        ctx.experiment.context_window = resolved
        ctx.experiment.context_window_source = info.get("context_window_source") or "alancode"
    ctx.experiment.save(ctx.state_path)


async def _run_teardown_hooks(hooks: list[Hook], reason: str, ctx: _LoopContext) -> None:
    """Fire TEARDOWN hooks unless the run was aborted; a hook fault never aborts teardown."""
    if reason in _ABORTED_REASONS:
        return
    for hook in hooks:
        if hook.phase is not HookPhase.TEARDOWN:
            continue
        ctx.logger.log(
            LogComponent.EVAL, "INFO", "hook_start", phase="teardown", hook=type(hook).__name__
        )
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
    interrupted: bool,
    turns: int,
    elapsed_s: float,
    limits: LimitsConfig,
    tool_calls_total: int = 0,
    protocol_reason: str | None = None,
) -> str | None:
    """Pure stop decision, checked before each turn. ``None`` means keep going."""
    if interrupted:
        return "interrupted"
    if protocol_reason is not None:
        return protocol_reason
    if limits.max_turns_per_task is not None and turns >= limits.max_turns_per_task:
        return "loop_limit"
    if limits.max_tool_calls is not None and tool_calls_total >= limits.max_tool_calls:
        return "tool_call_limit"
    if limits.max_seconds_per_task is not None and elapsed_s >= limits.max_seconds_per_task:
        return "walltime_limit"
    return None


async def _run_turn(message: str, ctx: _LoopContext) -> _TurnOutcome:
    """Send one message, consume the event stream, dispatch each event."""
    outcome = _TurnOutcome()
    if ctx.pending_warnings:
        message = "\n\n".join([*ctx.pending_warnings, message])
        ctx.pending_warnings.clear()
    ctx.transcript.write(UserMessage(message))  # record the actual next-turn input
    _save_state(ctx)
    consumer = asyncio.current_task()

    async def watch_interrupt() -> None:
        assert ctx.stop is not None and consumer is not None
        announced = False
        while True:
            await asyncio.sleep(0.05)
            if not ctx.stop.is_set():
                continue
            if not announced:
                ctx.logger.log(
                    LogComponent.ORCHESTRATOR,
                    "INFO",
                    "stop_requested",
                    message="Finishing active tools; interrupt again to force stop.",
                )
                announced = True
            if outcome.pending_tools and not ctx.stop.force_requested():
                continue
            ctx.interrupted = True
            try:
                async with asyncio.timeout(5):
                    await ctx.agent.abort()
            except Exception:
                pass
            finally:
                consumer.cancel()
            return

    interrupt_watch = asyncio.create_task(watch_interrupt()) if ctx.stop is not None else None
    try:
        async for event in ctx.agent.send(message):
            if ctx.interrupted:
                break
            ctx.transcript.write(event)
            await _dispatch_event(event, ctx, outcome)
            _save_state(ctx)  # live: duration + cheat counter update during a long turn
            if outcome.error_category is not None:
                break  # backend error: stop consuming this turn
            # A nested HTTP SubmitSolution/ExitTask may update experiment state while
            # its enclosing shell command is still writing files. Likewise, a result
            # from one parallel call does not mean the other calls have finished.
            # Honor cooperative stops only after all observed calls have returned.
            if outcome.pending_tools:
                continue
            budget_reached = (
                ctx.max_tool_calls is not None
                and ctx.experiment.tool_calls_total >= ctx.max_tool_calls
            )
            if (
                budget_reached
                or ctx.protocol.stop_reason() is not None
                or (ctx.stop is not None and ctx.stop.is_set())
            ):
                ctx.interrupted = ctx.stop is not None and ctx.stop.is_set()
                await ctx.agent.abort()
                break
    except asyncio.CancelledError:
        if not ctx.interrupted:
            raise
    except Exception as exc:  # an unexpected fault in a tool or the adapter
        ctx.logger.log(
            LogComponent.LOOP,
            "ERROR",
            "turn_crash",
            error_category=ErrorCategory.LOOP_CRASH,
            error=f"{type(exc).__name__}: {exc}",
        )
        outcome.crashed = True
    finally:
        if interrupt_watch is not None:
            interrupt_watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await interrupt_watch
    return outcome


async def _dispatch_event(event: AgentEvent, ctx: _LoopContext, outcome: _TurnOutcome) -> None:
    """Route one event: execute framework tools, record backend errors, else observe."""
    if isinstance(event, ToolCall):
        outcome.pending_tools.add(event.id)
        outcome.saw_tool_call = True  # any call = progress (feeds the doom-loop breaker)
        ctx.experiment.tool_calls_total += 1
        await _flag_suspicious_call(event, ctx)  # observe-and-log every call (never blocks)
        tool = ctx.tools_by_name.get(event.name)
        if tool is not None:  # a framework tool: the loop owns its execution
            result, notices = await _execute_framework_tool(tool, event, ctx)
            ctx.transcript.write(result)
            await ctx.agent.inject(result.output)
            for notice in notices:
                ctx.transcript.write(UserMessage(notice))
                await ctx.agent.inject(notice)
            outcome.pending_tools.discard(event.id)
    elif isinstance(event, ToolResult):
        outcome.pending_tools.discard(event.id)
        if not event.executed:  # never ran (e.g. cut at the output cap): refund its budget
            ctx.experiment.tool_calls_total -= 1
        await _flag_blocked_result(event, ctx)  # the OS sandbox denied an op (file/network)
    elif isinstance(event, AgentError):
        ctx.logger.log(
            LogComponent.AGENT,
            "ERROR",
            "agent_error",
            error_category=event.category,
            message=event.message,
        )
        outcome.error_category = event.category
        outcome.error_message = event.message
    # Include flags raised by trusted HTTP handlers during the tool, even if the
    # shell caught the denial and returned success. Wait for parallel calls too.
    if isinstance(event, (ToolCall, ToolResult)) and not outcome.pending_tools:
        await _maybe_warn_flagged(ctx)


async def _maybe_warn_flagged(ctx: _LoopContext) -> None:
    """Deliver once at a completed-tool boundary, subject to the per-task message cap.

    CLI adapters queue for the next send; Alan can inject before its next model request.
    A terminal boundary has no subsequent request, so do not claim message delivery.
    """
    flags = ctx.experiment.flagged_tool_calls
    new_flags = flags - ctx.flags_observed
    commands = _flagged_command_context(ctx, unattributed=new_flags > ctx.attributed_flags)
    ctx.warning_calls.clear()
    ctx.flagged_call_ids.clear()
    ctx.attributed_flags = 0
    if new_flags <= 0:
        return
    ctx.flags_observed = flags
    if (
        ctx.protocol.stop_reason() is not None
        or (ctx.stop is not None and ctx.stop.is_set())
        or (
            ctx.max_tool_calls is not None and ctx.experiment.tool_calls_total >= ctx.max_tool_calls
        )
    ):
        return
    if ctx.flagging_warning_cap <= 0 or ctx.warnings_injected >= ctx.flagging_warning_cap:
        return
    ctx.warnings_injected += 1
    message = _FLAGGING_WARNING + "\n\n" + commands + "\n\n" + ctx.protocol.interaction_guidance()
    immediate = ctx.agent.capabilities().supports_inject
    ctx.logger.log(
        LogComponent.AGENT,
        "WARNING",
        "flagging_warning",
        delivery="after_tool" if immediate else "next_turn",
        warning_number=ctx.warnings_injected,
    )
    if immediate:
        await ctx.agent.inject(message)
        ctx.transcript.write(UserMessage(message))
    else:
        # CLI adapters cannot accept messages inside their current send().
        # Add to the next send here, so the transcript shows its actual delivery.
        ctx.pending_warnings.append(message)


def _command_excerpt(call: ToolCall) -> str:
    """Keep the original command (or tool arguments), with bounded middle truncation."""
    value = call.input.get("command", call.input.get("cmd"))
    text = (
        value
        if isinstance(value, str)
        else call.name + " " + json.dumps(call.input, ensure_ascii=False)
    )
    if len(text) > 800:
        text = text[:390] + " ... [truncated] ... " + text[-390:]
    return text


def _flagged_command_context(ctx: _LoopContext, *, unattributed: bool) -> str:
    """Quote exact known calls; do not invent attribution for an HTTP-side flag.

    Trusted HTTP handlers share a flag counter, but cannot know which concurrent
    shell request triggered them. One pending call is unambiguous; several calls
    are listed as candidates instead of blaming the last completed clean call.
    """
    known = [value for key, value in ctx.warning_calls.items() if key in ctx.flagged_call_ids]
    uncertain = []
    if unattributed:
        if len(ctx.warning_calls) == 1:
            known = list(ctx.warning_calls.values())
        else:
            uncertain = list(ctx.warning_calls.values())
    lines = [f"Command flagged : '{text}'" for text in known[:5]]
    if len(known) > 5:
        lines.append(f"{len(known) - 5} additional flagged commands omitted.")
    if uncertain:
        lines.append(
            "A flag also occurred during concurrent commands; exact attribution is unavailable. Commands in that group:"
        )
        lines.extend(f"- '{text}'" for text in uncertain[:5])
        if len(uncertain) > 5:
            lines.append(f"{len(uncertain) - 5} additional commands omitted.")
    if not lines:
        lines.append("Command flagged : '<command unavailable in the tool event stream>'")
    return "\n".join(lines)


async def _flag_suspicious_call(call: ToolCall, ctx: _LoopContext) -> None:
    """Keyword camera: flag a call whose arguments reach for a forbidden path/module.

    Precise intent detection (the on-disk game data, escape modules); pairs with
    :func:`_flag_blocked_result`, which catches egress the keyword list cannot enumerate.
    Never blocks — it records a forensic count + WARNING, and (capped) nudges the agent to stop.
    """
    ctx.warning_calls[call.id] = _command_excerpt(call)
    flags = flag_tool_call(call.name, call.input, ctx.policy)
    if not flags:
        return
    ctx.experiment.flagged_tool_calls += len(flags)
    ctx.attributed_flags += len(flags)
    ctx.flagged_call_ids.add(call.id)
    ctx.logger.log(
        LogComponent.AGENT,
        "WARNING",
        "flagged_tool_call",
        tool=call.name,
        flags=flags,
    )


async def _flag_blocked_result(result: ToolResult, ctx: _LoopContext) -> None:
    """Egress camera: count an errored result where the sandbox/proxy blocked an external host.

    A blocked curl (DNS failure) or the egress proxy's 403 is real evidence the agent tried
    to leave its box for the internet — no need to guess intent from the command, and it
    covers hosts the keyword list cannot enumerate. Never blocks; nudges the agent (capped).
    """
    if not result.is_error or not flag_os_denial(result.output):
        return
    ctx.experiment.flagged_tool_calls += 1
    ctx.attributed_flags += 1
    ctx.flagged_call_ids.add(result.id)
    ctx.logger.log(LogComponent.AGENT, "WARNING", "flagged_tool_call", reason="egress_denied")


async def _execute_framework_tool(
    tool: Tool, call: ToolCall, ctx: _LoopContext
) -> tuple[ToolResult, list[str]]:
    """Run one framework tool and normalize its result (controlled failures stay results).

    Execution logging lives on the tool itself (``LoggingTool``), shared with the
    HTTP control channel and backend-executed dispatch paths.
    """
    output = await tool.call(call.input, ToolContext(cwd=ctx.cwd))
    return ToolResult(
        id=call.id, output=str(output.data), is_error=output.is_error
    ), output.messages
