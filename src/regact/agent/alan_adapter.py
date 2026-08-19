"""Alan Code glue: construct the ``alancode`` agent and normalize its events.

The only module that imports ``alancode``. The ``alan`` backend runs the agent in
a sandboxable child process (:mod:`regact.agent.alan_subprocess` spawns
:mod:`regact.agent.alan_runner`); the child calls :func:`build_alan_agent` to
configure the backend and :func:`map_alan_events` to translate its
``query_events_async`` stream into the normalized event union. Imports of
``alancode`` are deferred so this module loads without it installed.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from regact.agent.events import (
    AgentError,
    AgentEvent,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnComplete,
)
from regact.obs.errors import ErrorCategory

# Cap a truncation-recovery completion (alancode escalates the output budget to this on a
# length-truncation). alancode's own default is 64000, window-clamped - ~85min on a 12 tok/s
# model; 12000 caps it to ~17min, plenty to finish a cut-off turn. Override via
# ``agent.args.escalated_max_tokens``.
_DEFAULT_ESCALATED_MAX_TOKENS = 12000


def _usage_dict(usage: Any) -> dict[str, Any] | None:
    """Coerce a native usage record into a plain JSON-able dict (``None`` if opaque)."""
    if usage is None or isinstance(usage, dict):
        return usage
    if dataclasses.is_dataclass(usage) and not isinstance(usage, type):
        return dataclasses.asdict(usage)
    return None


def _result_text(content: Any) -> str:
    """Flatten a tool-result payload (plain string or list of text blocks) into a string."""
    if isinstance(content, list):
        return "".join(str(getattr(block, "text", block)) for block in content)
    return str(content)


def _as_bool(value: Any) -> bool:
    """Coerce a config/env value to bool (Hydra/env pass e.g. the string ``"true"``)."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def build_alan_agent(
    *,
    cwd: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    system_prompt: str | None,
    extra_tools: list[Any],
    args: dict[str, Any],
) -> Any:
    """Construct an ``alancode.AlanCodeAgent`` from regact's parameters.

    Called by the subprocess runner (in the child); deferred import, alancode is
    optional.
    """
    from alancode import AlanCodeAgent
    from alancode.tools.registry import find_tool_by_name, get_all_builtin_tools

    # Bash-only agent (Mini-SWE-Agent style): the model is given ONE tool and does all
    # file operations (read/write/edit) through shell commands. ``tools=[...]``
    bash = find_tool_by_name(get_all_builtin_tools(), "Bash")
    tools = [bash] if bash is not None else None

    extra: dict[str, Any] = {}
    if args.get("context_window") is not None:
        extra["context_window"] = int(args["context_window"])  # env interp can yield a str
    agent = AlanCodeAgent(
        backend=args.get("backend"),  # e.g. "scripted" (+ model=remote → HTTP-driven provider)
        model=model,
        base_url=base_url,
        api_key=api_key,
        cwd=cwd,
        programmatic=True,
        custom_system_prompt=system_prompt,
        tools=tools,  # only Bash
        extra_tools=extra_tools,
        permission_mode=args.get("permission_mode"),
        max_iterations_per_turn=args.get("max_iterations_per_turn"),
        max_output_tokens=args.get("max_output_tokens"),
        memory=args.get("memory"),
        tool_call_format=args.get("tool_call_format"),
        **extra,
    )
    # escalated_max_tokens is a SETTINGS key, not a constructor kwarg: an unknown ctor kwarg
    # becomes a backend kwarg and flows to the LLM transport as an API param, silently missing
    # settings. Apply it via the public settings API after construction (queries read live).
    escalated = args.get("escalated_max_tokens")
    value = int(escalated) if escalated is not None else _DEFAULT_ESCALATED_MAX_TOKENS
    err = agent.update_session_setting("escalated_max_tokens", value)
    if err is not None:
        raise RuntimeError(f"alancode rejected escalated_max_tokens={value}: {err}")

    # Optional empty_response-sweep settings: applied ONLY when the bench sets them, so a
    # default run still works on an older alancode. Settings (not ctor kwargs) - an unknown
    # ctor kwarg would silently become an LLM API param (see escalated_max_tokens above); a
    # rejected setting raises loudly, which is right (a requested sweep arm must not run wrong).
    for name, cast in (("empty_response_retries", int), ("persist_thinking", _as_bool)):
        raw = args.get(name)
        if raw is None:
            continue
        err = agent.update_session_setting(name, cast(raw))
        if err is not None:
            raise RuntimeError(f"alancode rejected {name}={raw!r}: {err}")
    return agent


def map_alan_events(native: Any) -> list[AgentEvent]:
    """Translate one ``alancode`` stream item into zero or more normalized events.

    ``query_events_async`` yields whole messages, NOT individual blocks:

    - streaming display deltas: ``AssistantMessage`` with ``hide_in_api=True``,
      re-carried verbatim by the assembled message — dropped here so text and
      thinking appear exactly once;
    - assembled ``AssistantMessage`` (a ``.content`` list of TextBlock/ThinkingBlock/
      ToolUseBlock) - unpacked into its events and closed with a ``TurnComplete`` that
      marks the completion boundary (one completion = one turn) and carries its usage;
    - ``UserMessage`` — tool results come back as a ``.content`` list holding
      ``ToolResultBlock`` items, unpacked into ``ToolResult`` events; plain-string
      user messages are model-facing context and map to nothing.

    A message flagged as an API error becomes an ``AgentError``. Falls back to
    :func:`_map_legacy` for any single-block/legacy item so older alancode builds
    still work. Dispatch is by class name to avoid importing alancode's types here.
    """
    kind = type(native).__name__
    if kind == "AssistantMessage":
        events: list[AgentEvent] = []
        if getattr(native, "is_api_error_message", False):
            detail = getattr(native, "error_details", None) or getattr(native, "api_error", "")
            events.append(
                AgentError(
                    category=ErrorCategory.AGENT_API,
                    message=str(detail or getattr(native, "text", "") or "assistant API error"),
                )
            )
            return events
        if getattr(native, "hide_in_api", False):
            return []
        texts: list[str] = []
        for block in getattr(native, "content", []) or []:
            bkind = type(block).__name__
            if bkind == "TextBlock":
                text = getattr(block, "text", "")
                if text:
                    texts.append(text)
                    events.append(TextDelta(text=text))
            elif bkind in ("ThinkingBlock", "RedactedThinkingBlock"):
                events.append(
                    ThinkingDelta(text=getattr(block, "thinking", getattr(block, "text", "")))
                )
            elif bkind == "ToolUseBlock":
                events.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        input=getattr(block, "input", {}) or {},
                    )
                )
        # One completion = one turn: close every assembled message with a TurnComplete
        # (carrying its own usage), so the viewer groups per completion and per-turn usage
        # is accurate. A bash-only agent calls a tool every turn, so gating this on
        # "no tool call" (the old behaviour) emitted almost none and the viewer collapsed
        # the whole run into a single "turn".
        events.append(
            TurnComplete(
                final_text="".join(texts),
                usage=_usage_dict(getattr(native, "usage", None)),
            )
        )
        return events
    if kind == "UserMessage":
        content = getattr(native, "content", None)
        if not isinstance(content, list):
            return []
        return [
            ToolResult(
                id=getattr(block, "tool_use_id", ""),
                output=_result_text(getattr(block, "content", "")),
                is_error=bool(getattr(block, "is_error", False)),
            )
            for block in content
            if type(block).__name__ == "ToolResultBlock"
        ]
    one = _map_legacy(native)
    return [one] if one is not None else []


def _map_legacy(native: Any) -> AgentEvent | None:
    """Translate one single ``alancode`` block/message into the normalized union.

    Handles legacy per-block streams and terminal messages; unknown items map to
    ``None`` and are dropped. Whole messages are unpacked by :func:`map_alan_events`,
    not here.
    """
    kind = type(native).__name__
    if kind == "TextBlock":
        return TextDelta(text=getattr(native, "text", ""))
    if kind in ("ThinkingBlock", "RedactedThinkingBlock"):
        return ThinkingDelta(text=getattr(native, "thinking", getattr(native, "text", "")))
    if kind == "ToolUseBlock":
        return ToolCall(
            id=getattr(native, "id", ""),
            name=getattr(native, "name", ""),
            input=getattr(native, "input", {}) or {},
        )
    if kind == "ToolResultBlock":
        return ToolResult(
            id=getattr(native, "tool_use_id", getattr(native, "id", "")),
            output=_result_text(getattr(native, "content", "")),
            is_error=bool(getattr(native, "is_error", False)),
        )
    if kind in ("ResultMessage", "TurnComplete"):
        return TurnComplete(
            final_text=getattr(native, "result", getattr(native, "final_text", "")) or "",
            usage=_usage_dict(getattr(native, "usage", None)),
        )
    if kind in ("ErrorMessage", "APIError"):
        return AgentError(
            category=ErrorCategory.AGENT_API,
            message=str(getattr(native, "message", native)),
        )
    return None
