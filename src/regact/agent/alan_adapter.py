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

    extra: dict[str, Any] = {}
    if args.get("context_window") is not None:
        extra["context_window"] = int(args["context_window"])  # env interp can yield a str
    return AlanCodeAgent(
        backend=args.get("backend"),  # e.g. "scripted" (+ model=remote → HTTP-driven provider)
        model=model,
        base_url=base_url,
        api_key=api_key,
        cwd=cwd,
        programmatic=True,
        custom_system_prompt=system_prompt,
        extra_tools=extra_tools,
        permission_mode=args.get("permission_mode"),
        max_iterations_per_turn=args.get("max_iterations_per_turn"),
        max_output_tokens=args.get("max_output_tokens"),
        memory=args.get("memory"),
        tool_call_format=args.get("tool_call_format"),
        **extra,
    )


def map_alan_events(native: Any) -> list[AgentEvent]:
    """Translate one ``alancode`` stream item into zero or more normalized events.

    ``query_events_async`` yields whole messages, NOT individual blocks:

    - streaming display deltas: ``AssistantMessage`` with ``hide_in_api=True``,
      re-carried verbatim by the assembled message — dropped here so text and
      thinking appear exactly once;
    - assembled ``AssistantMessage`` (a ``.content`` list of TextBlock/ThinkingBlock/
      ToolUseBlock) — unpacked into its events; when it carries no tool call the
      query loop ends there, so a ``TurnComplete`` is derived from it;
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
        has_tool_call = False
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
                has_tool_call = True
                events.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        input=getattr(block, "input", {}) or {},
                    )
                )
        if not has_tool_call:
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
