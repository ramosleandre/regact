"""Alan Code adapter.

The only module that imports ``alancode``. Wraps ``AlanCodeAgent`` and translates
its ``query_events_async`` stream into the normalized event union. Framework
actions are exposed as native in-process tools; the native session lives under
``<cwd>/.alan/``. The import is deferred to :meth:`start` so merely constructing
the adapter (and declaring its capabilities) never requires ``alancode``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Callable
from typing import Any

from regact.agent.base import CodeAgent
from regact.agent.capabilities import Capabilities
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
from regact.tools.base import Tool


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


def _to_alan_tools(tools: list[Tool]) -> list[Any]:
    """Wrap regact ``Tool`` objects as alancode tools so alancode can schema-ize and run them.

    alancode owns execution of its tools (``to_schema()`` + ``call()``); each wrapper exposes
    the regact tool's name/description/input_schema and delegates ``call`` to it. Deferred
    import (alancode is optional). The loop must therefore NOT re-run them (see ``executes_tools``).
    """
    from alancode.tools.base import Tool as AlanTool
    from alancode.tools.base import ToolResult as AlanToolResult
    from alancode.tools.base import ToolUseContext

    from regact.tools.base import ToolContext

    class _Wrapped(AlanTool):  # type: ignore[misc]
        def __init__(self, tool: Tool) -> None:
            self._tool = tool

        @property
        def name(self) -> str:
            return self._tool.name

        @property
        def description(self) -> str:
            return self._tool.description

        @property
        def input_schema(self) -> dict[str, Any]:
            return self._tool.input_schema

        async def call(self, args: dict[str, Any], context: ToolUseContext) -> Any:
            out = await self._tool.call(args, ToolContext(cwd=context.cwd))
            return AlanToolResult(data=str(out.data), is_error=out.is_error)

    return [_Wrapped(t) for t in tools]


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

    Shared by the in-process adapter and the subprocess runner so the backend is
    configured identically either way. Deferred import: alancode is optional.
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


class AlanAgent(CodeAgent):
    """``CodeAgent`` backed by an in-process ``AlanCodeAgent``."""

    def __init__(self, args: dict[str, Any] | None = None) -> None:
        self._args = dict(args or {})  # alancode tuning: permission_mode, max_output_tokens, ...
        self._agent: Any = None  # set in start(): an alancode.AlanCodeAgent
        self._tools: list[Tool] = []

    async def start(
        self,
        *,
        cwd: str,
        model: str | None,
        base_url: str | None,
        api_key: str | None,
        system_prompt: str | None,
        tools: list[Tool] | None = None,
        env: dict[str, str] | None = None,
        runtime_wrap: Callable[[list[str]], list[str]] | None = None,
    ) -> None:
        self._tools = list(tools) if tools is not None else []
        self._agent = build_alan_agent(
            cwd=cwd,
            model=model,
            base_url=base_url,
            api_key=api_key,
            system_prompt=system_prompt,
            extra_tools=_to_alan_tools(self._tools),  # in-process: alancode runs them itself
            args=self._args,
        )

    async def send(self, message: str) -> AsyncIterator[AgentEvent]:
        async for native in self._agent.query_events_async(message):
            for event in self._map_all(native):
                yield event

    async def inject(self, message: str) -> None:
        self._agent.inject_message(message)

    async def abort(self) -> None:
        await self._agent.abort()

    async def close(self) -> None:
        if self._agent is not None:
            await self._agent.close()
            self._agent = None

    def session_id(self) -> str | None:
        return getattr(self._agent, "_session_id", None)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            system_prompt="replace",
            control_actions="native_tools",
            permission_hooks=True,
            streams_tool_calls=True,
            supports_inject=True,
            writes_native_transcript=True,
            executes_tools=True,  # alancode runs the framework tools; the loop only observes
        )

    def host_read_paths(self) -> list[str]:
        # Alan is in-process (no subprocess sandbox wraps it), so this is not applied
        # today; its native session lives under <workdir>/.alan. Declared for symmetry —
        # add Alan's host config dirs here if it is ever run wrapped (e.g. on JZ/Adastra).
        return []

    def host_egress_hosts(self) -> list[str]:
        # Alan reaches its model via the configured base_url (e.g. a local server), not a
        # fixed external host, so there is no static egress host to allowlist.
        return []

    @classmethod
    def _map_all(cls, native: Any) -> list[AgentEvent]:
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
        :meth:`_map` for any single-block/legacy item so older alancode builds still
        work. Dispatch is by class name to avoid importing alancode's types here.
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
        one = cls._map(native)
        return [one] if one is not None else []

    @staticmethod
    def _map(native: Any) -> AgentEvent | None:
        """Translate one single ``alancode`` block/message into the normalized union.

        Handles legacy per-block streams and terminal messages; unknown items map to
        ``None`` and are dropped. Whole ``AssistantMessage`` objects are unpacked by
        :meth:`_map_all`, not here.
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


def map_alan_events(native: Any) -> list[AgentEvent]:
    """Translate one ``alancode`` stream item into zero or more normalized events.

    The public entry point for that mapping: the in-process adapter uses it through
    :meth:`AlanAgent._map_all`, and the subprocess runner (which owns its own alancode
    instance) calls it directly, so both produce the identical event stream.
    """
    return AlanAgent._map_all(native)
