"""Alan Code adapter.

The only module that imports ``alancode``. Wraps ``AlanCodeAgent`` and translates
its ``query_events_async`` stream into the normalized event union. Framework
actions are exposed as native in-process tools; the native session lives under
``<cwd>/.alan/``. The import is deferred to :meth:`start` so merely constructing
the adapter (and declaring its capabilities) never requires ``alancode``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
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


class AlanAgent(CodeAgent):
    """``CodeAgent`` backed by an in-process ``AlanCodeAgent``."""

    def __init__(self) -> None:
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
    ) -> None:
        from alancode import AlanCodeAgent

        self._tools = list(tools) if tools is not None else []
        self._agent = AlanCodeAgent(
            model=model,
            base_url=base_url,
            api_key=api_key,
            cwd=cwd,
            programmatic=True,
            custom_system_prompt=system_prompt,
            extra_tools=self._tools,
        )

    async def send(self, message: str) -> AsyncIterator[AgentEvent]:
        async for native in self._agent.query_events_async(message):
            event = self._map(native)
            if event is not None:
                yield event

    async def inject(self, message: str) -> None:
        self._agent.inject_message(message)

    async def abort(self) -> None:
        await self._agent.abort()

    async def close(self) -> None:
        if self._agent is not None:
            await self._agent.close()
            self._agent = None

    def capabilities(self) -> Capabilities:
        return Capabilities(
            system_prompt="replace",
            control_actions="native_tools",
            permission_hooks=True,
            streams_tool_calls=True,
            supports_inject=True,
            writes_native_transcript=True,
        )

    @staticmethod
    def _map(native: Any) -> AgentEvent | None:
        """Translate one ``alancode`` stream event into the normalized union.

        Dispatches on the native event's class name so we don't import alancode's
        block types here; unknown events map to ``None`` and are dropped.
        """
        kind = type(native).__name__
        if kind == "TextBlock":
            return TextDelta(text=getattr(native, "text", ""))
        if kind == "ThinkingBlock":
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
                output=str(getattr(native, "content", "")),
                is_error=bool(getattr(native, "is_error", False)),
            )
        if kind in ("ResultMessage", "TurnComplete"):
            return TurnComplete(
                final_text=getattr(native, "result", getattr(native, "final_text", "")) or "",
                usage=getattr(native, "usage", None),
            )
        if kind in ("ErrorMessage", "APIError"):
            return AgentError(
                category=ErrorCategory.AGENT_API,
                message=str(getattr(native, "message", native)),
            )
        return None
