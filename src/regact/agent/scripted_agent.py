"""Deterministic ``CodeAgent`` test double.

Replays a prepared list of turns (each turn = a list of :class:`AgentEvent`) with
no LLM and no subprocess. Each ``send`` consumes the next scripted turn; injected
messages and the call log are recorded so tests can assert on what the loop did.
Paired with ``FakeNativeEnv`` it makes the whole stack runnable in CI.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from regact.agent.base import CodeAgent
from regact.agent.capabilities import Capabilities
from regact.agent.events import AgentEvent, TurnComplete
from regact.tools.base import Tool


class ScriptedAgent(CodeAgent):
    """A ``CodeAgent`` that emits a fixed sequence of events per turn."""

    def __init__(self, turns: list[list[AgentEvent]] | None = None) -> None:
        self._turns = list(turns) if turns is not None else []
        self._turn = 0
        self.started = False
        self.closed = False
        self.aborted = False
        self.cwd: str | None = None
        self.tools: list[Tool] = []
        self.sent: list[str] = []
        self.injected: list[str] = []

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
        self.started = True
        self.cwd = cwd
        self.tools = list(tools) if tools is not None else []  # in-process: runtime_wrap N/A

    async def send(self, message: str) -> AsyncIterator[AgentEvent]:
        self.sent.append(message)
        events = self._turns[self._turn] if self._turn < len(self._turns) else [TurnComplete()]
        self._turn += 1
        for event in events:
            yield event

    async def inject(self, message: str) -> None:
        self.injected.append(message)

    async def abort(self) -> None:
        self.aborted = True

    async def close(self) -> None:
        self.closed = True

    def capabilities(self) -> Capabilities:
        return Capabilities(
            system_prompt="replace",
            control_actions="native_tools",
            permission_hooks=False,
            streams_tool_calls=True,
            supports_inject=True,
            writes_native_transcript=False,
        )
