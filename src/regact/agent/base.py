"""The agnostic code-agent interface.

Any code agent (Alan in-process, Claude Code CLI, future CLIs) is wrapped as a
``CodeAgent`` and driven only through this ABC. Adapters inherit it, so a missing
method fails at construction rather than silently. ``build_agent`` is the registry
that maps an :class:`AgentName` to its adapter, importing each backend lazily so
this module never pulls a backend SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from regact.agent.capabilities import Capabilities
from regact.agent.events import AgentEvent
from regact.config.schema import AgentConfig, AgentName
from regact.tools.base import Tool


class CodeAgent(ABC):
    """A driveable code agent."""

    @abstractmethod
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
        """Configure and launch the agent against a workdir and LLM endpoint."""
        ...

    @abstractmethod
    def send(self, message: str) -> AsyncIterator[AgentEvent]:
        """Run one turn; stream normalized events."""
        ...

    @abstractmethod
    async def inject(self, message: str) -> None:
        """Queue a user message to be delivered on the next turn."""
        ...

    @abstractmethod
    async def abort(self) -> None:
        """Interrupt the current turn."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release all resources (subprocess, sockets, sessions)."""
        ...

    @abstractmethod
    def capabilities(self) -> Capabilities:
        """Static description of what this backend supports."""
        ...


def build_agent(config: AgentConfig) -> CodeAgent:
    """Construct the configured agent; each backend is imported only on its branch."""
    if config.name is AgentName.SCRIPTED:
        from regact.agent.scripted_agent import ScriptedAgent

        return ScriptedAgent()
    if config.name is AgentName.ALAN:
        from regact.agent.alan_adapter import AlanAgent

        return AlanAgent()
    if config.name is AgentName.CLAUDE:
        from regact.agent.claude_adapter import ClaudeAgent

        return ClaudeAgent()
    if config.name is AgentName.CODEX:
        from regact.agent.codex_adapter import CodexAgent

        return CodexAgent()
    raise ValueError(f"unknown agent {config.name!r}")
