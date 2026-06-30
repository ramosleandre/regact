"""The agnostic code-agent interface.

Any code agent (Alan in-process, Claude Code CLI, future CLIs) is wrapped as a
``CodeAgent`` and driven only through this ABC. Adapters inherit it, so a missing
method fails at construction rather than silently. ``build_agent`` is the registry
that maps an :class:`AgentName` to its adapter, importing each backend lazily so
this module never pulls a backend SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable

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
        runtime_wrap: Callable[[list[str]], list[str]] | None = None,
    ) -> None:
        """Configure and launch the agent against a workdir and LLM endpoint.
        ``runtime_wrap`` (when set) wraps the agent's subprocess argv to run inside
        an OS sandbox.
        """
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

    def host_read_paths(self) -> list[str]:
        """Host config/auth dirs THIS backend needs readable inside the sandbox.

        Each backend declares its own (Claude: ``~/.claude``; codex: ``~/.codex``), so a
        deny-by-default sandbox allowlist contains only the *loaded* agent's paths, never
        another backend's. Agnostic by construction: returns plain paths, so the security
        layer never imports a backend type. In-process backends (scripted/Alan) aren't
        wrapped, so the default is none.
        """
        return []

    def host_egress_hosts(self) -> list[str]:
        """External hosts THIS backend must reach (for an egress-allowlist proxy).

        Per-backend like :meth:`host_read_paths` (Claude: ``api.anthropic.com``; codex:
        ``api.openai.com`` / ``auth.openai.com`` / ``chatgpt.com``). Empty when the model
        is reached via a configured ``base_url`` (e.g. a local server on HPC) rather than
        a fixed host. Plain strings, so the security/proxy layer stays agnostic.
        """
        return []

    def host_write_prefixes(self) -> list[str]:
        """Path PREFIXES under which the backend creates scratch files with RANDOM leaf
        names (so a fixed subpath rule cannot name them; the sandbox allows read+write on
        anything starting with the prefix). E.g. Claude Code's ``/tmp/claude-<rand>-cwd``.
        Plain strings, regex-anchored by the sandbox. Default: none.
        """
        return []


def build_agent(config: AgentConfig) -> CodeAgent:
    """Construct the configured agent; each backend is imported only on its branch."""
    if config.name is AgentName.SCRIPTED:
        from regact.agent.scripted_agent import ScriptedAgent

        return ScriptedAgent()
    if config.name is AgentName.ALAN:
        from regact.agent.alan_adapter import AlanAgent

        return AlanAgent(config.args)
    if config.name is AgentName.CLAUDE:
        from regact.agent.claude_adapter import ClaudeAgent

        return ClaudeAgent(config.args)
    if config.name is AgentName.CODEX:
        from regact.agent.codex_adapter import CodexAgent

        return CodexAgent(config.args)
    raise ValueError(f"unknown agent {config.name!r}")
