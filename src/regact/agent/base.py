"""The agnostic code-agent interface.

Any code agent (Alan in-process, Claude Code CLI, future CLIs) is wrapped as a
``CodeAgent`` and driven only through this ABC. Adapters inherit it, so a missing
method fails at construction rather than silently. ``build_agent`` is the registry
that maps an :class:`AgentName` to its adapter, importing each backend lazily so
this module never pulls a backend SDK.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable

from regact.agent.capabilities import Capabilities
from regact.agent.events import AgentEvent
from regact.config.schema import AgentConfig, AgentName
from regact.tools.base import Tool


def executable_paths(binary: str) -> list[str]:
    """The dirs a sandbox must expose so ``binary`` can be exec'd inside it.

    Where PATH finds it (often a symlink), where the real file lives — installers
    keep versioned trees elsewhere (e.g. ``~/.local/share/<cli>/versions``) — plus
    every intermediate hop of the symlink chain, so ``execvp`` can traverse aliased
    install paths. Empty when the binary is absent, so declaring it on a host
    without the CLI binds nothing.
    """
    from regact.security.runtime import symlink_chain_dirs

    found = shutil.which(binary)
    if found is None:
        return []
    real = os.path.realpath(found)
    return sorted({os.path.dirname(found), os.path.dirname(real), *symlink_chain_dirs(found)})


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

    def session_id(self) -> str | None:
        """The backend's native session/thread id, once known (``None`` until then).

        Locates the backend's own transcript store (e.g. ``.claude``/``.alan`` sessions)
        from the canonical experiment state, so audits can correlate the two. Backends
        without a discoverable id keep the default.
        """
        return None

    def host_read_paths(self) -> list[str]:
        """Host paths THIS backend needs readable inside the sandbox (install dirs, caches).

        Each backend declares its own, so a deny-by-default sandbox allowlist contains
        only the *loaded* agent's paths, never another backend's. Never a user-level
        config/session store: those hold ambient data (other sessions' transcripts) a
        scored agent must not see — give the CLI an isolated home via
        :meth:`host_rw_paths` instead. Agnostic by construction: returns plain paths, so
        the security layer never imports a backend type. In-process backends
        (scripted/Alan) aren't wrapped, so the default is none.
        """
        return []

    def host_rw_paths(self) -> list[str]:
        """Host paths THIS backend must read AND write inside the sandbox.

        Separate from :meth:`host_read_paths` because read-only backends bind those
        immutably — a CLI's isolated config home (where it writes its own sessions)
        must stay writable. Implementations create the path before returning it, so a
        bind/subpath rule can name it. Default: none.
        """
        return []

    def host_egress_hosts(self) -> list[str]:
        """External hosts THIS backend must reach (for an egress-allowlist proxy).

        Per-backend like :meth:`host_read_paths` (Claude: ``api.anthropic.com``; codex:
        ``api.openai.com`` / ``auth.openai.com`` / ``chatgpt.com``; alan: the
        ``base_url`` host when external). Empty when the model is reached over loopback
        (a local server, bridged into the sandbox rather than allow-listed). Plain
        strings, so the security/proxy layer stays agnostic.
        """
        return []

    def auth_check(self) -> tuple[str, str] | None:
        """Cheaply verify the backend can actually AUTHENTICATE, not just launch.

        ``launch_probe_argv`` (a ``--version``) proves the binary runs; it does NOT prove
        the run will start — a CLI can be present yet unauthenticated (the classic
        "Not logged in" that only surfaces once a real turn fires). Return ``(status,
        detail)`` with ``status`` in ``{"ok","warn"}``; ``None`` (default) means the
        backend has no cheap auth check to offer. Must be side-effect-free and fast.
        """
        return None

    def launch_probe_argv(self) -> list[str]:
        """A cheap argv proving THIS backend can actually start (e.g. ``<cli> --version``).

        Declarative like :meth:`host_read_paths`: the diagnostic wraps this argv in the
        run's sandbox and reports whether it executes, which catches a backend whose
        executable is absent from the allowlist — a failure the conformance probe cannot
        see, since it only ever wraps the interpreter (always allowed). Empty for an
        in-process backend, which launches nothing.
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
        from regact.agent.alan_subprocess import AlanSubprocessAgent

        return AlanSubprocessAgent(config.args, base_url=config.base_url)
    if config.name is AgentName.CLAUDE:
        from regact.agent.claude_adapter import ClaudeAgent

        return ClaudeAgent(config.args)
    if config.name is AgentName.CODEX:
        from regact.agent.codex_adapter import CodexAgent

        return CodexAgent(config.args)
    raise ValueError(f"unknown agent {config.name!r}")
