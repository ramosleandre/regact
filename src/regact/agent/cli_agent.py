"""Shared driver for CLI code agents (Claude Code, codex).

Both spawn a CLI subprocess per turn in the workdir, stream its newline-delimited
JSON stdout, and resume across turns by a session/thread id. This base owns that
loop; a subclass supplies three things: the argv to run, how to map one JSON line
to ``AgentEvent``s, and how to capture the session id for resume.

Subprocesses are launched with ``asyncio.create_subprocess_exec`` (argv list, no
shell) — never a shell string — so there is no command-injection surface.

Framework tools (submit/exit) are NOT passed natively to these agents — their
``control_actions`` is ``client_cli``, so the agent reaches them via a workdir CLI
(the control channel, wired separately). Here we only spawn + normalize the stream.
"""

from __future__ import annotations

import asyncio
import json
import os
from abc import abstractmethod
from collections.abc import AsyncIterator

from regact.agent.base import CodeAgent
from regact.agent.events import AgentError, AgentEvent
from regact.obs.errors import ErrorCategory
from regact.tools.base import Tool


class _CliAgent(CodeAgent):
    """Base for subprocess CLI agents; subclasses override the three hooks below."""

    def __init__(self) -> None:
        self._cwd: str = ""
        self._model: str | None = None
        self._system_prompt: str | None = None
        self._env_overrides: dict[str, str] = {}
        self._session_id: str | None = None
        self._pending: list[str] = []  # messages queued by inject(), prepended next turn
        self._proc: asyncio.subprocess.Process | None = None

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
        # CLI agents default to their own auth (e.g. the Claude subscription);
        # base_url/api_key are only forwarded by a subclass that needs them.
        self._cwd = cwd
        self._model = model
        self._system_prompt = system_prompt
        self._env_overrides = dict(env or {})

    async def send(self, message: str) -> AsyncIterator[AgentEvent]:
        if self._pending:
            message = "\n\n".join([*self._pending, message])
            self._pending.clear()

        argv, stdin_data = self._command(message)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self._cwd or None,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,  # inherit: surfaces CLI errors to our logs without buffer deadlock
            env={**os.environ, **self._env_overrides},
        )
        self._proc = proc
        if stdin_data is not None and proc.stdin is not None:
            proc.stdin.write(stdin_data.encode())
            proc.stdin.close()

        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # CLIs may interleave non-JSON log lines
            if not isinstance(obj, dict):
                continue
            self._track_session(obj)
            for event in self._parse_events(obj):
                yield event

        await proc.wait()
        if proc.returncode:
            yield AgentError(
                ErrorCategory.AGENT_API,
                f"{type(self).__name__} CLI exited with code {proc.returncode}",
            )
        self._proc = None

    async def inject(self, message: str) -> None:
        self._pending.append(message)

    async def abort(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.kill()

    async def close(self) -> None:
        await self.abort()
        self._proc = None

    # --- subclass hooks ---------------------------------------------------- #
    @abstractmethod
    def _command(self, message: str) -> tuple[list[str], str | None]:
        """The argv to spawn and optional stdin payload (``None`` = no stdin)."""
        ...

    @abstractmethod
    def _parse_events(self, obj: dict[str, object]) -> list[AgentEvent]:
        """Map one decoded stdout JSON object to zero or more normalized events."""
        ...

    def _track_session(self, obj: dict[str, object]) -> None:
        """Capture the session/thread id for resume. Default: no-op."""
