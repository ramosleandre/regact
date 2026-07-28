"""Alan Code, run out-of-process so the OS sandbox applies to it.

The in-process :class:`~regact.agent.alan_adapter.AlanAgent` shares the
orchestrator's process, so ``runtime_wrap`` — an argv transformation — cannot
confine it: it keeps the orchestrator's filesystem and network authority, and the
game engine is in its address space. This backend closes that gap by driving the
same ``alancode`` agent from a child process (``regact.agent.alan_runner``) whose
argv IS wrapped, so Alan gets the same confinement as the Claude/codex CLIs.

One long-lived child, not one per turn: alancode keeps its session in memory, so
the process persists for the run and turns are multiplexed over its stdin/stdout
(newline-delimited JSON, the transcript's event shape). Framework tools reach the
agent over the workdir control CLI (``control_actions == "client_cli"``), the same
generic channel the other subprocess agents use — nothing about it is Alan-specific.

The child is launched from an argv list (never a shell string), so there is no
command-injection surface — the same rule the other subprocess adapters follow.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
from collections.abc import AsyncIterator, Callable
from typing import Any

from regact.agent.alan_runner import FATAL, READY, TURN_END
from regact.agent.base import CodeAgent
from regact.agent.capabilities import Capabilities
from regact.agent.events import AgentError, AgentEvent
from regact.obs.errors import ErrorCategory
from regact.obs.transcript import event_from_json
from regact.tools.base import Tool

_STDOUT_LINE_LIMIT = 64 * 1024 * 1024


class AlanSubprocessAgent(CodeAgent):
    """``CodeAgent`` backed by an ``alancode`` agent in a sandboxable child process."""

    def __init__(self, args: dict[str, Any] | None = None) -> None:
        self._args = dict(args or {})  # alancode tuning, forwarded verbatim to the child
        self._proc: asyncio.subprocess.Process | None = None
        self._pending: list[str] = []  # queued by inject(), prepended to the next turn

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
        """Spawn the runner (sandboxed when ``runtime_wrap`` is set) and configure it.

        ``tools`` is ignored on purpose: a subprocess agent calls framework tools over
        the control CLI, so they are never handed to the child as Python objects.
        """
        argv = [sys.executable, "-m", "regact.agent.alan_runner"]
        if runtime_wrap is not None:
            argv = runtime_wrap(argv)  # the whole child runs inside the OS sandbox
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd or None,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            env={**os.environ, **(env or {})},
            limit=_STDOUT_LINE_LIMIT,
            start_new_session=True,
        )
        self._send_command(
            {
                "cmd": "start",
                "cwd": cwd,
                "model": model,
                "base_url": base_url,
                "api_key": api_key,  # over stdin, so it never lands in the host process list
                "system_prompt": system_prompt,
                "args": self._args,
            }
        )
        await self._await_ready()

    async def send(self, message: str) -> AsyncIterator[AgentEvent]:
        """Run one turn in the child; yield its normalized events."""
        if self._pending:
            message = "\n\n".join([*self._pending, message])
            self._pending.clear()
        if self._proc is None or self._proc.stdout is None:
            yield AgentError(ErrorCategory.AGENT_API, "alan runner is not started")
            return

        self._send_command({"cmd": "send", "message": message})
        async for frame in self._read_frames():
            if frame.get("type") == TURN_END:
                return
            event = self._to_event(frame)
            if event is not None:
                yield event
        yield AgentError(ErrorCategory.AGENT_API, self._exit_message())

    async def inject(self, message: str) -> None:
        """Queue a message; it is prepended to the next turn (mirrors the CLI agents)."""
        self._pending.append(message)

    async def abort(self) -> None:
        """Kill the child's process group; the loop's walltime watchdog calls this."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            proc.kill()

    async def close(self) -> None:
        """Ask the child to exit, then make sure it is gone."""
        if self._proc is not None and self._proc.returncode is None:
            with contextlib.suppress(OSError):
                self._send_command({"cmd": "close"})
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
        await self.abort()
        self._proc = None

    def capabilities(self) -> Capabilities:
        return Capabilities(
            system_prompt="replace",  # alancode takes a custom_system_prompt
            control_actions="client_cli",  # tools over the workdir CLI, not native objects
            permission_hooks=False,  # alancode's hooks are not reachable across the boundary
            streams_tool_calls=True,
            supports_inject=True,  # queued, delivered on the next turn
            writes_native_transcript=True,  # <workdir>/.alan
            executes_tools=False,  # framework tools run behind the control channel
        )

    def launch_probe_argv(self) -> list[str]:
        """The child's real prerequisite: importing ``alancode`` inside the sandbox.

        There is no CLI binary here — the runner is this interpreter — so what must be
        verified is that the backend library is reachable from the confined child.
        """
        return [sys.executable, "-c", "import alancode"]

    def host_read_paths(self) -> list[str]:
        """Where ``alancode`` actually lives, so the confined child can import it.

        An editable install (``pip install -e ../alancode``) leaves the package OUTSIDE
        the interpreter prefix — the venv holds only a link — so binding the venv is not
        enough and the child dies with ``ModuleNotFoundError``. Resolved with
        ``find_spec`` (no import), and empty when alancode is absent.
        """
        import importlib.util

        try:
            spec = importlib.util.find_spec("alancode")
        except (ImportError, ValueError):
            return []
        if spec is None:
            return []
        if spec.submodule_search_locations:
            # The package dir's parent, so `import alancode` resolves on sys.path.
            return [os.path.realpath(os.path.dirname(p)) for p in spec.submodule_search_locations]
        return [os.path.realpath(os.path.dirname(spec.origin))] if spec.origin else []

    def host_egress_hosts(self) -> list[str]:
        # The model is reached via the configured base_url (typically a local server),
        # so there is no fixed external host to allow-list.
        return []

    # --- internals --------------------------------------------------------- #
    def _send_command(self, command: dict[str, Any]) -> None:
        """Write one command frame to the child's stdin."""
        if self._proc is None or self._proc.stdin is None:
            return
        self._proc.stdin.write((json.dumps(command) + "\n").encode())

    async def _read_frames(self) -> AsyncIterator[dict[str, Any]]:
        """Yield decoded JSON frames from the child's stdout, skipping any noise."""
        assert self._proc is not None and self._proc.stdout is not None
        async for raw in self._proc.stdout:
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue  # the backend may interleave plain log lines
            if isinstance(frame, dict):
                yield frame

    async def _await_ready(self) -> None:
        """Block until the child confirms it built the agent, or report why it could not."""
        async for frame in self._read_frames():
            kind = frame.get("type")
            if kind == READY:
                return
            if kind == FATAL:
                raise RuntimeError(f"alan runner failed to start: {frame.get('message')}")
        raise RuntimeError(f"alan runner exited before becoming ready: {self._exit_message()}")

    @staticmethod
    def _to_event(frame: dict[str, Any]) -> AgentEvent | None:
        """Map one child frame to an event (a ``_fatal`` control frame becomes an error)."""
        if frame.get("type") == FATAL:
            return AgentError(ErrorCategory.AGENT_API, str(frame.get("message", "runner fault")))
        return event_from_json(frame)

    def _exit_message(self) -> str:
        code = self._proc.returncode if self._proc is not None else None
        return f"alan runner exited with code {code}"
