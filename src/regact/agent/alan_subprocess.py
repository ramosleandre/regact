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
agent over the workdir control CLI (``tool_protocol == "bash_block"``), the same
generic channel the other subprocess agents use — nothing about it is Alan-specific.

The child is launched from an argv list (never a shell string), so there is no
command-injection surface — the same rule the other subprocess adapters follow.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import os
import signal
import sys
from collections.abc import AsyncIterator, Callable
from typing import Any
from urllib.parse import urlparse

from regact.agent.alan_runner import FATAL, READY, TURN_END
from regact.agent.base import CodeAgent
from regact.agent.capabilities import TOOL_PROTOCOLS, Capabilities
from regact.agent.events import AgentError, AgentEvent
from regact.obs.errors import ErrorCategory
from regact.obs.transcript import event_from_json
from regact.tools.base import Tool

_STDOUT_LINE_LIMIT = 64 * 1024 * 1024
# Child-stderr tail kept in memory for the crash report (lines / chars of the joined tail).
_STDERR_TAIL_LINES = 40
_STDERR_TAIL_CHARS = 4000
_REAP_TIMEOUT_S = 3.0  # how long to wait for the exit status after stdout EOF
_DRAIN_TIMEOUT_S = 10.0  # bound on consuming an abandoned turn's leftover frames
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _alancode_paths() -> list[str]:
    """The dir(s) holding the ``alancode`` package this process would import, via ``find_spec``
    (no import): an editable or PYTHONPATH install lives outside the interpreter prefix."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("alancode")
    except (ImportError, ValueError):
        return []
    if spec is None:
        return []
    if spec.submodule_search_locations:  # the package dir's parent, so `import alancode` resolves
        return [os.path.realpath(os.path.dirname(p)) for p in spec.submodule_search_locations]
    return [os.path.realpath(os.path.dirname(spec.origin))] if spec.origin else []


def _runner_regact_paths() -> list[str]:
    """The regact source the in-sandbox alan runner imports (its full closure, incl. the lazy
    ``regact.agent.alan_adapter``): the regact/regact.agent/regact.obs package markers plus the
    runner + adapter + events + errors + transcript modules. Bound file-by-file so the sibling
    adapters (regact.agent.claude_adapter, ...) and regact.obs.result stay out of the namespace."""
    import regact

    src = os.path.dirname(os.path.dirname(os.path.abspath(regact.__file__)))
    rel = (
        ("agent", "__init__.py"),
        ("agent", "alan_runner.py"),
        ("agent", "alan_adapter.py"),
        ("agent", "events.py"),
        ("obs", "__init__.py"),
        ("obs", "errors.py"),
        ("obs", "transcript.py"),
    )
    return [os.path.join(src, "regact", *parts) for parts in rel]


class AlanSubprocessAgent(CodeAgent):
    """``CodeAgent`` backed by an ``alancode`` agent in a sandboxable child process."""

    def __init__(
        self,
        args: dict[str, Any] | None = None,
        *,
        base_url: str | None = None,
        vision: bool = False,
    ) -> None:
        self._args = dict(args or {})  # alancode tuning, forwarded verbatim to the child
        if vision:
            # The image tool is a second tool: the model must be able to name the tool it calls.
            if self._args.get("tool_protocol", "bash_block") == "bash_block":
                raise ValueError(
                    "agent.vision=true with agent=alan needs a tool format that names the tool "
                    "(agent.args.tool_protocol: native, hermes_xml, glm, ...); bash_block has "
                    "only the shell"
                )
            self._args["vision"] = True
        self._display_prompt: str | None = None
        self._session_id: str | None = None  # alancode's session, reported by the child
        self._base_url = base_url
        self._proc: asyncio.subprocess.Process | None = None
        self._pending: list[str] = []  # queued by inject(), prepended to the next turn
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=_STDERR_TAIL_LINES)
        self._stderr_task: asyncio.Task[None] | None = None
        self._at_tool_result = False  # child is waiting for post-tool notices
        self._needs_drain = False  # a prior turn's stream was abandoned before its _turn_end
        self._aborted = False  # we killed the child (e.g. walltime): its exit is not an error
        self._model_info: dict[str, Any] | None = None  # resolved window/source, from _turn_end

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
        child_env = {**os.environ, **(env or {})}
        # The child imports alancode from where THIS process found it (a pinned checkout on the
        # parent's PYTHONPATH, not only the venv's install), the same dirs host_read_paths binds.
        child_env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [child_env.get("PYTHONPATH", ""), *_alancode_paths()])
        )
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd or None,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,  # drained below; its tail feeds the crash report
            env=child_env,
            limit=_STDOUT_LINE_LIMIT,
            start_new_session=True,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())
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

    def prompt_for_transcript(self, prepared: str) -> str:
        if self._display_prompt is None:
            return (
                "[Alan Code system prompt unavailable: child did not report its assembled prompt]"
            )
        return self._display_prompt

    async def send(self, message: str) -> AsyncIterator[AgentEvent]:
        """Run one turn in the child; yield its normalized events."""
        if self._pending:
            message = "\n\n".join([*self._pending, message])
            self._pending.clear()
        if self._proc is None or self._proc.stdout is None:
            yield AgentError(ErrorCategory.AGENT_API, "alan runner is not started")
            return

        if self._needs_drain:
            await self._drain_stale_turn()
        self._send_command({"cmd": "send", "message": message})
        self._needs_drain = True
        async for frame in self._read_frames():
            if frame.get("type") == TURN_END:
                if "context_window" in frame:  # the child resolved it this turn; keep the latest
                    self._model_info = {
                        k: frame[k]
                        for k in ("context_window", "context_window_source")
                        if k in frame
                    }
                self._needs_drain = False
                return
            event = self._to_event(frame)
            if event is not None:
                self._at_tool_result = bool(frame.get("_await_continue"))
                try:
                    yield event
                finally:
                    # Only acknowledge if the stream is resumed normally. Abort/close
                    # owns cleanup if the consumer stops at this result boundary.
                    self._at_tool_result = False
                if frame.get("_await_continue"):
                    self._send_command({"cmd": "continue"})
        if not self._aborted:
            yield AgentError(ErrorCategory.AGENT_API, await self._exit_message())

    async def inject(self, message: str) -> None:
        """Deliver at a paused tool result; otherwise prepend to the next turn."""
        if self._at_tool_result:
            self._send_command({"cmd": "inject", "message": message})
        else:
            self._pending.append(message)

    def resolved_model_info(self) -> dict[str, Any] | None:
        """alancode's resolved context window + source, captured from the child's ``_turn_end``
        (available after the first turn's probe). ``None`` until then."""
        return self._model_info

    async def abort(self) -> None:
        """Kill the child's process group; the loop's walltime watchdog calls this."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        self._aborted = True
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            proc.kill()

    def session_id(self) -> str | None:
        return self._session_id

    def resume_token(self) -> dict[str, Any] | None:
        # alancode keeps the conversation under <workdir>/.alan/sessions/<id>, so the id is all
        # a later process needs, and there is nothing to keep or delete at close.
        return None if self._session_id is None else {"session_id": self._session_id}

    def resume_from(self, token: dict[str, Any]) -> None:
        self._args = {**self._args, "session_id": token["session_id"]}

    async def close(self) -> None:
        """Ask the child to exit, then make sure it is gone."""
        if self._proc is not None and self._proc.returncode is None:
            with contextlib.suppress(OSError):
                self._send_command({"cmd": "close"})
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
        await self.abort()
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None
        self._proc = None

    def capabilities(self) -> Capabilities:
        # Prompt dialect, config-driven: default bash_block, override per served model via
        # ``agent.args.tool_protocol`` (must match alancode's tool_call_format parser). See
        # ToolProtocol for the dialects.
        protocol = self._args.get("tool_protocol", "bash_block")
        if protocol not in TOOL_PROTOCOLS:
            raise ValueError(
                f"unknown agent.args.tool_protocol={protocol!r}; expected one of {TOOL_PROTOCOLS}"
            )
        return Capabilities(
            system_prompt="replace",  # alancode takes a custom_system_prompt
            tool_protocol=protocol,
            permission_hooks=False,  # alancode's hooks are not reachable across the boundary
            streams_tool_calls=True,
            supports_inject=True,  # synchronized post-tool delivery; otherwise next turn
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
        """Where the confined child's imports live: ``alancode`` itself, plus the regact modules the
        runner pulls in.

        This backend is unique: the child is ``python -m regact.agent.alan_runner`` (which lazily
        imports ``regact.agent.alan_adapter``), so its regact closure must be inside the sandbox.
        The agent wrapper's default bind is only ``regact.envclient`` + ``netbridge`` (task.py), so
        the runner's closure is added here; scoring (``regact.controllers``) and the sandbox policy
        (``regact.security.runtime``) stay unbound. alancode via ``find_spec`` (no import): an
        editable install (``pip install -e ../alancode``) leaves it OUTSIDE the interpreter prefix,
        so binding the venv is not enough and the child dies with ``ModuleNotFoundError``.
        """
        return _alancode_paths() + _runner_regact_paths()

    def host_egress_hosts(self) -> list[str]:
        # A loopback base_url needs no egress: its port is bridged into the sandbox
        # by the LoopbackMirror instead.
        host = urlparse(self._base_url).hostname if self._base_url else None
        return [host] if host and host not in _LOOPBACK_HOSTS else []

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
                prompt = frame.get("system_prompt")
                self._display_prompt = prompt if isinstance(prompt, str) else None
                session = frame.get("session_id")
                self._session_id = session if isinstance(session, str) else None
                endpoint = frame.get("remote_endpoint")
                if isinstance(endpoint, str):
                    from regact.obs.console import console

                    console(f"Alan remote endpoint: {endpoint}")
                return
            if kind == FATAL:
                raise RuntimeError(f"alan runner failed to start: {frame.get('message')}")
        raise RuntimeError(
            f"alan runner exited before becoming ready: {await self._exit_message()}"
        )

    @staticmethod
    def _to_event(frame: dict[str, Any]) -> AgentEvent | None:
        """Map one child frame to an event (a ``_fatal`` control frame becomes an error)."""
        if frame.get("type") == FATAL:
            return AgentError(ErrorCategory.AGENT_API, str(frame.get("message", "runner fault")))
        return event_from_json({k: v for k, v in frame.items() if k != "_await_continue"})

    async def _drain_stale_turn(self) -> None:
        """Consume frames left over when a turn's consumer stopped early (the loop breaks
        on an error event), so the next turn does not read a stale ``_turn_end`` as its own."""
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(_DRAIN_TIMEOUT_S):
                async for frame in self._read_frames():
                    if frame.get("type") == TURN_END:
                        break
        self._needs_drain = False

    async def _drain_stderr(self) -> None:
        """Keep the child's stderr pipe from filling, holding the last lines for diagnostics."""
        assert self._proc is not None and self._proc.stderr is not None
        async for raw in self._proc.stderr:
            self._stderr_tail.append(raw.decode(errors="replace").rstrip())

    async def _exit_message(self) -> str:
        """Describe how the child ended: its reaped exit code plus the stderr tail.

        ``returncode`` is None until the child is actually waited on, so reading it
        right after the stdout EOF would always print ``None`` - reap it (briefly)
        first. A timeout means stdout closed while the process lives on.
        """
        proc = self._proc
        code: int | None = None
        if proc is not None:
            with contextlib.suppress(TimeoutError):
                code = await asyncio.wait_for(proc.wait(), timeout=_REAP_TIMEOUT_S)
        if self._stderr_task is not None:  # let the drain reach EOF so the tail is complete
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(self._stderr_task), timeout=1.0)
        detail = (
            "alan runner closed stdout but is still alive"
            if proc is not None and code is None
            else f"alan runner exited with code {code}"
        )
        tail = "\n".join(self._stderr_tail)[-_STDERR_TAIL_CHARS:]
        return f"{detail}; stderr tail:\n{tail}" if tail else detail
