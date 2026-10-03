"""Bounded JSON RPC to a worker with immutable code and no database/network mounts."""

from __future__ import annotations

import contextlib
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from regact.protocols.cwm import limits as budgets
from regact.protocols.cwm.config import ExecutionConfig
from regact.protocols.cwm.store import canonical
from regact.security.runtime import SandboxRuntime, make_wrapper, resolve

_KILL_WAIT_SECONDS = 10


class WorkerError(RuntimeError):
    def __init__(self, message: str, *, kind: str = "code_error") -> None:
        super().__init__(message)
        self.kind = kind
        self.context: dict[str, Any] = {}
        self.evidence: dict[str, Any] | None = None


class Worker:
    def __init__(
        self,
        bundle: Path,
        config: ExecutionConfig,
        *,
        deadline: float,
        runtime: SandboxRuntime = SandboxRuntime.AUTO,
        deny_read: list[str] | None = None,
        task_deadline: Callable[[], float] | None = None,
        budget_key: str = "protocol.execution.max_seconds_per_RunController",
        budget_seconds: float | None = None,
        load_model: bool = True,
        clock: budgets.AgentClock | None = None,
    ) -> None:
        # The subprocess changes cwd to private scratch; bundle paths must survive that.
        bundle = bundle.resolve()
        self.config = config
        self.budget_key = budget_key
        self.budget_seconds = budget_seconds
        self.deadline = deadline
        self.task_deadline = task_deadline or (lambda: float("inf"))
        self.clock = clock
        self.clock_bound = False  # the next limit is the clock's, not max_seconds_per_call
        self.closed = False
        self.seq = 0
        self.buffer = b""
        self.scratch = tempfile.TemporaryDirectory(prefix="regact-cwm-worker-")
        backend = resolve(runtime)
        if backend is SandboxRuntime.NONE:
            self.scratch.cleanup()
            raise WorkerError(
                "CWM execution requires a usable OS sandbox", kind="isolation_unavailable"
            )
        entry = Path(__file__).with_name("worker_entry.py")
        wrap = make_wrapper(
            backend,
            workdir=self.scratch.name,
            allow_read=[str(bundle), str(entry)],
            deny_egress=True,
            deny_read=deny_read or [],
        )
        argv = wrap(
            [
                sys.executable,
                "-I",
                "-B",
                str(entry),
                str(bundle),
                str(config.max_memory_mb),
                str(int(load_model)),
            ]
        )
        if backend is SandboxRuntime.SEATBELT:
            # Generic agent profile allows host localhost; workers must not.
            argv[2] += "(deny network*)"
        env = {
            key: os.environ[key]
            for key in ("PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH")
            if key in os.environ
        }
        env.update(
            HOME=self.scratch.name,
            TMPDIR="/tmp" if backend is SandboxRuntime.BWRAP else self.scratch.name,
            PYTHONDONTWRITEBYTECODE="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd=self.scratch.name,
            start_new_session=True,
            close_fds=True,
        )
        assert self.proc.stdin is not None and self.proc.stdout is not None
        os.set_blocking(self.proc.stdin.fileno(), False)
        os.set_blocking(self.proc.stdout.fileno(), False)
        self.selector = selectors.DefaultSelector()
        assert self.proc.stdout is not None
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)
        try:
            spawned = time.monotonic()
            reply = self._receive(self._next_limit())
            if not reply.get("ready"):
                raise WorkerError(str(reply.get("error", "worker initialization failed")))
            self._charge(reply, spawned)
        except BaseException:
            self.close()
            raise

    def _charge(self, reply: dict[str, Any] | None, started: float) -> None:
        """Charge the clock the submitted-code seconds the worker reports. Submitted code shares
        that process and could misreport, so the report is clamped to the wall time observed; a
        callback with no reply (killed at its limit) is charged its whole wall time."""
        if self.clock is None:
            return
        elapsed = time.monotonic() - started
        reported = (reply or {}).get("seconds")
        if isinstance(reported, (int, float)) and not isinstance(reported, bool):
            elapsed = min(max(float(reported), 0.0), elapsed)
        self.clock.used += elapsed

    def _next_limit(self) -> float:
        """Wall limit of the next callback: max_seconds_per_call, or what the clock has left."""
        seconds = budgets.deadline(self.config.max_seconds_per_call, start=0.0)
        left = self.clock.left() if self.clock is not None else float("inf")
        self.clock_bound = left < seconds
        return time.monotonic() + min(seconds, left)

    def _operation_timeout(self) -> WorkerError:
        error = WorkerError(
            f"Operation stopped at its deadline ({self.budget_key}={self.budget_seconds:g} seconds; an earlier task deadline or interruption can also stop it)."
            if self.budget_seconds is not None
            else "Operation stopped at its deadline.",
            kind="operation_timeout",
        )
        if self.budget_seconds is not None:
            error.context["budget"] = {
                "parameter": self.budget_key,
                "value": self.budget_seconds,
                "unit": "seconds",
            }
        return error

    def _remaining(self, limit: float) -> float:
        now = time.monotonic()
        if now >= min(self.deadline, self.task_deadline()):
            raise self._operation_timeout()
        if now >= limit and self.clock_bound:
            raise self._operation_timeout()
        if now >= limit:
            error = WorkerError(
                f"Callback exceeded protocol.execution.max_seconds_per_call={budgets.describe(self.config.max_seconds_per_call)} seconds.",
                kind="code_timeout",
            )
            error.context["budget"] = {
                "parameter": "protocol.execution.max_seconds_per_call",
                "value": self.config.max_seconds_per_call,
                "unit": "seconds per callback",
            }
            raise error
        return min(limit, self.deadline, self.task_deadline()) - now

    def _receive(self, limit: float) -> dict[str, Any]:
        assert self.proc.stdout is not None
        while b"\n" not in self.buffer:
            if not self.selector.select(min(0.1, self._remaining(limit))):
                continue
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise WorkerError(
                    f"Execution of submitted code stopped before returning a result. The process may have terminated explicitly, crashed, or exceeded a resource limit; the cause is not established. Configured memory limit: protocol.execution.max_memory_mb={budgets.describe(self.config.max_memory_mb)} MiB.",
                    kind="worker_exit",
                )
            self.buffer += chunk
            if len(self.buffer) > 16 * 1024 * 1024:
                raise WorkerError("worker response exceeds 16 MiB", kind="response_limit")
        line, self.buffer = self.buffer.split(b"\n", 1)
        reply = json.loads(line)
        if not isinstance(reply, dict):
            raise WorkerError("invalid worker response")
        return reply

    def _send(self, data: bytes, limit: float) -> None:
        assert self.proc.stdin is not None
        with selectors.DefaultSelector() as ready:
            ready.register(self.proc.stdin, selectors.EVENT_WRITE)
            view = memoryview(data)
            while view:
                if ready.select(min(0.1, self._remaining(limit))):
                    try:
                        sent = os.write(self.proc.stdin.fileno(), view)
                    except BlockingIOError:
                        continue
                    view = view[sent:]

    def call(self, op: str, **kwargs: Any) -> Any:
        try:
            return self._call(op, **kwargs)
        except WorkerError as exc:
            exc.context.setdefault("callback", op)
            raise

    def _call(self, op: str, **kwargs: Any) -> Any:
        limit = self._next_limit()
        self._remaining(limit)
        self.seq += 1
        request = (canonical({"id": self.seq, "op": op, **kwargs}) + "\n").encode()
        if len(request) > 16 * 1024 * 1024:
            raise WorkerError("worker input exceeds 16 MiB", kind="input_limit")
        assert self.proc.stdin is not None
        started = time.monotonic()
        reply = None
        try:
            self._send(request, limit)
            reply = self._receive(limit)
            if reply.get("id") != self.seq:
                raise WorkerError("worker response ID mismatch")
            if "error" in reply:
                raise WorkerError(str(reply["error"]))
            return reply["result"]
        except (BrokenPipeError, OSError, ValueError, KeyError) as exc:
            raise WorkerError(f"worker transport error: {exc}") from exc
        finally:
            self._charge(reply, started)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        started = time.monotonic()
        try:
            self._shutdown()
        finally:
            self.close_seconds = time.monotonic() - started

    def _shutdown(self) -> None:
        if hasattr(self, "proc"):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGKILL)
            # A process stuck in uninterruptible I/O (shared storage) ignores SIGKILL until the
            # I/O returns; waiting on it unbounded held the coordinator for minutes.
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=_KILL_WAIT_SECONDS)
            for stream in (self.proc.stdin, self.proc.stdout):
                if stream:
                    stream.close()
        if hasattr(self, "selector"):
            self.selector.close()
        self.scratch.cleanup()

    def __enter__(self) -> Worker:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
