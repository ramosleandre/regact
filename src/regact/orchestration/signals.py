"""Graceful-shutdown signalling.

A small latch the loop polls so SIGINT/SIGTERM stops the session at the next safe
point instead of tearing it down mid-write. Kept transport-agnostic: the loop
only reads ``is_set()``, so unit tests drive the same path without real signals.
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Generator
from contextlib import contextmanager

STOP_FILE = "STOP"  # in a run directory: stop every task; in a task directory: that task
_FORCE = "force"  # the file's content when in-flight tools must not be waited for
_FILE_CHECK_S = 1.0


class StopSignal:
    """First interrupt requests a safe stop; subsequent interrupts request force."""

    def __init__(self) -> None:
        self._requests = 0

    def set(self) -> None:
        self._requests += 1

    def is_set(self) -> bool:
        return self._requests > 0

    def force_requested(self) -> bool:
        """A second interrupt explicitly abandons waiting for in-flight tools."""
        return self._requests > 1


class FileStop(StopSignal):
    """A task's stop latch: the launch's own signal, or a ``STOP`` file someone wrote in the run
    or task directory (``python -m regact.stop``)."""

    def __init__(self, parent: StopSignal, directories: list[str]) -> None:
        super().__init__()
        self._parent = parent
        self._paths = [os.path.join(directory, STOP_FILE) for directory in directories]
        self._checked = 0.0

    def _poll(self) -> None:
        if self._requests > 1 or time.monotonic() - self._checked < _FILE_CHECK_S:
            return
        self._checked = time.monotonic()
        for path in self._paths:
            try:
                with open(path, encoding="utf-8") as handle:
                    self._requests = max(self._requests, 2 if handle.read().strip() == _FORCE else 1)
            except OSError:
                continue

    def is_set(self) -> bool:
        self._poll()
        return self._requests > 0 or self._parent.is_set()

    def force_requested(self) -> bool:
        self._poll()
        return self._requests > 1 or self._parent.force_requested()


def request_stop(directory: str, *, force: bool = False) -> str:
    """Write the stop file in a run or task directory; returns its path."""
    path = os.path.join(directory, STOP_FILE)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_FORCE if force else "stop")
    return path


def clear_stop(directory: str) -> None:
    """Remove a stop request that a new launch must not inherit."""
    try:
        os.remove(os.path.join(directory, STOP_FILE))
    except FileNotFoundError:
        pass


@contextmanager
def install_stop_signal(
    *, signals: tuple[int, ...] = (signal.SIGINT, signal.SIGTERM)
) -> Generator[StopSignal, None, None]:
    """Install handlers that latch a :class:`StopSignal`, restoring them on exit."""
    stop = StopSignal()
    previous = {sig: signal.getsignal(sig) for sig in signals}

    def _handler(signum: int, frame: object) -> None:
        stop.set()

    for sig in signals:
        signal.signal(sig, _handler)
    try:
        yield stop
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
