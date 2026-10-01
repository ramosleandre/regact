"""Graceful-shutdown signalling.

A small latch the loop polls so SIGINT/SIGTERM stops the session at the next safe
point instead of tearing it down mid-write. Kept transport-agnostic: the loop
only reads ``is_set()``, so unit tests drive the same path without real signals.
"""

from __future__ import annotations

import signal
from collections.abc import Generator
from contextlib import contextmanager


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
