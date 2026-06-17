"""Graceful-shutdown signalling.

A tiny flag the loop polls so SIGINT/SIGTERM stops the session at the next safe
point instead of tearing it down mid-write. Kept transport-agnostic: the loop
only reads ``is_set()``, so unit tests drive the same path without real signals.
"""

from __future__ import annotations

import signal
from collections.abc import Generator
from contextlib import contextmanager


class StopSignal:
    """A one-way latch: cleared at start, set on the first interrupt."""

    def __init__(self) -> None:
        self._set = False

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


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
