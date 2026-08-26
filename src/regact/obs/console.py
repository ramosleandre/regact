"""Terminal reporter: concise, task-prefixed milestone lines for the whole run.

Distinct from :class:`~regact.obs.logger.RunLogger` (the per-task full detail that
lands in ``output.log``): the console is the cross-task operator view. One line per
milestone, prefixed with the task so parallel workers stay legible on a shared
stdout. Third-party INFO chatter (httpx logs one line per env HTTP call, uvicorn its
startup) is silenced here so the terminal shows only regact milestones.
"""

from __future__ import annotations

import logging
import sys
import threading

# INFO-spammy: httpx/uvicorn log one line per HTTP request. Raising them to WARNING is enough.
_INFO_NOISY = ("httpx", "httpcore", "uvicorn", "uvicorn.error", "uvicorn.access")
# WARNING-spammy: imageio_ffmpeg warns (at WARNING) once per encoded video about the harmless
# macro_block_size resize of MiniGrid's 120px frames, so it needs ERROR to go quiet.
_WARNING_NOISY = ("imageio", "imageio_ffmpeg")

_lock = threading.Lock()


def configure_console_logging() -> None:
    """Silence noisy third-party logs. Idempotent; called once per experiment."""
    for name in _INFO_NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in _WARNING_NOISY:
        logging.getLogger(name).setLevel(logging.ERROR)


def console(message: str, *, task: str | None = None) -> None:
    """Write one milestone line to stdout, task-prefixed and flushed.

    Locked because the async workers share one event loop but the offloaded eval and
    the env-server thread may also reach stdout; the lock keeps each line whole.
    """
    prefix = f"[{task}] " if task else "[experiment] "
    with _lock:
        sys.stdout.write(f"{prefix}{message}\n")
        sys.stdout.flush()
