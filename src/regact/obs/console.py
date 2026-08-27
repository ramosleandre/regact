"""Terminal reporter: concise, task-prefixed milestone lines for the whole run.

Distinct from :class:`~regact.obs.logger.RunLogger` (the per-task full detail that
lands in ``output.log``): the console is the cross-task operator view. One line per
milestone, prefixed with the task so parallel workers stay legible on a shared
stdout. Third-party INFO chatter (httpx logs one line per env HTTP call, uvicorn its
startup) is silenced here so the terminal shows only regact milestones.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import threading
from typing import TextIO

# INFO-spammy: httpx/uvicorn log one line per HTTP request; arc_agi/arcengine log
# "Found latest version of <game>: ..." on every env load/reset. Raising them to WARNING is
# enough (a real env-side ERROR - e.g. a malformed action - still comes through).
_INFO_NOISY = (
    "httpx",
    "httpcore",
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "arc_agi",
    "arcengine",
)
# WARNING-spammy: imageio_ffmpeg warns (at WARNING) once per encoded video about the harmless
# macro_block_size resize of MiniGrid's 120px frames, so it needs ERROR to go quiet.
_WARNING_NOISY = ("imageio", "imageio_ffmpeg")

_lock = threading.Lock()
_run_log: TextIO | None = None  # the run-level tee; every console line also lands here


def configure_console_logging(run_log_path: str | None = None) -> None:
    """Silence noisy third-party logs; optionally tee the console to a run-level log file.

    Idempotent; called once per experiment. ``run_log_path`` (``<run>/run.log``) makes the
    terminal narration also persist under the run folder, so a finished or in-flight run is
    reviewable from disk. Reconfiguring closes any previous tee (tests reuse the process).
    """
    for name in _INFO_NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in _WARNING_NOISY:
        logging.getLogger(name).setLevel(logging.ERROR)
    global _run_log
    if _run_log is not None:
        _run_log.close()
        _run_log = None
    if run_log_path is not None:
        os.makedirs(os.path.dirname(run_log_path) or ".", exist_ok=True)
        _run_log = open(run_log_path, "a", encoding="utf-8")  # noqa: SIM115 - lives for the run


def console(message: str, *, task: str | None = None) -> None:
    """Write one milestone line to stdout (and the run.log tee), task-prefixed and flushed.

    Locked because the async workers share one event loop but the offloaded eval and
    the env-server thread may also reach stdout; the lock keeps each line whole.
    """
    prefix = f"[{task}] " if task else "[experiment] "
    line = f"{datetime.datetime.now().strftime('%H:%M:%S')} {prefix}{message}\n"
    with _lock:
        sys.stdout.write(line)
        sys.stdout.flush()
        if _run_log is not None:
            _run_log.write(line)
            _run_log.flush()
