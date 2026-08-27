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

# INFO-spammy: httpx/uvicorn log one line per HTTP request; arc_agi logs "Found latest version of
# <game>: ..." on every env load/reset. arc_agi is the awkward one - `arc_agi.base` and
# `arc_agi.scorecard` RE-RAISE their own level to INFO and RE-ADD a stdout handler on every game
# load (base.py) / at import (scorecard.py), so a one-shot setLevel is overwritten. We name those
# child loggers explicitly and (below) attach a level FILTER, which survives their re-config.
_INFO_NOISY = (
    "httpx",
    "httpcore",
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "arc_agi",
    "arc_agi.base",
    "arc_agi.scorecard",
    "arcengine",
)
# WARNING-spammy: imageio_ffmpeg warns (at WARNING) once per encoded video about the harmless
# macro_block_size resize of MiniGrid's 120px frames, so it needs ERROR to go quiet.
_WARNING_NOISY = ("imageio", "imageio_ffmpeg")

_lock = threading.Lock()
_run_log: TextIO | None = None  # the run-level tee; every console line also lands here


class _MinLevelFilter(logging.Filter):
    """Drop records below ``level``, regardless of the logger's own level or handlers.

    A one-shot ``setLevel(WARNING)`` loses to a third-party lib (arc_agi) that re-raises its logger
    to INFO and re-adds a stdout handler on every game load. A logger-level FILTER is checked in
    ``Logger.handle`` before any handler and is NOT cleared by ``handlers.clear()`` or ``setLevel``,
    so it sticks: once attached to the (singleton) logger, the lib's later re-config can't undo it.
    """

    def __init__(self, level: int) -> None:
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.level


def _silence(name: str, min_level: int) -> None:
    """Raise a noisy logger's floor to ``min_level`` durably: set the level AND attach a filter that
    survives the logger being reconfigured by its owning library (see :class:`_MinLevelFilter`)."""
    lg = logging.getLogger(name)
    lg.setLevel(min_level)
    if not any(isinstance(f, _MinLevelFilter) for f in lg.filters):
        lg.addFilter(_MinLevelFilter(min_level))


def configure_console_logging(run_log_path: str | None = None) -> None:
    """Silence noisy third-party logs; optionally tee the console to a run-level log file.

    Idempotent; called once per experiment. ``run_log_path`` (``<run>/run.log``) makes the
    terminal narration also persist under the run folder, so a finished or in-flight run is
    reviewable from disk. Reconfiguring closes any previous tee (tests reuse the process).
    """
    for name in _INFO_NOISY:
        _silence(name, logging.WARNING)
    for name in _WARNING_NOISY:
        _silence(name, logging.ERROR)
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
