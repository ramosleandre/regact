"""Structured + human logging writer.

Emits ``LogRecord`` lines to ``events.jsonl`` (machine-readable, component- and
error-tagged) alongside a readable ``output.log``. The ``component`` axis is the
same one the error taxonomy uses, so logs filter by origin. Distinct from the
transcript: this is the framework's own operational log, not the agent stream.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import IO

from regact.obs.console import console as write_console
from regact.obs.errors import ErrorCategory, LogComponent, LogRecord


class RunLogger:
    """Per-task structured + human logger.

    Every record lands in ``events.jsonl`` (machine) and ``output.log`` (human). When
    ``console=True`` a milestone subset also goes to stdout, task-prefixed, so the
    terminal shows a concise cross-task view without the full per-task detail.
    """

    def __init__(self, logs_dir: str, *, task: str, console: bool = False) -> None:
        self._task = task
        self._console = console
        # The logger owns these handles for its lifetime; close() / __exit__ release them.
        self._events: IO[str] = open(f"{logs_dir}/events.jsonl", "w", encoding="utf-8")  # noqa: SIM115
        self._human: IO[str] = open(f"{logs_dir}/output.log", "w", encoding="utf-8")  # noqa: SIM115

    def log(
        self,
        component: LogComponent,
        level: str,
        event: str,
        *,
        phase: str | None = None,
        error_category: ErrorCategory | None = None,
        **detail: object,
    ) -> None:
        """Append one ``LogRecord`` to ``events.jsonl`` and mirror to ``output.log``."""
        self.emit(
            LogRecord(
                ts=datetime.now(UTC).isoformat(),
                component=component,
                level=level,
                event=event,
                task=self._task,
                phase=phase,
                error_category=error_category,
                detail=dict(detail),
            )
        )

    def emit(self, record: LogRecord) -> None:
        self._events.write(json.dumps(record.to_json()) + "\n")
        self._events.flush()
        self._human.write(_human_line(record) + "\n")
        self._human.flush()
        if self._console:
            message = _console_message(record)
            if message is not None:
                write_console(message, task=self._task)

    def close(self) -> None:
        self._events.close()
        self._human.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _human_line(record: LogRecord) -> str:
    tag = f"[{record.level}] {record.component.value}: {record.event}"
    if record.error_category is not None:
        tag += f" ({record.error_category.value})"
    return f"{record.ts} {tag}" + (f" {record.detail}" if record.detail else "")


def _console_message(record: LogRecord) -> str | None:
    """The terminal milestone line for ``record``, or None to keep it off the terminal.

    Every WARNING/ERROR surfaces; among INFO only a few lifecycle milestones do, so the
    terminal stays a concise progress view while ``output.log`` keeps the full detail.
    """
    if record.level in ("WARNING", "ERROR"):
        extra = f" {record.detail}" if record.detail else ""
        return f"{record.level}: {record.event}{extra}"
    event, detail = record.event, record.detail
    if event == "session_start":
        return "started"
    if event == "tool_executed":
        tool = detail.get("tool", "?")
        return f"{tool}" + (" (error)" if detail.get("is_error") else "")
    if event == "hook_start":
        # The re-score is the one teardown step worth announcing (it used to look frozen).
        return "final evaluation..." if detail.get("hook") == "FinalizeControllerHook" else None
    if event == "session_end":
        return f"done: {detail.get('reason', '?')}"
    return None
