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

from regact.obs.errors import ErrorCategory, LogComponent, LogRecord


class RunLogger:
    """Per-task structured + human logger."""

    def __init__(self, logs_dir: str, *, task: str) -> None:
        self._task = task
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
