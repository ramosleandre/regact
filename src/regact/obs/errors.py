"""Error and log taxonomy.

A single ``component`` axis tags both log lines and errors, so failures can be
attributed to one part of the system without string-matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class LogComponent(StrEnum):
    """Which part of the system emitted a record."""

    ENV_SERVER = "env_server"
    ORCHESTRATOR = "orchestrator"
    SCHEDULER = "scheduler"
    EVAL = "eval"
    LOOP = "loop"
    AGENT = "agent"


class ErrorCategory(StrEnum):
    """Fine-grained failure class, keyed to where it is raised."""

    AGENT_API = "agent_api"  # LLM endpoint: auth, rate-limit, timeout, context overflow
    AGENT_SOLUTION = "agent_solution"  # agent-authored code: import / get_controller / act() raised
    ENV_RUNTIME = "env_runtime"  # env: step()/reset()/make raised, obs decode failed
    EVAL_HARNESS = "eval_harness"  # orchestration: build failed, task mismatch, no result file
    LOOP_LIMIT = "loop_limit"  # keep-alive cap, max_moves, walltime, token budget
    LOOP_CRASH = "loop_crash"  # uncaught framework exception, killed before init


class RegactError(Exception):
    """Framework error carrying a machine-readable category."""

    def __init__(
        self, category: ErrorCategory, message: str, *, cause: BaseException | None = None
    ) -> None:
        super().__init__(message)
        self._category = category
        if cause is not None:
            self.__cause__ = cause

    @property
    def category(self) -> ErrorCategory:
        return self._category


@dataclass
class LogRecord:
    """One structured log entry. Serialized one-per-line to ``events.jsonl``."""

    ts: str
    component: LogComponent
    level: str  # "DEBUG" | "INFO" | "WARNING" | "ERROR"
    event: str  # short machine-friendly event name
    task: str | None = None
    phase: str | None = None  # "bootstrap" | "explore" | "submit" | "eval" | "teardown"
    error_category: ErrorCategory | None = None
    detail: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "ts": self.ts,
            "component": self.component.value,
            "level": self.level,
            "event": self.event,
            "task": self.task,
            "phase": self.phase,
            "error_category": self.error_category.value if self.error_category else None,
            "detail": self.detail,
        }
