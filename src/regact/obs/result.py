"""The eval result schema.

Serialized to ``eval/submissions/<n>/results.json``. Kept problem-agnostic via a
union-of-metric-keys design, and consumed verbatim by the visualizer. Ported in
shape from GameAgents ``eval_runner._write_results`` (dicts there, typed here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from regact.obs.errors import ErrorCategory


@dataclass
class EpisodeResult:
    """One evaluation episode."""

    episode: int
    stop_kind: str | None = None
    stop_reason: str | None = None
    milestones: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)  # success/steps/reward/levels
    error: str | None = None
    error_category: ErrorCategory | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "stop_kind": self.stop_kind,
            "stop_reason": self.stop_reason,
            "milestones": self.milestones,
            "metrics": self.metrics,
            "error": self.error,
            "error_category": self.error_category.value if self.error_category else None,
        }


@dataclass
class EvalResult:
    """Aggregate + per-episode results for one submission."""

    task: str
    aggregate: dict[str, Any] = field(default_factory=dict)  # n_episodes, success_rate, ...
    episodes: list[EpisodeResult] = field(default_factory=list)
    error: str | None = None
    error_category: ErrorCategory | None = None
    executor: str | None = None  # "subprocess" | "in_process" — disambiguates the schema shape

    def to_json(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "aggregate": self.aggregate,
            "episodes": [e.to_json() for e in self.episodes],
            "error": self.error,
            "error_category": self.error_category.value if self.error_category else None,
            "executor": self.executor,
        }
