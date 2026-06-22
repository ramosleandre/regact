"""Per-task persisted state.

Written early and updated continuously with an atomic save (tmpfile + replace),
so the polling visualizer never reads a half-written file. Field names are
load-bearing for the viewer. The submit/exit tools mutate this object; the loop
saves it. Ported from GameAgents ``session/state.py``.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentState:
    """The live state of one task's run."""

    problem_name: str
    task_name: str
    n_eval_episodes: int
    n_videos: int
    problem_kwargs: dict[str, Any] = field(default_factory=dict)
    submission_count: int = 0
    exit_requested: bool = False
    agent_session_id: str | None = None  # locates the native transcript dir
    last_submission_results: dict[str, Any] | None = None
    last_error_category: str | None = None
    exit_reason: str | None = None  # set at teardown; None while the run is still going
    cheat_attempts: int = 0  # tool calls flagged as reaching for forbidden paths/modules
    win_levels: int | None = None  # total levels to win (from the game's first observation)
    duration_s: float = 0.0  # wall-clock the agent has spent on this task so far
    schema_version: int = 1

    def save(self, path: str) -> None:
        """Atomic write (tmpfile + os.replace) so a poller never sees a partial file."""
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(dataclasses.asdict(self), handle, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> ExperimentState:
        with open(path, encoding="utf-8") as handle:
            return cls(**json.load(handle))
