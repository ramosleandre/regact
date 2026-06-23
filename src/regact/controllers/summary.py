"""The value returned by a controller rollout.

Returned by ``run_controller(env, ctrl, ...)``. Controller-specific (hence the
``Controller`` prefix and the ``controllers/`` home), not the agnostic agent loop.
Renders a compact digest; the full per-step trace is written to disk and
referenced by path, so inspection works whether or not the process persists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from regact.envclient.obs import Obs


@dataclass
class MilestoneEvent:
    """An env-defined milestone (the problem's milestone_detector writes the text)."""

    step: int  # absolute step index since the rollout began
    description: str  # e.g. "levels_completed: 0->1"


@dataclass
class ControllerRun:
    """One controller's execution segment.

    ``events`` is a heterogeneous ordered list whose elements are each a
    ``MilestoneEvent`` or a nested ``ControllerRun``. Nesting + ``stopped_with``
    are exercised only once hierarchical / sub-controllers land (deferred); for
    v1 a run has a flat list of milestones.
    """

    name: str
    events: list[MilestoneEvent | ControllerRun] = field(default_factory=list)
    stopped_with: str | None = None


@dataclass
class ControllerSummary:
    """Outcome of a controller rollout."""

    stop_kind: str  # "env_done" | "max_steps" | "controller_stop"
    stop_reason: str
    total_steps: int
    history: ControllerRun
    final_obs: Obs
    trace_path: str | None = None  # path to the full per-step trace on disk, or None
    frames: list[dict[str, Any]] = field(default_factory=list)  # per-step obs (JSON), for video

    @property
    def milestones(self) -> list[MilestoneEvent]:
        """Flat list of milestones extracted from ``history`` (depth-first)."""
        out: list[MilestoneEvent] = []

        def walk(run: ControllerRun) -> None:
            for event in run.events:
                if isinstance(event, MilestoneEvent):
                    out.append(event)
                else:
                    walk(event)

        walk(self.history)
        return out

    def __repr__(self) -> str:
        return (
            f"ControllerSummary(stop={self.stop_kind}, total_steps={self.total_steps}, "
            f"milestones={len(self.milestones)})"
        )
