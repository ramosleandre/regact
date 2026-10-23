"""The Replay tab: every real frame of a run and every conversation item, on the viewer's three
x-axes (real environment actions, tool calls, active seconds).

Frames come from the run's experience store (CWM and vanilla protocols); a run without one
replays its conversation only. A real step has no time of its own: it takes the time of the
controller command that played it, spread over that command's tool call, so the time axis
moves through a long controller call instead of jumping over it.
"""

from __future__ import annotations

import bisect
import json
import sqlite3
from pathlib import Path
from typing import Any

from regact.problems.base import BaseProblem
from regact.protocols.cwm import viewer as cwm_viewer
from regact.viz.reader import GameView, ToolCallView

_RUN_EVENTS = ("RunController", "SubmitExplorationController")
_RESET_EVENTS = ("ResetLevel", "ResetEnvironment")
_MODEL_EVENT = "UpdateCodeWorldModel"


def has_viewer_renderer(problem: BaseProblem) -> bool:
    """Whether the problem draws a picture for people that differs from the agent's."""
    return type(problem).render_frame_for_viewer is not BaseProblem.render_frame_for_viewer


def timeline(task: Path, view: GameView, problem: BaseProblem | None) -> dict[str, Any]:
    calls = [
        tool
        for turn in view.turns
        for tool in turn.tools
        if not tool.synthetic and tool.started is not None
    ]
    starts = sorted(tool.started for tool in calls if tool.started is not None)
    clock = cwm_viewer.active_clock(task)
    frames, markers = _store_frames(task, problem, calls) if problem is not None else ([], [])
    frame_times = [f["at"] for f in frames]
    frame_env = [f["env_actions"] for f in frames]

    def env_at(at: float | None) -> int | None:
        if at is None or not frames:
            return None
        k = bisect.bisect_right(frame_times, at)
        return frame_env[k - 1] if k else 0

    def place(point: dict[str, Any], at: float | None) -> None:
        point["tool_calls"] = bisect.bisect_right(starts, at) if at is not None else None
        point["seconds"] = round(clock(at), 2) if at is not None else None

    for frame in frames:
        place(frame, frame["at"])
    for marker in markers:
        place(marker, marker["at"])
        if marker.get("env_actions") is None:
            marker["env_actions"] = env_at(marker["at"])

    items: list[list[Any]] = []  # [turn, item, env actions, tool calls, seconds]
    n_calls, last = 0, None
    for i, turn in enumerate(view.turns):
        for j, item in enumerate(turn.items):
            at = item.tool.started if item.tool is not None else item.at
            if item.tool is not None and not item.tool.synthetic:
                n_calls += 1
            last = at if at is not None and (last is None or at > last) else last
            seconds = round(clock(last), 2) if last is not None else None
            items.append([i, j, env_at(last), n_calls, seconds])
    for k in (2, 4):  # a stamp out of order must not hide an earlier item: keep them nondecreasing
        best = None
        for row in items:
            if row[k] is not None:
                best = row[k] if best is None else max(best, row[k])
                row[k] = best
    for frame in frames:
        del frame["at"]
    for marker in markers:
        del marker["at"]
    views = []
    if frames and problem is not None:
        views = ["viewer", "agent"] if has_viewer_renderer(problem) else ["agent"]
    return {
        "frames": frames,
        "items": items,
        "markers": markers,
        "views": views,
        "axes": [
            axis
            for axis, ok in (
                ("env_actions", bool(frames)),
                ("tool_calls", True),
                ("seconds", any(row[4] is not None for row in items)),
            )
            if ok
        ],
        "progress_label": problem.progress_label if problem is not None else None,
    }


def _store_frames(
    task: Path, problem: BaseProblem, calls: list[ToolCallView]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The real frames in order, and the CWM markers (model updates, prediction mismatches)."""
    try:
        with cwm_viewer.database(task) as db:
            return _read_store(db, problem, calls)
    except (ValueError, sqlite3.Error):  # no store (older protocols) or one still being created
        return [], []


def _read_store(
    db: sqlite3.Connection, problem: BaseProblem, calls: list[ToolCallView]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes = {
        row[0]: (row[1], row[2] or "")
        for row in db.execute("SELECT id, initial_obs_id, purpose FROM episodes ORDER BY id")
    }
    if not episodes:
        return [], []
    steps = db.execute(
        """SELECT s.episode_id, t.after_id, a.payload, json_extract(o.payload,'$.info')
        FROM step_events s JOIN transitions t ON t.id=s.transition_id
        JOIN actions a ON a.id=t.action_id JOIN observations o ON o.id=t.after_id
        ORDER BY s.id"""
    ).fetchall()
    events = [
        (float(at), kind, json.loads(payload))
        for at, kind, payload in db.execute(
            "SELECT timestamp, kind, payload FROM events WHERE kind IN (?,?,?,?,?,?) ORDER BY id",
            (*_RUN_EVENTS, *_RESET_EVENTS, _MODEL_EVENT, "dataset_prepared"),
        )
    ]
    times, mismatch_at = _step_times(events, len(steps), calls)
    reset_times = [at for at, kind, _ in events if kind in _RESET_EVENTS]
    info = {
        obs: db.execute(
            "SELECT json_extract(payload,'$.info') FROM observations WHERE id=?", (obs,)
        ).fetchone()[0]
        for obs, _ in episodes.values()
    }

    frames: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []
    actions = resets = 0
    best = 0.0
    origin = next((at for at, kind, _ in events if kind == "dataset_prepared"), None)
    origin = origin if origin is not None else (times[0] if times else 0.0)

    def add(obs: int, at: float, raw_info: str | None, **extra: Any) -> None:
        nonlocal best
        at = max(at, frames[-1]["at"]) if frames else at  # event clocks can disagree by a little
        frame = {"obs": obs, "env_actions": actions + resets, "at": at, **extra}
        value = problem.progress_value(json.loads(raw_info or "{}"))
        if value is not None and value > best:
            best = value
            markers.append(
                {
                    "kind": "progress",
                    "label": f"{value:g}",
                    "at": at,
                    "env_actions": frame["env_actions"],
                }
            )
        frames.append(frame)

    shown_episodes: set[int] = set()

    def start_episode(episode: int, at: float) -> None:
        nonlocal resets
        obs, purpose = episodes[episode]
        shown_episodes.add(episode)
        if purpose.startswith("reset_"):  # a reset is an action even when the picture is unchanged
            resets += 1
            add(obs, reset_times.pop(0) if reset_times else at, info.get(obs), reset=True)
        elif not frames or frames[-1]["obs"] != obs:
            add(obs, at, info.get(obs))

    first = min(episodes)
    start_episode(first, origin)
    for index, (episode, after, action, raw_info) in enumerate(steps):
        at = times[index] if index < len(times) else (frames[-1]["at"] if frames else origin)
        for pending in sorted(e for e in episodes if e <= episode and e not in shown_episodes):
            start_episode(pending, at)
        decoded = json.loads(action)
        actions += problem.counts_as_action(decoded)
        add(after, at, raw_info, action=decoded)
        if index in mismatch_at:
            frames[-1]["diagnostic"] = mismatch_at[index]
            markers.append(
                {
                    "kind": "mismatch",
                    "label": "prediction mismatch",
                    "at": at,
                    "env_actions": frames[-1]["env_actions"],
                }
            )
    for pending in sorted(e for e in episodes if e not in shown_episodes):
        start_episode(pending, reset_times[0] if reset_times else frames[-1]["at"])

    for at, kind, payload in events:
        if kind == _MODEL_EVENT:
            result = payload.get("result") or {}
            accepted = result.get("accepted") is True
            markers.append(
                {
                    "kind": "model" if accepted else "model_refused",
                    "label": "model accepted" if accepted else "model refused",
                    "at": at,
                }
            )
    markers.sort(key=lambda m: m["at"])
    return frames, markers


def _step_times(
    events: list[tuple[float, str, dict[str, Any]]], n_steps: int, calls: list[ToolCallView]
) -> tuple[list[float], dict[int, int]]:
    """The time of every real step, and the steps where a prediction mismatch stopped a call.

    Steps are consumed in order by the controller runs (``transition_sequence`` per run); the
    steps before the first run are the initial data collection."""
    runs = [
        (at, payload.get("result") or {})
        for at, kind, payload in events
        if kind in _RUN_EVENTS and "transition_sequence" in (payload.get("result") or {})
    ]
    played = sum(len(result["transition_sequence"]) for _, result in runs)
    prepared = next((at for at, kind, _ in events if kind == "dataset_prepared"), None)
    first_run = runs[0][0] if runs else None
    initial_at = prepared if prepared is not None else (first_run or 0.0)
    times = [initial_at] * max(0, n_steps - played)
    starts = sorted((tool.started, tool.ended) for tool in calls if tool.started is not None)
    begins = [start for start, _ in starts]
    mismatch_at: dict[int, int] = {}
    for at, result in runs:
        n = len(result["transition_sequence"])
        k = bisect.bisect_right(begins, at)
        begin = begins[k - 1] if k else at
        # A script can chain runs in one tool call: the second starts where the first ended.
        begin = min(max(begin, times[-1] if times else begin), at)
        times.extend(begin + (i + 1) / n * (at - begin) for i in range(n))
        example = result.get("counterexample") or {}
        if n and example.get("kind") == "prediction_mismatch" and example.get("diagnostic_id"):
            mismatch_at[len(times) - 1] = int(example["diagnostic_id"])
    return times[:n_steps], mismatch_at


def frame_png(task: Path, problem: BaseProblem, observation: int, view: str) -> bytes:
    with cwm_viewer.database(task) as db:
        row = db.execute("SELECT payload FROM observations WHERE id=?", (observation,)).fetchone()
    if row is None:
        raise ValueError("unknown observation")
    obs = json.loads(row[0])
    task_name = cwm_viewer.task_name_of(task) if view == "viewer" else None
    return cwm_viewer.png(problem, obs, task_name)
