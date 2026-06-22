"""Read a regact experiment directory into viz-ready structures.

An experiment dir holds one subdir per game (``<exp>/<game>/``) with
``logs/{transcript.jsonl, experiment_state.json}`` and
``workdir/submissions/<n|final>/results.json`` (+ optional ``*.mp4``).

The transcript is our flat normalized event stream; here we group it into
**turns** (text + thinking + tool calls/results + token usage) so the conversation
reads naturally, and pair each ``ToolResult`` to its ``ToolCall`` by id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from regact.security.detection import flag_tool_call
from regact.security.policy import default_policy


@dataclass
class ToolCallView:
    id: str
    name: str
    input: dict[str, Any]
    result: str | None = None
    is_error: bool = False
    tag: str | None = None  # "cheat" | "submit" | "submit_win" — drives the UI's call coloring


@dataclass
class TurnItem:
    """One thing in a turn, kept in chronological order."""

    kind: str  # "thinking" | "text" | "tool"
    text: str = ""  # for thinking / text
    tool: ToolCallView | None = None  # for tool


@dataclass
class TurnView:
    """One assistant turn: its items in the order they happened, + usage/error."""

    items: list[TurnItem] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    error: dict[str, str] | None = None  # {category, message} if the turn errored

    # Convenience views (used by metrics; the UI renders ``items`` in order).
    @property
    def thinkings(self) -> list[str]:
        return [i.text for i in self.items if i.kind == "thinking"]

    @property
    def texts(self) -> list[str]:
        return [i.text for i in self.items if i.kind == "text"]

    @property
    def tools(self) -> list[ToolCallView]:
        return [i.tool for i in self.items if i.kind == "tool" and i.tool is not None]


@dataclass
class SubmissionView:
    name: str  # "0", "1", …, "final"
    aggregate: dict[str, Any]
    episodes: list[dict[str, Any]]
    error: str | None
    videos: list[str]  # relative file names under the submission dir


@dataclass
class GameView:
    name: str
    state: dict[str, Any]
    turns: list[TurnView]
    submissions: list[SubmissionView]
    config: dict[str, Any]  # the resolved run config (agent, problem, limits, security…)


@dataclass
class ArtifactFile:
    relpath: str
    content: str
    too_large: bool = False


_MAX_ARTIFACT_BYTES = 200_000


def list_games(experiment_dir: str) -> list[str]:
    """Subdirectory names that look like game runs (have a logs/ dir)."""
    root = Path(experiment_dir)
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "logs").is_dir())


def load_game(experiment_dir: str, game: str) -> GameView:
    base = Path(experiment_dir) / game
    state = _load_json(base / "logs" / "experiment_state.json") or {}
    turns = _group_turns(_load_events(base / "logs" / "transcript.jsonl"))
    submissions = _load_submissions(base / "workdir" / "submissions")
    config = _load_json(base / "config.json") or {}
    _tag_tool_calls(turns, submissions)
    return GameView(name=game, state=state, turns=turns, submissions=submissions, config=config)


def list_artifacts(experiment_dir: str, game: str) -> list[ArtifactFile]:
    """The agent-authored Python in the workdir (solution.py, code_library/…)."""
    workdir = Path(experiment_dir) / game / "workdir"
    out: list[ArtifactFile] = []
    if not workdir.is_dir():
        return out
    for path in sorted(workdir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(workdir))
        try:
            if path.stat().st_size > _MAX_ARTIFACT_BYTES:
                out.append(ArtifactFile(rel, "", too_large=True))
            else:
                out.append(ArtifactFile(rel, path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    return out


def load_logs(experiment_dir: str, game: str) -> dict[str, Any]:
    """The human ``output.log`` + the structured ``events.jsonl`` (for error analysis)."""
    logs = Path(experiment_dir) / game / "logs"
    try:
        output = (logs / "output.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        output = ""
    return {"output": output, "events": _load_events(logs / "events.jsonl")}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (OSError, json.JSONDecodeError):
        return None


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return events
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a torn trailing line from a live write
    return events


def _group_turns(events: list[dict[str, Any]]) -> list[TurnView]:
    """Fold the flat event stream into turns (one per TurnComplete / error)."""
    turns: list[TurnView] = []
    current = TurnView()
    by_id: dict[str, ToolCallView] = {}

    def flush() -> None:
        nonlocal current, by_id
        if current.items or current.usage or current.error:
            turns.append(current)
        current = TurnView()
        by_id = {}

    for event in events:
        kind = event.get("type")
        if kind == "TextDelta":
            current.items.append(TurnItem("text", text=str(event.get("text", ""))))
        elif kind == "ThinkingDelta":
            current.items.append(TurnItem("thinking", text=str(event.get("text", ""))))
        elif kind == "SystemPrompt":
            current.items.append(TurnItem("system", text=str(event.get("text", ""))))
        elif kind == "UserMessage":
            current.items.append(TurnItem("user", text=str(event.get("text", ""))))
        elif kind == "ToolCall":
            call = ToolCallView(
                id=str(event.get("id", "")),
                name=str(event.get("name", "")),
                input=event.get("input") or {},
            )
            current.items.append(TurnItem("tool", tool=call))
            by_id[call.id] = call
        elif kind == "ToolResult":
            target = by_id.get(str(event.get("id", "")))
            if target is not None:
                target.result = str(event.get("output", ""))
                target.is_error = bool(event.get("is_error", False))
        elif kind == "TurnComplete":
            current.usage = event.get("usage")
            flush()
        elif kind == "AgentError":
            current.error = {
                "category": str(event.get("category", "")),
                "message": str(event.get("message", "")),
            }
            flush()
    flush()
    return turns


def _load_submissions(submissions_dir: Path) -> list[SubmissionView]:
    if not submissions_dir.is_dir():
        return []
    out: list[SubmissionView] = []
    for sub in sorted(submissions_dir.iterdir(), key=_submission_sort_key):
        results = _load_json(sub / "results.json") or {}
        videos = sorted(p.name for p in sub.glob("*.mp4"))
        out.append(
            SubmissionView(
                name=sub.name,
                aggregate=results.get("aggregate", {}),
                episodes=results.get("episodes", []),
                error=results.get("error"),
                videos=videos,
            )
        )
    return out


def _submission_sort_key(path: Path) -> tuple[int, str]:
    # numbered submissions first (by number), then "final" / others last.
    return (int(path.name), "") if path.name.isdigit() else (1_000_000, path.name)


def _tag_tool_calls(turns: list[TurnView], submissions: list[SubmissionView]) -> None:
    """Tag each tool call for the UI: ``submit`` / ``submit_win`` / ``cheat``.

    Submits are matched to numbered submissions in order (the k-th submit wrote the
    k-th submission), so a submit that advanced the cleared-level count is a *win*.
    Cheats reuse the loop's own (non-blocking) flagger so the colors match the run's
    forensic count. ``submit`` wins over ``cheat`` when a call is both.
    """
    policy = default_policy()
    wins = _submission_wins(submissions)
    submit_index = 0
    for turn in turns:
        for call in turn.tools:
            if _is_submit_call(call):
                won = wins[submit_index] if submit_index < len(wins) else False
                call.tag = "submit_win" if won else "submit"
                submit_index += 1
            elif flag_tool_call(call.name, call.input, policy):
                call.tag = "cheat"


def _is_submit_call(call: ToolCallView) -> bool:
    """A SubmitSolution — a native tool call, or a workdir ``control.py SubmitSolution`` shell."""
    if call.name == "SubmitSolution":
        return True
    blob = json.dumps(call.input).lower()
    return "submitsolution" in blob and "control" in blob


def _submission_wins(submissions: list[SubmissionView]) -> list[bool]:
    """Per numbered submission (in order): did it clear a new level vs. all prior ones?"""
    wins: list[bool] = []
    running = 0.0
    for sub in sorted((s for s in submissions if s.name.isdigit()), key=lambda s: int(s.name)):
        levels = _submission_levels(sub.aggregate)
        wins.append(levels > running)
        running = max(running, levels)
    return wins


def _submission_levels(aggregate: dict[str, Any]) -> float:
    """Cleared-level count for a submission (falls back to a solved/success signal)."""
    levels = aggregate.get("mean_levels_completed")
    if levels is not None:
        return float(levels)
    return 1.0 if (aggregate.get("success_rate") or 0) > 0 else 0.0
