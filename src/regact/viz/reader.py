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
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from regact.security.detection import flag_os_denial, flag_tool_call
from regact.security.policy import default_policy


@dataclass
class ToolCallView:
    id: str
    name: str
    input: dict[str, Any]
    result: str | None = None
    is_error: bool = False
    images: list[dict[str, str]] = field(default_factory=list)
    tag: str | None = None  # policy submissions, CWM commands, or flagged calls
    framework_tool: str | None = None
    succeeded: bool | None = None
    controller_playback_id: int | None = None
    flags: list[str] = field(default_factory=list)  # why a call was tagged "cheat" (the reasons)


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
    aggregate: dict[str, Any]  # verified (shadow-replay) score on a shadow run, else the direct one
    episodes: list[dict[str, Any]]
    error: str | None
    videos: list[str]  # relative file names under the submission dir
    aggregate_unverified: dict[str, Any] | None = None  # direct score, when shadow-replay also ran
    features: dict[str, Any] = field(default_factory=dict)  # per-feature metrics, {"cwm": {...}}
    derived: dict[str, Any] = field(
        default_factory=dict
    )  # problem-derived metrics (ARC rhae/lrhae)


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
    size_bytes: int = 0


_MAX_ARTIFACT_BYTES = 200_000


_MAX_GAME_DEPTH = 8  # bound the walk; a sweep nests model/stamp/task = ~3 levels
# A real game leaf is marked by its run state, NOT by a bare ``logs/`` - the experiments root is
# littered with Slurm job-log dirs (``logs/{sbatch.*.out,simplelm.*.log}``, no run state) that would
# otherwise masquerade as games and flood the browser.
_GAME_MARKER = os.path.join("logs", "experiment_state.json")


def list_games(experiment_dir: str) -> list[str]:
    """Every run dir under the experiment root, as a path RELATIVE to the root.

    A run is a dir whose ``logs/experiment_state.json`` exists (:data:`_GAME_MARKER`) - a bare
    ``logs/`` is not enough, so Slurm job-log folders are skipped. Recursive so ONE viz serves a
    whole sweep (model x task x seed = many nested runs), not just a single flat run: the id is the
    relative path (e.g. ``Coder-480B_seed0/2026-08-15_.../MiniGrid-DoorKey-8x8-v0``).
    A dir that IS a run is not descended into; the walk is depth-bounded; and a run is
    deduped by realpath so a ``latest`` symlink is not listed alongside its timestamp dir.
    A flat single-run dir yields the same one-element list as before (relpath == name).
    """
    root = Path(experiment_dir)
    if not root.is_dir():
        return []
    games: list[str] = []
    seen: set[str] = set()

    def walk(d: Path, depth: int) -> None:
        if depth > _MAX_GAME_DEPTH:
            return
        try:
            subdirs = sorted(p for p in d.iterdir() if p.is_dir())
        except OSError:
            return
        for sub in subdirs:
            if (sub / _GAME_MARKER).is_file():
                real = os.path.realpath(sub)
                if real in seen:  # a `latest` symlink resolving to an already-listed run
                    continue
                seen.add(real)
                games.append(str(sub.relative_to(root)))  # a run: record it, do not descend
            else:
                walk(sub, depth + 1)

    walk(root, 0)
    return sorted(games)


# A run dir is stamped ``%Y-%m-%d_%H-%M-%S`` (experiment._STAMP_FORMAT). We name runs by this
# pattern rather than by "its children are all tasks", because a repeated task nests an extra
# ``<task>/attempt_N`` level (n_attempts_per_task) that would otherwise push every kind up one rung.
_RUN_STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def build_tree(experiment_dir: str) -> list[dict[str, Any]]:
    """A lazy nested tree of the folders down to task leaves, built from :func:`list_games`'s cheap
    walk - NO metric parsing - so a top folder holding many benchmarks renders instantly.

    Each node is ``{name, path (relpath from root), kind, n_tasks, n_children, children}``. ``kind``
    is one rung of the hierarchy benchmark -> experiment -> run -> task: ``run`` (a timestamped run
    dir, whatever nests below it), ``experiment`` (its children are runs), ``benchmark`` (its
    children are experiments), ``task`` (a leaf game dir, see :data:`_GAME_MARKER`), or ``group`` (a
    mixed / deeper / legacy folder). The frontend browses this; a task leaf opens the per-game view,
    everything else scopes the games list to its subtree (``/api/games?under=``, dashboard+graphs).
    """
    nested: dict[str, Any] = {}
    for game in sorted(list_games(experiment_dir)):
        node = nested
        for part in game.split("/"):
            node = node.setdefault(part, {})

    def to_nodes(children: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
        nodes = []
        for name, sub in sorted(children.items()):
            path = f"{prefix}/{name}" if prefix else name
            kids = to_nodes(sub, path)
            if _RUN_STAMP_RE.match(name):
                # A timestamped run: everything under it is task-level (a task, or task/attempt_N
                # for a repeated task). n_children counts the task dirs directly beneath it.
                kind, n_tasks = "run", len(kids)
            elif not kids:
                kind, n_tasks = "task", 1
            else:
                n_tasks = sum(k["n_tasks"] for k in kids)
                child_kinds = {k["kind"] for k in kids}
                # Named by what its children are: experiment (of runs), benchmark (of experiments),
                # else a mixed/deeper group.
                kind = (
                    "experiment"
                    if child_kinds == {"run"}
                    else "benchmark"
                    if child_kinds == {"experiment"}
                    else "group"
                )
            nodes.append(
                {
                    "name": name,
                    "path": path,
                    "kind": kind,
                    "n_tasks": n_tasks,
                    "n_children": len(kids),
                    "children": kids,
                }
            )
        return nodes

    return to_nodes(nested, "")


def load_game(experiment_dir: str, game: str) -> GameView:
    base = Path(experiment_dir) / game
    state = _load_json(base / "logs" / "experiment_state.json") or {}
    turns = _group_turns(_load_events(base / "logs" / "transcript.jsonl"))
    submissions = _load_submissions(base / "workdir" / "submissions")
    config = _load_json(base / "config.json") or {}
    _enrich_derived_metrics(game, submissions, config)
    _tag_tool_calls(turns, submissions)
    return GameView(name=game, state=state, turns=turns, submissions=submissions, config=config)


def _enrich_derived_metrics(
    game: str, submissions: list[SubmissionView], config: dict[str, Any]
) -> None:
    """Recompute a game's offline derived metrics (e.g. ARC's RHAE/LRHAE) into ``sub.derived``.

    Kept separate from the game-score aggregate so the viewer shows them under "Other". Delegates to
    the problem named in the resolved config, so the viewer stays agnostic of a game's metric keys.
    Best-effort: a missing game library or benchmark leaves ``derived`` empty rather than failing.
    """
    problem_cfg = config.get("problem") or {}
    name = problem_cfg.get("name")
    if not name or not submissions:
        return
    try:
        from regact.problems.base import build_problem

        problem = build_problem(name, problem_cfg.get("kwargs") or {})
        for sub in submissions:
            if sub.episodes:
                # A separate channel from the game score aggregate: derived metrics (ARC RHAE/LRHAE)
                # are secondary, so the viz shows them under "Other", not among the main score.
                sub.derived.update(problem.derived_submission_metrics(game, sub.episodes))
    except Exception:  # a viewer must render even if the problem cannot be rebuilt here
        return


def list_artifacts(experiment_dir: str, game: str) -> list[ArtifactFile]:
    """Python and Markdown workspace files, including generated interface guides."""
    workdir = Path(experiment_dir) / game / "workdir"
    out: list[ArtifactFile] = []
    if not workdir.is_dir():
        return out
    for path in sorted(workdir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".py", ".md") or "__pycache__" in path.parts:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(workdir.resolve()):
            continue
        rel = str(path.relative_to(workdir))
        try:
            size = path.stat().st_size
            if size > _MAX_ARTIFACT_BYTES:
                out.append(ArtifactFile(rel, "", too_large=True, size_bytes=size))
            else:
                out.append(ArtifactFile(rel, path.read_text(encoding="utf-8", errors="replace"), size_bytes=size))
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
    """Fold the flat event stream into groups (one per IterationComplete / error)."""
    turns: list[TurnView] = []
    current = TurnView()
    by_id: dict[str, ToolCallView] = {}

    def flush() -> None:
        nonlocal current
        if current.items or current.usage or current.error:
            turns.append(current)
        current = TurnView()
        # by_id is NOT reset: the alan adapter emits an IterationComplete per completion, so a
        # ToolResult can arrive a group after its ToolCall and must still pair to it (ids
        # are unique per call, so a persistent map cannot mis-pair).

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
                target.images = event.get("images") or []
        elif kind in ("IterationComplete", "TurnComplete"):  # "TurnComplete" = pre-rename bench 01
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
                aggregate_unverified=results.get("aggregate_unverified"),
                features=results.get("features") or {},
            )
        )
    return out


def _submission_sort_key(path: Path) -> tuple[int, str]:
    # numbered submissions first (by number), then "final" / others last.
    return (int(path.name), "") if path.name.isdigit() else (1_000_000, path.name)


def _tag_tool_calls(turns: list[TurnView], submissions: list[SubmissionView]) -> None:
    """Tag each tool call for the UI: ``submit`` / ``submit_win`` / ``cheat``.

    Submits are matched to numbered submissions in order (the k-th submit wrote the
    k-th submission), so a submit that advanced the cleared-level count is a *win*. A
    cheat is a call that reaches for a forbidden path/module in its args (unsandboxed)
    or whose result reads like an OS/proxy denial (sandboxed: blocked reads, curls) —
    the same signals the loop counts. ``submit`` wins over ``cheat`` when a call is both.
    """
    policy = default_policy()
    wins = _submission_wins(submissions)
    submit_index = 0
    for turn in turns:
        for call in turn.tools:
            if command := _cwm_command(call):
                call.framework_tool = command
                call.tag = "cwm"
                feedback = _cwm_feedback(call.result or "")
                call.succeeded = (
                    feedback.get("status") in ("Accepted", "Plan found", "Completed")
                    and not call.is_error
                ) if feedback else None
                if type(feedback.get("exploration_id")) is int and type(feedback.get("current_observation_id")) is int:
                    call.controller_playback_id = feedback["exploration_id"]
            elif _is_submit_call(call):
                won = wins[submit_index] if submit_index < len(wins) else False
                call.tag = "submit_win" if won else "submit"
                submit_index += 1
            kw = flag_tool_call(call.name, call.input, policy)
            denied = flag_os_denial(call.result or "")
            cwm_denied = "cwm_direct_environment_disabled" in (call.result or "") or "Direct environment access is unavailable in CWM" in (call.result or "")
            if kw or denied or cwm_denied:
                if not call.framework_tool and call.tag not in ("submit", "submit_win"):
                    call.tag = "cheat"
                call.flags = [*kw, *(["OS/proxy denial in result"] if denied else []), *(["Direct CWM environment access denied"] if cwm_denied else [])]


def _cwm_command(call: ToolCallView, _depth: int = 0) -> str | None:
    """Recognize native calls or actual Python CLI invocations, not grep/echo mentions."""
    import shlex

    from regact.protocols.cwm.commands import COMMANDS as CWM_COMMANDS

    COMMANDS = {**CWM_COMMANDS, "SubmitExplorationController": "", "ResetLevel": "", "ResetEnvironment": ""}

    if _depth > 8:
        return None
    if call.name in COMMANDS:
        return call.name
    raw = call.input if isinstance(call.input, dict) else {}
    text = raw.get("command", raw.get("cmd", ""))
    if not isinstance(text, str):
        return None
    # Codex quotes the entire script in `sh -lc '...'`. Unwrap that *before*
    # stripping heredocs: otherwise removing a quoted heredoc also destroys the
    # outer quotes and hides commands following it. Only parse; never execute.
    try:
        outer = shlex.shlex(text, posix=True, punctuation_chars=";&|()\n")
        outer.whitespace_split = True
        if Path(outer.get_token() or "").name in ("sh", "bash", "dash", "zsh"):
            for token in outer:
                if not token.startswith("-"):
                    break
                if "c" in token[1:]:
                    import dataclasses

                    script = outer.get_token()
                    if script is not None:
                        nested = dataclasses.replace(call, input={"command": script})
                        if command := _cwm_command(nested, _depth + 1):
                            return command
                    break
    except ValueError:
        pass
    lines = []
    delimiter = None
    for line in text.splitlines(keepends=True):
        if delimiter is not None:
            if line.strip() == delimiter:
                delimiter = None
            continue
        lines.append(line)
        match = re.search(r"<<-?\s*['\"]?(\w+)['\"]?\s*$", line)
        if match:
            delimiter = match.group(1)
    text = "".join(lines)
    try:
        lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|()\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        statements: list[list[str]] = [[]]
        for token in lexer:
            if token and all(c in ";&|()\n" for c in token):
                statements.append([])
            else:
                statements[-1].append(token)
        for tokens in statements:
            if len(tokens) >= 3 and Path(tokens[0]).name in ("sh", "bash", "dash", "zsh"):
                for i, token in enumerate(tokens[1:-1], 1):
                    if token.startswith("-") and "c" in token[1:]:
                        import dataclasses
                        nested = dataclasses.replace(call, input={"command": tokens[i + 1]})
                        command = _cwm_command(nested, _depth + 1)
                        if command:
                            return command
                        break
            if tokens[:2] == ["uv", "run"]:
                tokens = tokens[2:]
            if not tokens or not Path(tokens[0]).name.startswith("python"):
                continue
            # -c and stdin execute code; their text is not a CLI invocation.
            if "-c" in tokens or "-" in tokens:
                continue
            for i, token in enumerate(tokens[:-1]):
                if Path(token).name in {"control.py", "commands.py"} and tokens[i + 1] in COMMANDS:
                    return tokens[i + 1]
    except ValueError:
        pass
    return None


def _cwm_feedback(text: str) -> dict[str, Any]:
    """Read available result fields, including a prefix cut by `head` or a tool cap.

    Only complete top-level key/value pairs count; never infer missing fields or
    mistake a nested status for the command's outcome. The transcript is unchanged.
    """
    offset = 0
    for line in text.splitlines(keepends=True):
        start, offset = offset, offset + len(line)
        if line.lstrip().startswith("{"):
            try:
                value, _ = json.JSONDecoder().raw_decode(text[start:].lstrip())
            except ValueError:
                value = _json_object_prefix(text[start:].lstrip())
            if isinstance(value, dict) and "status" in value:
                return value
    return {}


def _json_object_prefix(text: str) -> dict[str, Any]:
    """Recover complete fields before an interrupted JSON object value."""
    decoder = json.JSONDecoder()
    result: dict[str, Any] = {}
    position = 1  # caller checked the opening brace
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text) or text[position] == "}":
            return result
        try:
            key, position = decoder.raw_decode(text, position)
            if not isinstance(key, str):
                return {}
            while position < len(text) and text[position].isspace():
                position += 1
            if position == len(text):
                return result
            if text[position] != ":":
                return {}
            position += 1
            while position < len(text) and text[position].isspace():
                position += 1
            value, position = decoder.raw_decode(text, position)
        except ValueError:
            return result
        # Require a delimiter: a cut number, e.g. 123|45, is not a complete ID.
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            return result
        if text[position] not in ",}":
            return {}
        result[key] = value
        if text[position] == "}":
            return result
        position += 1


def _is_submit_call(call: ToolCallView) -> bool:
    """A SubmitSolution — a native tool call, or a workdir ``control.py SubmitSolution`` shell.

    Matches the actual *invocation* (``control.py submitsolution``), NOT a grep/sed that merely
    mentions those strings (e.g. ``rg "…|control/.*tool|SubmitSolution"``) — a false submit there
    would shift the submit-to-submission alignment and mis-color the wins.
    """
    if call.name == "SubmitSolution":
        return True
    return "control.py submitsolution" in json.dumps(call.input).lower()


def _submission_wins(submissions: list[SubmissionView]) -> list[bool]:
    """Per numbered submission (in order): did it improve on all prior ones?"""
    wins: list[bool] = []
    running = 0.0
    for sub in sorted((s for s in submissions if s.name.isdigit()), key=lambda s: int(s.name)):
        progress = _submission_progress(sub.aggregate)
        wins.append(progress > running)
        running = max(running, progress)
    return wins


def _submission_progress(aggregate: dict[str, Any]) -> float:
    """A monotone progress signal for a submission, game-agnostic.

    Prefers a game's own graded depth if it reports one (ARC's cleared-level count),
    else falls back to the universal success signal every game reports — so the
    'improved?' marking works for any game without naming its metric keys.
    """
    depth = aggregate.get("mean_levels_completed")
    if depth is not None:
        return float(depth)
    return 1.0 if (aggregate.get("success_rate") or 0) > 0 else 0.0
