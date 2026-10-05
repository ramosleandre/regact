"""Data API and command transport shared by externally managed protocols."""

from regact.workspace.templates import TemplateFile

HELPER = '''"""Controllers built from actions; simulation and reality each use a fresh instance."""


class ExplorationControllerFromListActions:
    """Execute actions in order, optionally stopping when achieved(state) is true.

    Example in controller.py:
        from framework.action_list_controller import ExplorationControllerFromListActions
        def get_controller():
            actions = [1, 2, 1]  # Replace with actions valid for your experiment.
            return ExplorationControllerFromListActions(actions)

    Pass a list/iterable of problem-format actions, not a filename. Optionally
    pass achieved=your_predicate to stop early and report objective attainment.
    Without achieved, list exhaustion stops execution but does not establish
    that any goal was reached.
    """
    def __init__(self, actions, achieved=None):
        self.actions = list(actions)
        self.index = 0
        self.achieved = achieved

    def is_done(self, state):
        """Stop before another action if the list ended or the goal predicate holds."""
        return self.index >= len(self.actions) or bool(self.achieved and self.achieved(state))

    def act(self, state):
        """Return the next action and advance private controller memory."""
        if self.is_done(state):
            raise RuntimeError("action list already finished")
        action = self.actions[self.index]
        self.index += 1
        return action

    def objective_reached(self, state):
        """True/False when a goal predicate exists; otherwise None (not evaluated)."""
        return bool(self.achieved(state)) if self.achieved is not None else None

    def completion_reason(self, state):
        """Distinguish attaining the goal from merely exhausting the action list."""
        return "goal_achieved" if self.objective_reached(state) else "plan_exhausted"
'''
DATA = '''"""Read recorded real experience. Reading never resets or steps the environment.

Do not call this API inside submitted callbacks (controller.py, world_model/): they run without
dataset access.

- An observation is a dictionary: frame, reward, is_done, available_actions, info. It is exactly
  what your controller and world model receive. frame is a list of grids made of plain Python
  lists, not numpy: use np.array(obs["frame"][-1]) for the last grid.
- A transition is one recorded step: observation --action--> next_observation.
- Each distinct observation and transition has a positive integer ID. A repeated step adds no new
  ID, so IDs are not a time order. For what happened in what order, use list_episodes() and
  load_history(episode_id).

Example:
    from framework import data_api
    print(data_api.summary())
    for t in data_api.load_transitions(data_api.list_transition_ids()):
        print(t["observation_id"], t["action"], t["next_observation_id"])__IMAGE_EXAMPLE__
"""
import base64
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx

_BASE_URL = __BASE_URL__
_GAME_ID = __GAME_ID__
_BATCH = __BATCH__  # the server's per-query cap; batching is done here, never by the caller


def _query(op: str, **args: Any) -> Any:
    with httpx.Client(timeout=120, limits=httpx.Limits(max_keepalive_connections=0)) as http:
        response = http.post(f"{_BASE_URL}/data/{_GAME_ID}", json={"op": op, **args})
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ValueError(f"Dataset query failed: {detail}") from None
        return response.json()


def _ids(ids: int | str | Iterable[int]) -> list[int]:
    if isinstance(ids, int):
        return [ids]
    if isinstance(ids, str):
        out: list[int] = []
        for part in ids.strip().strip("[]").split(","):
            if part.strip():
                start, _, end = part.partition(":")
                out.extend(range(int(start), int(end or start) + 1))
        return out
    return [int(i) for i in ids]


def _fetch(op: str, ids: list[int]) -> list[Any]:
    try:
        return _query(op, ids=ids)
    except ValueError as error:
        if len(ids) == 1 or "exceeds" not in str(error):
            raise
        half = len(ids) // 2  # over the byte cap: split until it fits (one record always fits)
        return _fetch(op, ids[:half]) + _fetch(op, ids[half:])


def _load(op: str, ids: int | str | Iterable[int]) -> list[Any]:
    ids = _ids(ids)
    size = _BATCH or max(1, len(ids))
    return [item for i in range(0, len(ids), size) for item in _fetch(op, ids[i : i + size])]


def _all(op: str) -> list[int]:
    ids: list[int] = []
    while page := _query(op, after_id=ids[-1] if ids else 0, limit=_BATCH):
        ids += page
        if not _BATCH:
            break
    return ids


def summary() -> dict[str, Any]:
    """The dataset summary, a dictionary with these keys:

    phase; initial_observation_id; current_observation_id (the live environment);
    controller_start_observation_id (where the next RunController starts);
    episode_id (the live episode); reset_actions;
    n_unique_observations, n_unique_transitions (number of IDs);
    n_total_observations, n_total_transitions (occurrences, repeats included);
    n_started_episodes; milestones (first occurrences of game events, when any).
    """
    return _query("summary")


def list_observation_ids() -> list[int]:
    """All observation IDs, ascending."""
    return _all("list_observation_ids")


def list_transition_ids() -> list[int]:
    """All transition IDs, ascending."""
    return _all("list_transition_ids")


def load_observations(ids: int | str | Iterable[int]) -> list[dict[str, Any]]:
    """Observation dictionaries, in the same order as ids.

    ids: a list of IDs, one ID, or a range string from command feedback: "[1:4, 8]" = 1, 2, 3, 4, 8.
    The dictionaries do not contain their ID: zip(ids, load_observations(ids)) pairs them.
    """
    return _load("observations", ids)


def load_transitions(ids: int | str | Iterable[int]) -> list[dict[str, Any]]:
    """Transition dictionaries, in the same order as ids (same ids formats as load_observations).

    Keys: transition_id, observation_id, action, next_observation_id,
    observation (full dictionary before the action), next_observation (full dictionary after).
    """
    return _load("transitions", ids)


def list_episodes() -> list[dict[str, Any]]:
    """Every episode in start order. Keys: episode_id; started_by ("initial_collection",
    "exploration", "reset_level", "reset_environment"); start_observation_id; n_steps; live
    (True for the episode the environment is in now)."""
    return _query("episodes")


def load_history(episode_id: int, step: int | None = None) -> tuple[list[dict[str, Any]], list[Any]]:
    """(observations, actions) of one episode in time order, up to `step` actions (default: all).
    observations[0] is the episode's first observation and observations[t] the one after
    actions[t - 1]."""
    history = _query("history", episode_id=episode_id, step=step)
    ids = history["observation_ids"]
    distinct = sorted(set(ids))
    by_id = dict(zip(distinct, load_observations(distinct)))
    return [by_id[i] for i in ids], history["actions"]


def load_diagnostic(diagnostic_id: int) -> dict[str, Any]:
    """One diagnostic returned by a framework command. kind names the failed check; observation
    comparisons hold predicted and observed dictionaries plus their differences."""
    return _query("diagnostic", id=diagnostic_id)
'''
SAVE_IMAGE = '''

def save_image(path, *, observation_id=None, transition_id=None, diagnostic_id=None, which=None):
    """Save a PNG, print its source and path, and return None; no real action occurs.

    Supply EXACTLY ONE source:
      observation_id: its frame (omit which).
      transition_id: which="before" or "after"; default "after".
      diagnostic_id: which="predicted" or "observed"; default "observed".
    Examples:
      save_image("obs.png", observation_id=1)
      save_image("before.png", transition_id=2, which="before")
      save_image("predicted.png", diagnostic_id=3, which="predicted")
    Use IDs returned by the API. Missing/ambiguous sources, invalid sides and
    absent diagnostic observations raise ValueError. Not all diagnostics have
    both sides. PNGs show the frame, not differences in reward/info/etc.
    The parent folder must exist. Then open the file with your image-reading tool
    if available; printing the filename alone does not show the image to you.
    """
    value = _query("image", observation_id=observation_id, transition_id=transition_id,
                   diagnostic_id=diagnostic_id, which=which)
    Path(path).write_bytes(base64.b64decode(value["png_base64"]))
    if observation_id is not None:
        source = f"observation {observation_id}"
    elif transition_id is not None:
        source = f"transition {transition_id} ({which or 'after'})"
    else:
        source = f"diagnostic {diagnostic_id} ({which or 'observed'})"
    print(f"Image of {source} saved at {path}")
'''
CONTROL = '''"""Run a framework command with its fixed workspace files; no positional arguments.

__COMMAND_HELP__
"""
import sys
import uuid
import httpx
_BASE_URL = __BASE_URL__
_GAME_ID = __GAME_ID__

if __name__ == "__main__":
    names = __COMMAND_NAMES__
    if sys.argv[1:] in ([], ["--help"], ["-h"]):
        print(__doc__)
        raise SystemExit(0)
    if len(sys.argv) != 2 or sys.argv[1] not in names:
        raise SystemExit("usage: python framework/commands.py " + "|".join(names) + " (no arguments)")
    with httpx.Client(timeout=None, limits=httpx.Limits(max_keepalive_connections=0)) as client:
        response = client.post(f"{_BASE_URL}/control/{_GAME_ID}/tool",
            json={"name": sys.argv[1], "input": {}},
            headers={"X-Regact-Request-ID": str(uuid.uuid4())})
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise SystemExit(f"Framework command failed: {detail}")
        result = response.json()
    print(result["output"], flush=True)
    for notice in result.get("messages", []):
        print("\\n" + notice, flush=True)
    raise SystemExit(1 if result.get("is_error") else 0)
'''


IMAGE_EXAMPLE = """
    data_api.save_image("observation.png", observation_id=data_api.list_observation_ids()[0])

Open observation.png with your image-reading tool. Saving a PNG prints its path and returns None; it does not display the image to you automatically. The function docstrings also cover transition and diagnostic images."""


def templates(ctx, options, commands, *, vision):
    data = DATA.replace("__IMAGE_EXAMPLE__", IMAGE_EXAMPLE if vision else "")
    data += SAVE_IMAGE if vision else ""
    for filename, body in (("framework/commands.py", CONTROL), ("framework/data_api.py", data)):
        yield TemplateFile(
            filename,
            body.replace("__BASE_URL__", repr(ctx.env_base_url))
            .replace("__GAME_ID__", repr(ctx.task_name))
            .replace(
                "__COMMAND_HELP__", "\n".join(f"{name}: {text}" for name, text in commands.items())
            )
            .replace("__COMMAND_NAMES__", repr(tuple(commands)))
            .replace("__BATCH__", repr(options.data_api.max_items)),
        )
    yield TemplateFile("framework/action_list_controller.py", HELPER)
