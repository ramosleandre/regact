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
DATA = '''"""Read recorded real experience without interacting with the environment.

Start by reading the function docstrings below or calling help(function). Initial experience and later exploration use this same API; reads never reset or step the environment. Do not call this API inside submitted callbacks: those run without dataset access.

IDs are stable positive integers. Observations and transitions are deduplicated: a repeated real step adds an occurrence, not necessarily a new observation or edge. Full observations are dictionaries with frame, reward, is_done, available_actions and info. Simulations never add predicted observations to this dataset.

Queries accept at most __MAX_ITEMS__ items. Bulk replies are capped at __MAX_BYTES__ serialized bytes; request smaller batches if needed. One complete record or image is always readable, even above the byte cap. No observation data is truncated. Inclusive feedback ranges such as "[4:9]" mean IDs 4 through 9 and can be passed directly to the loading functions.

Example from a workspace script:
    from framework import data_api
    print(data_api.summary())
    ids = data_api.list_observation_ids(limit=3)
    observations = data_api.load_observations(ids)  # same order as ids
    print(observations[0]["available_actions"])
    data_api.save_image("observation.png", observation_id=ids[0])

If your tools can display images, open observation.png; otherwise analyse the arrays from load_observations. Saving a PNG prints its path and returns None; it does not display the image to you automatically. The function docstrings also cover transition and diagnostic images.
"""
import base64
from pathlib import Path
import httpx
_BASE_URL = __BASE_URL__
_GAME_ID = __GAME_ID__


def _query(op, **args):
    with httpx.Client(timeout=120, limits=httpx.Limits(max_keepalive_connections=0)) as http:
        response = http.post(f"{_BASE_URL}/data/{_GAME_ID}", json={"op": op, **args})
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ValueError(f"Dataset query failed: {detail}") from None
        return response.json()


def summary():
    """Return the current dataset summary; does not reset or step the environment.

    phase: the current workflow phase.
    initial_observation_id: the first observation recorded in this task.
    current_observation_id: the live environment observation, updated after actions/resets.
    controller_start_observation_id: where the next RunController starts: current
      in single_instance, original initial observation in multi_instance.
    episode_id: the current real episode; controller calls can share it.
    reset_actions: explicit reset requests completed (each consumes one task action).
    n_unique_observations: distinct complete observation dictionaries.
    n_total_observations: recorded occurrences, including repetitions: one initial
      observation per episode plus one successor per real step. Reading data does
      not increase this count; simulations are excluded.
    n_unique_transitions: distinct (observation, action, next observation) triples.
    n_total_transitions: real steps recorded, including repeated transitions.
    n_started_episodes: real episodes begun, including those with no steps.
    milestones (only when nonempty): first occurrences across this task, with
      name, kind (progress/failure/event), observation_id, transition_id,
      episode_id and event_id. These refer to real experience, never simulations.
    Example: print(summary()); use its current_observation_id rather than guessing.
    """
    return _query("summary")


def list_observation_ids(after_id=0, limit=__DEFAULT_LIMIT__):
    """Return one page of unique observation IDs, ascending and greater than after_id.

    limit defaults to __MAX_ITEMS__, the maximum page size. Empty list means no
    more IDs at present. Example: page=list_observation_ids(limit=10); then request
    list_observation_ids(after_id=page[-1], limit=10) if page is nonempty.
    Fetch the selected full dictionaries with load_observations(page).
    None requests all remaining IDs, allowed only when the page cap is unlimited.
    Invalid page bounds raise ValueError.
    """
    return _query("list_observation_ids", after_id=after_id, limit=limit)


def list_transition_ids(after_id=0, limit=__DEFAULT_LIMIT__):
    """Return one page of unique transition IDs, ascending, without observation grids.

    after_id is exclusive; limit defaults to __MAX_ITEMS__ (the maximum).
    Example: ids=list_transition_ids(limit=5); edges=load_transitions(ids).
    Repeat with after_id=ids[-1] to paginate. IDs identify distinct edges, not
    chronological steps; repeated occurrences do not produce new IDs.
    None requests all remaining IDs, allowed only when the page cap is unlimited.
    Invalid page bounds raise ValueError; no matching edges returns [].
    """
    return _query("list_transition_ids", after_id=after_id, limit=limit)


def load_observations(ids):
    """Return full observation dictionaries in the same order as the requested IDs.

    ids accepts a list or an inclusive range string from exploration feedback:
    load_observations("[1:4, 8]") loads IDs 1, 2, 3, 4, 8 (not a Python slice).
    Each dictionary has frame, reward, is_done, available_actions and info.
    IDs are not embedded in the dictionaries: keep your input IDs alongside them.
    Example: obs=load_observations([summary()["initial_observation_id"]])[0].
    Maximum __MAX_ITEMS__ IDs; unknown/invalid IDs or oversized responses raise
    ValueError. Reduce the batch size if the response is too large; a single
    observation is always readable, even above the bulk byte cap.
    """
    return _query("observations", ids=ids if isinstance(ids, str) else list(ids))


def load_transitions(ids):
    """Return recorded transitions in request order, including both full observations.

    ids accepts a list or an inclusive range string, e.g. "[1:4, 8]".
    Each entry: transition_id; before_obs_id; after_obs_id; action (problem format);
    o (complete before observation); o_next (complete successor observation).
    Example: t=load_transitions(list_transition_ids(limit=1))[0]; print(t["action"]).
    These are unique edges, not episodes. Maximum __MAX_ITEMS__ IDs; invalid IDs
    or oversized responses raise ValueError. Fetch smaller batches when needed.
    """
    return _query("transitions", ids=ids if isinstance(ids, str) else list(ids))


def load_diagnostic(diagnostic_id):
    """Read one diagnostic ID returned by a framework command.

    kind describes the failed check; available evidence IDs identify real data.
    Observation comparisons contain predicted and observed dictionaries plus
    bounded differences. Repeatability diagnostics instead contain callback,
    first_output and second_output. Simulation errors can include predicted state,
    action and actions_from_start; these are predictions, not real experience.
    Example: d=load_diagnostic(3); print(d["kind"]). Use an actually returned ID.
    Not every diagnostic has images; a size failure may have only numerical data.
    Unknown IDs raise ValueError. Complete diagnostics remain readable regardless
    of the bulk byte cap.
    """
    return _query("diagnostic", id=diagnostic_id)


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


def templates(ctx, options, commands):
    for filename, body in (("framework/commands.py", CONTROL), ("framework/data_api.py", DATA)):
        yield TemplateFile(
            filename,
            body.replace("__BASE_URL__", repr(ctx.env_base_url))
            .replace("__GAME_ID__", repr(ctx.task_name))
            .replace(
                "__COMMAND_HELP__", "\n".join(f"{name}: {text}" for name, text in commands.items())
            )
            .replace("__COMMAND_NAMES__", repr(tuple(commands)))
            .replace("__DEFAULT_LIMIT__", repr(options.data_api.max_items))
            .replace(
                "__MAX_ITEMS__",
                str(options.data_api.max_items)
                if options.data_api.max_items is not None
                else "unlimited",
            )
            .replace(
                "__MAX_BYTES__",
                str(options.data_api.max_response_bytes)
                if options.data_api.max_response_bytes is not None
                else "unlimited",
            ),
        )
    yield TemplateFile("framework/action_list_controller.py", HELPER)
