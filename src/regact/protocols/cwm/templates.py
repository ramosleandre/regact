"""Small agent-facing files; protocol enforcement remains in the trusted coordinator."""

from collections.abc import Iterator

from regact.features.base import FeatureContext
from regact.workspace.templates import TemplateFile

HELPER = '''"""A fresh action-list controller for each dream or real episode."""


class ExplorationControllerFromListActions:
    def __init__(self, actions, achieved=None):
        self.actions = list(actions)
        self.index = 0
        self.achieved = achieved

    def is_done(self, state):
        return self.index >= len(self.actions) or bool(self.achieved and self.achieved(state))

    def act(self, state):
        if self.is_done(state):
            raise RuntimeError("action list already finished")
        action = self.actions[self.index]
        self.index += 1
        return action

    def objective_reached(self, state):
        return bool(self.achieved(state)) if self.achieved is not None else None

    def completion_reason(self, state):
        return "goal_achieved" if self.objective_reached(state) else "plan_exhausted"
'''
CLIENT = '''import sys
import uuid
from regact.envclient.client import EnvClient
from regact.envclient.errors import InvalidActionError


class CWMPhaseRestriction(RuntimeError):
    """Expected boundary: real environment interaction is now restricted."""


class CwmEnvClient(EnvClient):
    def _post(self, route, body):
        body = dict(body, request_id=str(uuid.uuid4()))
        response = self._http.post(f"/env/{self._game_id}/{route}", json=body)
        if response.status_code in (409, 422):
            detail = response.json().get("detail", {})
            if isinstance(detail, dict):
                if detail.get("code") == "cwm_phase_restriction":
                    raise CWMPhaseRestriction(detail["message"])
                if detail.get("code") == "invalid_action":
                    raise InvalidActionError(detail["message"])
        response.raise_for_status()
        return response.json()

    def _apply(self, envelope):
        if envelope.get("notice"):
            print(envelope["notice"], file=sys.stderr)
        return super()._apply(envelope)

    def stop(self):
        self._post("stop", {})
'''
DATA = '''"""Read-only CWM experience API. Pages contain stable integer IDs."""

import base64
from pathlib import Path
import httpx
from framework.make_env import _BASE_URL, _GAME_ID


def _query(op, **args):
    with httpx.Client(timeout=120, limits=httpx.Limits(max_keepalive_connections=0)) as http:
        response = http.post(f"{_BASE_URL}/data/{_GAME_ID}", json={"op": op, **args})
        response.raise_for_status()
        return response.json()


def summary():
    return _query("summary")


def list_observations(after_id=0, limit=None):
    return _query(
        "list_observations", after_id=after_id, **({"limit": limit} if limit is not None else {})
    )


def list_transitions(after_id=0, limit=None):
    return _query(
        "list_transitions", after_id=after_id, **({"limit": limit} if limit is not None else {})
    )


def load_observations(ids):
    return _query("observations", ids=list(ids))


def load_transitions(ids):
    return _query("transitions", ids=list(ids))


def load_diagnostic(diagnostic_id):
    return _query("diagnostic", id=diagnostic_id)


def save_image(path, *, observation_id=None, transition_id=None, diagnostic_id=None, which="after"):
    value = _query(
        "image",
        observation_id=observation_id,
        transition_id=transition_id,
        diagnostic_id=diagnostic_id,
        which=which,
    )
    Path(path).write_bytes(base64.b64decode(value["png_base64"]))
    return str(path)
'''
CONTROL = '''"""Call one CWM framework tool. No automatic retries of uncertain requests."""

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from framework.make_env import _BASE_URL, _GAME_ID

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: python framework/control.py TOOL [request_id]")
    request_id = sys.argv[2] if len(sys.argv) == 3 else str(uuid.uuid4())
    print("request_id=" + request_id, file=sys.stderr, flush=True)
    with httpx.Client(timeout=None, limits=httpx.Limits(max_keepalive_connections=0)) as client:
        response = client.post(
            f"{_BASE_URL}/control/{_GAME_ID}/tool",
            json={"name": sys.argv[1], "input": {"request_id": request_id}},
        )
        response.raise_for_status()
        result = response.json()
    print(result["output"])
    raise SystemExit(1 if result.get("is_error") else 0)
'''
GUIDE = """# CWM workspace interfaces

Real data: `from framework import cwm_data`. Use `summary()`, paginated
`list_observations(after_id=0, limit=...)` and `list_transitions(...)`, then
`load_observations(ids)` or `load_transitions(ids)`. Transitions contain `o`,
`action`, `o_next`, and stable IDs. `load_diagnostic(id)` retrieves a counterexample.
`save_image(path, observation_id=...)`, or `transition_id=..., which="before"/"after"`,
or `diagnostic_id=..., which="predicted"/"observed"` renders evidence to PNG.

The complete observation is a JSON dictionary with frame, reward, is_done,
available_actions and info. Exact equality covers all five fields. Observation
identity is sorted compact UTF-8 JSON (object key order is irrelevant).

`world_model/model_state.py`: define `State` (prefer a dataclass).
`model_parser.py`: `parse(obs_dict) -> State`.
`model_render.py`: `render(state) -> complete_obs_dict`.
`model_transition.py`: `step(state, action) -> State`.
Use `from model_state import State` within those modules. Functions must be
repeatable and must not mutate their inputs. State fields may contain JSON
values, tuples, and classes defined in bundled Python modules; dictionary keys
cannot start with @. The serializer includes class names/field names in size.

Submit with `python framework/control.py UpdateCodeWorldModel`. Validation checks
all unique real observations/transitions, injectivity, exact reconstruction and
prediction, and aggregate serialized-state / serialized-observation size.
The accepted model is an immutable revision; editing files does not update it.

In phase 2 choose an informative experiment. Describe the goal in the module
level docstring of `exploration.py`. Implement `get_controller()` returning an
object with `act(state)` and `is_done(state) -> bool`. Optional
`objective_reached(state) -> bool` distinguishes goal attainment from stopping.
Private controller memory is allowed, with a fresh instance for dream and reality.
Submit using `python framework/control.py SubmitExplorationController`.

Optional planning: describe the goal in the module docstring of `goal.py`.
Implement `achieved(state) -> bool` and optionally `utility(state) -> float in [0,1]`;
default utility is float(achieved). Achieved goals must have utility 1, while
utility 1 need not mean achieved. `python framework/control.py PlanInCWM` writes
`plans/plan_NNN.py` with ACTIONS and PLAN_ID if it finds a novelty-bearing candidate.
A plan can be incomplete; read the returned achieved/status fields.
Planning spends zero real actions. It does not automatically submit exploration.

To use a plan, import ACTIONS from its module and
`from framework.exploration import ExplorationControllerFromListActions`.
Return `ExplorationControllerFromListActions(ACTIONS, achieved=...)` from your
factory (goal predicate optional). Its completion can mean plan exhaustion.
The controller is always dream-checked again before real execution.

Submitted code executes in a separate restricted process: no experience DB,
workdir, live environment or network. Access these while writing code, not from
model/goal/controller callbacks. Static local Python imports are bundled;
dynamic imports of local files and external data files are not captured.
Put required constants in imported Python modules. Bundle limit: 256 Python
files / 10 MiB. Do not rely on external mutable files or installed game engines.
Only frozen code plus explicit observation/state/action inputs reaches execution.
"""


def templates(ctx: FeatureContext) -> Iterator[TemplateFile]:
    yield TemplateFile("framework/cwm_client.py", CLIENT)
    yield TemplateFile(
        "framework/make_env.py",
        f"""from framework.cwm_client import CwmEnvClient
_BASE_URL = {ctx.env_base_url!r}
_GAME_ID = {ctx.task_name!r}
def make_env(*, record_frames=False):
    env = CwmEnvClient.connect(_BASE_URL, _GAME_ID)
    env.reset()
    return env
""",
    )
    yield TemplateFile("framework/control.py", CONTROL)
    yield TemplateFile("framework/cwm_data.py", DATA)
    yield TemplateFile("framework/exploration.py", HELPER)
    yield TemplateFile("CWM_INTERFACE.md", GUIDE)
    yield TemplateFile("world_model/__init__.py", "")
    yield TemplateFile(
        "world_model/model_state.py",
        "from dataclasses import dataclass\n\n@dataclass(frozen=True)\nclass State:\n    pass\n",
    )
    for name, sig in [
        ("parser", "parse(obs)"),
        ("render", "render(state)"),
        ("transition", "step(state, action)"),
    ]:
        yield TemplateFile(
            f"world_model/model_{name}.py",
            f"from model_state import State\n\ndef {sig}:\n"
            '    raise NotImplementedError("Implement the CWM using recorded experience")\n',
        )
    yield TemplateFile(
        "goal.py",
        (
            '"""Describe the experimental goal here."""\ndef achieved(state):\n '
            "   raise NotImplementedError\n"
        ),
    )
    yield TemplateFile(
        "exploration.py",
        (
            '"""Describe the experimental goal here."""\ndef get_controller():\n'
            "    raise NotImplementedError\n"
        ),
    )
