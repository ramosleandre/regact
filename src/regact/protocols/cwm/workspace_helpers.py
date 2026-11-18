"""Optional local CWM environment template."""

CWM_ENV = r'''"""Local environment backed by your current workspace CWM; no real actions.

EnvCWM(initial_state=state) never queries the dataset.
make_cwm_env() starts where the next controller starts.
Local simulation does not validate or accept code. Submit changed CWM code with
UpdateCodeWorldModel before using RunController.
serialized_size(value) measures a State or an observation as the compactness check does.
"""
import dataclasses
import json
from copy import deepcopy


def serialize(value):
    """Return the JSON text whose UTF-8 bytes the compactness check counts.

    Tuples become {"@tuple": [...]} and instances of your classes become
    {"@class": "module:Qualname", "fields": {...}}; keys are sorted, no spaces.
    """
    def encode(v):
        if v is None or type(v) in (bool, int, float, str):
            return v
        if type(v) is list:
            return [encode(x) for x in v]
        if type(v) is tuple:
            return {"@tuple": [encode(x) for x in v]}
        if type(v) is dict:
            return {k: encode(x) for k, x in v.items()}
        cls = type(v)
        if dataclasses.is_dataclass(v):
            fields = {f.name: getattr(v, f.name) for f in dataclasses.fields(v)}
        else:
            fields = vars(v)
        return {
            "@class": cls.__module__ + ":" + cls.__qualname__,
            "fields": {k: encode(x) for k, x in fields.items()},
        }

    return json.dumps(
        encode(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def serialized_size(value):
    """Bytes the compactness check counts for this State or observation."""
    return len(serialize(value).encode())


class EnvCWM:
    """reset()/step(action) return State, or an observation dict if obs_mode=True.

    Unlike Gym, step returns just that value, not a reward/termination tuple.
    Access observation() for reward, is_done and available_actions in either mode.
    reset() restores a copy of the constructor's starting state. reset(state)
    starts this episode elsewhere without changing the default starting state.
    The state property and returned values are copies, so callers cannot mutate
    the environment accidentally. Submitted callbacks must pass initial_state;
    this class never calls the experience API.
    """
    def __init__(self, obs_mode=False, initial_state=None):
        if type(obs_mode) is not bool:
            raise TypeError("obs_mode must be bool")
        if initial_state is None:
            raise ValueError(
                "EnvCWM needs initial_state. In workspace scripts, use "
                "framework.cwm_env.make_cwm_env() to load the recorded start."
            )
        self.obs_mode = obs_mode
        self._initial_state = deepcopy(initial_state)
        self.reset()

    @property
    def state(self):
        return deepcopy(self._state)

    def observation(self):
        """Render the full observation; does not advance the simulation."""
        from world_model import model_render
        return deepcopy(model_render.render(self.state))

    def reset(self, initial_state=None):
        """Restart from the default state or from the supplied State for this episode."""
        self._state = deepcopy(self._initial_state if initial_state is None else initial_state)
        return self.observation() if self.obs_mode else self.state

    def step(self, action):
        """Apply one problem-format action in the CWM only."""
        if self.observation()["is_done"]:
            raise RuntimeError("The simulated environment has ended; call reset() before step().")
        from world_model import model_transition
        self._state = deepcopy(model_transition.step(self.state, deepcopy(action)))
        return self.observation() if self.obs_mode else self.state


def make_cwm_env(obs_mode=False, initial_state=None):
    """Create EnvCWM starting where the next RunController will start, unless a State is given.

    single_instance: the current State, rebuilt with your workspace CWM from the live episode
    (get_initial_state on its first observation, then step through every recorded action).
    multi_instance: get_initial_state(initial observation). This factory is for workspace
    scripts, not submitted callbacks. Further resets never query the dataset.
    """
    if initial_state is None:
        from framework import data_api
        from world_model import model_initial_state, model_transition
        summary = data_api.summary()
        if summary["lifecycle"] == "single_instance":
            observations, actions = data_api.load_history(summary["episode_id"])
        else:
            observations = data_api.load_observations([summary["initial_observation_id"]])
            actions = []
        initial_state = model_initial_state.get_initial_state(deepcopy(observations[0]))
        for action in actions:
            initial_state = model_transition.step(initial_state, deepcopy(action))
    return EnvCWM(obs_mode=obs_mode, initial_state=initial_state)
'''


def _stochastic_helper() -> str:
    import inspect

    from regact.protocols.cwm import worker_entry

    return (
        _CWM_STOCHASTIC_HEAD
        + inspect.getsource(worker_entry._Trace)
        + _CWM_STOCHASTIC_TAIL
    )


_CWM_STOCHASTIC_HEAD = r'''"""Run and score your workspace stochastic CWM locally; no real actions, nothing is submitted.

current_state()                   the State your CWM gives for the live episode, now
explain_step(state, action, obs)  the explanation the framework would find for one transition
sample_step(state, action, rng)   one random outcome of step
replay(episode_id=None)           explain a whole recorded episode and report its bits
"""
import heapq
import json
import math
import random
from copy import deepcopy
from typing import Any


'''

_CWM_STOCHASTIC_TAIL = r'''

def _model():
    from world_model.model_initial_state import get_initial_state
    from world_model.model_render import render
    from world_model.model_transition import step
    try:
        from world_model.model_explain import explain
    except ImportError:
        explain = None
    return get_initial_state, render, step, explain


def _start(previous, obs):
    get_initial_state = _model()[0]
    try:
        from world_model.model_reset import reset
    except ImportError:
        reset = None
    return reset(deepcopy(previous), obs) if reset is not None and previous is not None else get_initial_state(obs)


def current_state(max_nodes=__EXPLAIN_NODES__):
    """The State at the current point of the live episode, as the framework computes it: every
    recorded episode is explained in order, and each new episode starts from reset(State at the end
    of the previous one, its first observation). Raises if a step cannot be explained."""
    from framework import data_api

    state = None
    for episode in data_api.list_episodes():
        episode_id = episode["episode_id"] if isinstance(episode, dict) else episode
        out = replay(episode_id, max_nodes, verbose=False, previous=state)
        if out["failed"]:
            raise ValueError(f"episode {episode_id}: step {out['failed']['step']} is not explained")
        state = out["state"]
    return state


def sample_step(state, action, rng=None):
    """(next State, [(name, value, probability, pinned), ...]) for one random outcome; with
    rng=None, the most probable value of every choice."""
    _, _, step, _ = _model()
    trace = _Trace({}, [], rng)
    return step(deepcopy(state), action, trace.choose), trace.items


def explain_step(state, action, obs, max_nodes=__EXPLAIN_NODES__):
    """What the framework does for one recorded transition. Returns a dictionary: explained,
    bits, choices [(name, value, probability, pinned)], state (the next State, or the most
    probable one when unexplained), rendered (its observation), runs."""
    _, render, step, explain = _model()
    pins = explain(deepcopy(state), action, obs) if explain is not None else {}
    target = json.dumps(obs, sort_keys=True)
    queue, runs, first = [(0.0, 0, [])], 0, None
    while queue and runs < max_nodes:
        _, _, prefix = heapq.heappop(queue)
        trace = _Trace(pins, prefix)
        runs += 1
        after = step(deepcopy(state), action, trace.choose)
        rendered = render(after)
        first = first or (after, trace, rendered)
        if trace.possible and json.dumps(rendered, sort_keys=True) == target:
            return {"explained": True, "bits": trace.bits(), "choices": trace.items,
                    "state": after, "rendered": rendered, "runs": runs}
        cost = 0.0
        for k, (value, p, ranked) in enumerate(trace.free):
            if k >= len(prefix):
                for other, q in ranked:
                    if other != value:
                        heapq.heappush(queue, (cost - math.log2(q), len(queue) + runs * 100000,
                                               [v for v, _, _ in trace.free[:k]] + [other]))
            cost -= math.log2(p)
    after, trace, rendered = first
    return {"explained": False, "bits": None, "choices": trace.items, "state": after,
            "rendered": rendered, "runs": runs,
            "impossible": [name for name, _, p, _ in trace.items if p == 0]}


def replay(episode_id=None, max_nodes=__EXPLAIN_NODES__, verbose=True, previous=None):
    """Explain a recorded episode (default: the latest) step by step. Returns a dictionary:
    steps explained, total bits, bits per choice group, and the first unexplained step if any
    (with the State before it, the action, the observed and the rendered observations).
    `previous` is the State the previous episode ended in, when this one follows a reset."""
    from framework import data_api

    get_initial_state, render, _, _ = _model()
    if episode_id is None:
        last = data_api.list_episodes()[-1]
        episode_id = last["episode_id"] if isinstance(last, dict) else last
    observations, actions = data_api.load_history(episode_id)
    state = _start(previous, observations[0])
    out = {"episode_id": episode_id, "steps": 0, "bits": 0.0, "by_choice": {}, "failed": None}
    if json.dumps(render(state), sort_keys=True) != json.dumps(observations[0], sort_keys=True):
        out["failed"] = {"step": 0, "observed": observations[0], "rendered": render(state)}
        return out
    for i, action in enumerate(actions):
        result = explain_step(state, action, observations[i + 1], max_nodes)
        if not result["explained"]:
            out["failed"] = {"step": i + 1, "action": action, "state_before": state,
                             "observed": observations[i + 1], **result}
            break
        state = result["state"]
        out["steps"] += 1
        out["bits"] += result["bits"]
        for name, _, p, _ in result["choices"]:
            group = out["by_choice"].setdefault(name.split("@", 1)[0], [0, 0.0])
            group[0] += 1
            group[1] -= math.log2(p)
    out["state"] = state
    if verbose:
        print(f"episode {episode_id}: {out['steps']}/{len(actions)} steps explained, "
              f"{out['bits']:.1f} bits", {k: (n, round(b, 1)) for k, (n, b) in out["by_choice"].items()})
        if out["failed"]:
            print("first unexplained step:", out["failed"]["step"], "action", out["failed"].get("action"))
    return out
'''

CWM_STOCHASTIC = _stochastic_helper().replace("__EXPLAIN_NODES__", "2000")
