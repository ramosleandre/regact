"""Isolated worker entry point. Stdlib only; never import this in the trusted process."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import inspect
import json
import math
import random
import resource
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

_now = time.perf_counter  # bound before any submitted code can rebind time.perf_counter
spent = 0.0  # seconds in submitted code for the current request


def agent(fn: Callable[..., Any], *args: Any) -> Any:
    """Run submitted code, charging its duration to the agent's RunController budget."""
    global spent
    started = _now()
    try:
        return fn(*args)
    finally:
        spent += _now() - started


def encode(value: Any, path: str = "state") -> Any:
    if value is None or type(value) in (bool, int, float, str):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} is {value}: State numbers must be finite")
        return value
    if type(value) is list:
        return [encode(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if type(value) is tuple:
        return {"@tuple": [encode(v, f"{path}[{i}]") for i, v in enumerate(value)]}
    if type(value) is dict:
        for k in value:
            if not isinstance(k, str) or k.startswith("@"):
                raise ValueError(
                    f"{path} has the key {k!r}: State dictionary keys must be strings "
                    "not starting with @"
                )
        return {k: encode(v, f"{path}[{k!r}]") for k, v in value.items()}
    cls = type(value)
    module = importlib.import_module(cls.__module__)
    file = getattr(module, "__file__", "")
    if not file or not Path(file).resolve().is_relative_to(BUNDLE):
        hint = " Convert it with .tolist()." if hasattr(value, "tolist") else ""
        raise ValueError(
            f"{path} is a {cls.__module__}.{cls.__qualname__}: State fields must be JSON values "
            "(numbers, strings, booleans, None, lists, dictionaries with string keys), tuples, "
            f"or instances of classes defined in world_model/.{hint}"
        )
    if dataclasses.is_dataclass(value):
        fields = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    elif hasattr(value, "__dict__"):
        fields = vars(value)
    else:
        raise ValueError(
            f"{path} is a {cls.__qualname__} without instance attributes (__slots__?): "
            "use a dataclass or a plain class"
        )
    return {
        "@class": cls.__module__ + ":" + cls.__qualname__,
        "fields": {k: encode(v, f"{path}.{k}") for k, v in fields.items()},
    }


def decode(value: Any) -> Any:
    if type(value) is list:
        return [decode(v) for v in value]
    if type(value) is not dict:
        return value
    if "@tuple" in value:
        return tuple(decode(v) for v in value["@tuple"])
    if "@class" in value:
        module, name = value["@class"].split(":", 1)
        cls: Any = importlib.import_module(module)
        for part in name.split("."):
            cls = getattr(cls, part)
        obj = object.__new__(cls)
        for key, field in decode(value["fields"]).items():
            object.__setattr__(obj, key, field)
        return obj
    return {k: decode(v) for k, v in value.items()}


def immutable_call(fn: Callable[..., Any], value: Any, *args: Any) -> Any:
    before = encode((value, args))
    result = agent(fn, value, *args)
    if encode((value, args)) != before:
        raise ValueError("CWM/goal callbacks must not mutate their input")
    return result


def state_result(state: Any) -> Any:
    if not isinstance(state, State):
        raise TypeError(
            "get_initial_state/step must return an instance of world_model.model_state.State"
        )
    return encode(state)


class _Trace:
    """One run of a stochastic step: the value given to every `choose` and its probability."""

    def __init__(self, pins: dict[str, Any], prefix: list[Any], rng: Any = None) -> None:
        self.pins, self.prefix, self.rng = pins, prefix, rng
        self.items: list[tuple[str, Any, float, bool]] = []  # name, value, probability, pinned
        self.free: list[tuple[Any, float, list[tuple[Any, float]]]] = []  # value, p, alternatives
        self.dists: list[list[tuple[Any, float]]] = []  # per item, its values most probable first
        self.possible = True

    def choose(self, name: Any, dist: Any) -> Any:
        if not isinstance(name, str) or not name:
            raise TypeError("choose(name, dist): name must be a non-empty string")
        if any(name == item[0] for item in self.items):
            raise ValueError(f"choose was called twice with the name {name!r} in one step")
        if isinstance(dist, dict):
            options = list(dist.items())
        elif isinstance(dist, (list, tuple, set, frozenset)) and dist:
            values = list(dist) if isinstance(dist, (list, tuple)) else sorted(dist, key=repr)
            options = [(value, 1 / len(values)) for value in values]
        else:
            raise TypeError(f"choose({name!r}): dist must be a dictionary value -> probability "
                            "or a non-empty list of equally likely values")
        total = 0.0
        for _, p in options:
            if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
                raise ValueError(f"choose({name!r}): probabilities must be numbers in [0, 1]")
            total += p
        if abs(total - 1) > 1e-6:
            raise ValueError(f"choose({name!r}): probabilities sum to {total}, not 1")
        ranked = sorted((item for item in options if item[1] > 0), key=lambda item: -item[1])
        self.dists.append(ranked)
        if name in self.pins:
            value = self.pins[name]
            p = next((q for v, q in ranked if v == value), 0.0)
            if p == 0:
                self.possible = False
                value, p = ranked[0][0], 0.0
            self.items.append((name, value, p, True))
            return value
        k = len(self.free)
        if self.rng is not None:
            value, p = self.rng.choices(ranked, weights=[q for _, q in ranked])[0]
        elif k < len(self.prefix):
            value = self.prefix[k]
            p = next(q for v, q in ranked if v == value)
        else:
            value, p = ranked[0]
        self.free.append((value, p, ranked))
        self.items.append((name, value, p, False))
        return value

    def bits(self) -> float:
        return sum(-math.log2(p) for _, _, p, _ in self.items) if self.possible else math.inf


def run_step(state: Any, action: Any, trace: _Trace) -> Any:
    before = encode(state)
    result = agent(step, state, action, trace.choose)
    if encode(state) != before:
        raise ValueError("CWM/goal callbacks must not mutate their input")
    return result


def choice_values(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise TypeError("explain must return a dictionary: choice name -> value")
    for name, item in value.items():
        if item is not None and type(item) not in (bool, int, float, str):
            raise TypeError(f"explain: the value of {name!r} must be a string, number, boolean or None")
    return value


def run_step_op(request: dict[str, Any]) -> dict[str, Any]:
    """One run of step with the given values: `pins` by name, `prefix` for the other choices in
    call order, the most probable value for the rest. Every choice comes back with its values
    and probabilities, so the trusted side prices the run and searches the alternatives."""
    trace = _Trace(request.get("pins") or {}, request.get("prefix") or [])
    after = run_step(decode(request["state"]), request["action"], trace)
    return {
        "state": state_result(after),
        "possible": trace.possible,
        "choices": [
            [name, value, p, pinned, [[v, q] for v, q in ranked]]
            for (name, value, p, pinned), ranked in zip(trace.items, trace.dists, strict=True)
        ],
    }


def handle(request: dict[str, Any]) -> Any:
    global controller
    op = request["op"]
    if op == "get_initial_state":
        return state_result(immutable_call(get_initial_state, request["obs"]))
    if op == "render":
        return immutable_call(render, decode(request["state"]))
    if op == "run_step":
        return run_step_op(request)
    if op == "explain":
        if explain is None:
            return {}
        return choice_values(immutable_call(explain, decode(request["state"]), request["action"], request["obs"]))
    if op == "start_state":
        previous = request.get("previous")
        if reset is None or previous is None:
            return state_result(immutable_call(get_initial_state, request["obs"]))
        return state_result(immutable_call(reset, decode(previous), request["obs"]))
    if op == "step" and STOCHASTIC:
        # outside validation: the likeliest outcome, or a sampled one when a seed is given
        seed = request.get("seed")
        rng = random.Random(seed) if seed is not None else None
        return state_result(run_step(decode(request["state"]), request["action"], _Trace({}, [], rng)))
    if op == "step":
        return state_result(immutable_call(step, decode(request["state"]), request["action"]))
    if op == "goal":
        goal = agent(importlib.import_module, "goal")
        state = decode(request["state"])
        achieved = immutable_call(goal.achieved, state)
        utility = (
            immutable_call(goal.utility, state) if hasattr(goal, "utility") else float(achieved)
        )
        if (
            type(achieved) is not bool
            or isinstance(utility, bool)
            or not isinstance(utility, (int, float))
            or not math.isfinite(utility)
            or not 0 <= utility <= 1
        ):
            raise ValueError("achieved must be bool and utility a finite number in [0,1]")
        if achieved and utility != 1:
            raise ValueError("an achieved goal must have utility 1")
        return {"achieved": achieved, "utility": utility}
    if op == "controller_init":
        controller = agent(lambda: importlib.import_module(request.get("module", "controller")).get_controller())
        if not callable(getattr(controller, "act", None)):
            raise TypeError("get_controller must return an object with act(state)")
        return None
    if op == "surprise_bits":
        # the controller's own threshold for stopping on an unlikely step; absent = the default
        if not hasattr(controller, "stop_on_surprise_bits"):
            return {"set": False}
        value = controller.stop_on_surprise_bits
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0
        ):
            raise TypeError("stop_on_surprise_bits must be a positive number, or None to never stop")
        return {"set": True, "value": None if value is None or math.isinf(value) else float(value)}
    if op == "is_done":
        if getattr(controller, "is_done", None) is None:
            return False  # optional: without it, only the environment and the limits stop a call
        result = agent(controller.is_done, decode(request["state"]))
        if type(result) is not bool:
            raise TypeError("is_done must return bool")
        return result
    if op == "act":
        return agent(controller.act, decode(request["state"]))
    if op == "completion_reason":
        fn = getattr(controller, "completion_reason", None)
        result = agent(fn, decode(request["state"])) if callable(fn) else "controller_done"
        if result not in ("controller_done", "goal_achieved", "plan_exhausted"):
            raise TypeError("invalid completion_reason")
        return result
    if op == "objective":
        fn = getattr(controller, "objective_reached", None)
        value = agent(fn, decode(request["state"])) if callable(fn) else None
        if value is not None and type(value) is not bool:
            raise TypeError("objective_reached must return bool or None")
        return value
    raise ValueError(f"unknown worker operation {op}")


if __name__ == "__main__":
    BUNDLE = Path(sys.argv[1]).resolve()
    if sys.argv[2] != "None":
        memory = int(sys.argv[2]) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    # Mirror the agent workspace: one root, ordinary package imports, one State class.
    sys.path.insert(0, str(BUNDLE))
    wire = sys.stdout
    STOCHASTIC, explain, reset = False, None, None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if len(sys.argv) < 4 or sys.argv[3] == "1":
                load = importlib.import_module
                State = agent(load, "world_model.model_state").State
                get_initial_state = agent(load, "world_model.model_initial_state").get_initial_state
                render = agent(load, "world_model.model_render").render
                step = agent(load, "world_model.model_transition").step
                STOCHASTIC = len(inspect.signature(step).parameters) == 3
                if STOCHASTIC and (BUNDLE / "world_model" / "model_explain.py").exists():
                    explain = agent(load, "world_model.model_explain").explain
                if STOCHASTIC and (BUNDLE / "world_model" / "model_reset.py").exists():
                    reset = agent(load, "world_model.model_reset").reset
        wire.write(json.dumps({"ready": True, "seconds": spent}) + "\n")
        wire.flush()
    except BaseException as exc:
        wire.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
        wire.flush()
        raise SystemExit(1) from None
    controller: Any = None
    for line in sys.stdin:
        request = json.loads(line)
        spent = 0.0
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = handle(request)
            payload = json.dumps(
                {"id": request["id"], "result": result, "seconds": spent}, allow_nan=False
            )
        except BaseException as exc:
            # Only submitted-code paths go back; no trusted traceback/credentials.
            payload = json.dumps(
                {"id": request["id"], "error": f"{type(exc).__name__}: {exc}", "seconds": spent}
            )
        wire.write(payload + "\n")
        wire.flush()
