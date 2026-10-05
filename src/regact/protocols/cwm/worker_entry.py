"""Isolated worker entry point. Stdlib only; never import this in the trusted process."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import json
import math
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


def encode(value: Any) -> Any:
    if value is None or type(value) in (bool, int, float, str):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("state contains a non-finite number")
        return value
    if type(value) is list:
        return [encode(v) for v in value]
    if type(value) is tuple:
        return {"@tuple": [encode(v) for v in value]}
    if type(value) is dict:
        if any(not isinstance(k, str) or k.startswith("@") for k in value):
            raise ValueError("state dictionary keys must be strings not starting with @")
        return {k: encode(v) for k, v in value.items()}
    cls = type(value)
    module = importlib.import_module(cls.__module__)
    file = getattr(module, "__file__", "")
    if not file or not Path(file).resolve().is_relative_to(BUNDLE):
        raise ValueError(
            "State fields must be JSON values, tuples, or classes from the submitted bundle"
        )
    fields = (
        {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
        if dataclasses.is_dataclass(value)
        else vars(value)
    )
    return {"@class": cls.__module__ + ":" + cls.__qualname__, "fields": encode(fields)}


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


def handle(request: dict[str, Any]) -> Any:
    global controller
    op = request["op"]
    if op == "get_initial_state":
        return state_result(immutable_call(get_initial_state, request["obs"]))
    if op == "render":
        return immutable_call(render, decode(request["state"]))
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
        controller = agent(lambda: importlib.import_module("controller").get_controller())
        if not callable(getattr(controller, "act", None)) or not callable(
            getattr(controller, "is_done", None)
        ):
            raise TypeError(
                "get_controller must return an object with act(state) and is_done(state)"
            )
        return None
    if op == "is_done":
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
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if len(sys.argv) < 4 or sys.argv[3] == "1":
                load = importlib.import_module
                State = agent(load, "world_model.model_state").State
                get_initial_state = agent(load, "world_model.model_initial_state").get_initial_state
                render = agent(load, "world_model.model_render").render
                step = agent(load, "world_model.model_transition").step
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
