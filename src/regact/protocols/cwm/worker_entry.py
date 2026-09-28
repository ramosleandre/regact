"""Isolated worker entry point. Stdlib only; never import this in the trusted process."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import json
import math
import resource
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


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
    result = fn(value, *args)
    if encode((value, args)) != before:
        raise ValueError("CWM/goal callbacks must not mutate their input")
    return result


def state_result(state: Any) -> Any:
    if not isinstance(state, State):
        raise TypeError("parse/step must return an instance of model_state.State")
    return encode(state)


def handle(request: dict[str, Any]) -> Any:
    global controller
    op = request["op"]
    if op == "parse":
        return state_result(immutable_call(parse, request["obs"]))
    if op == "render":
        return immutable_call(render, decode(request["state"]))
    if op == "step":
        return state_result(immutable_call(step, decode(request["state"]), request["action"]))
    if op == "goal":
        goal = importlib.import_module("goal")
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
        controller = importlib.import_module("exploration").get_controller()
        if not callable(getattr(controller, "act", None)) or not callable(
            getattr(controller, "is_done", None)
        ):
            raise TypeError(
                "get_controller must return an object with act(state) and is_done(state)"
            )
        return None
    if op == "is_done":
        result = controller.is_done(decode(request["state"]))
        if type(result) is not bool:
            raise TypeError("is_done must return bool")
        return result
    if op == "act":
        return controller.act(decode(request["state"]))
    if op == "completion_reason":
        fn = getattr(controller, "completion_reason", None)
        result = fn(decode(request["state"])) if callable(fn) else "controller_done"
        if result not in ("controller_done", "goal_achieved", "plan_exhausted"):
            raise TypeError("invalid completion_reason")
        return result
    if op == "objective":
        fn = getattr(controller, "objective_reached", None)
        value = fn(decode(request["state"])) if callable(fn) else None
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
    sys.path[:0] = [str(BUNDLE / "world_model"), str(BUNDLE)]
    wire = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            State = importlib.import_module("model_state").State
            parse = importlib.import_module("model_parser").parse
            render = importlib.import_module("model_render").render
            step = importlib.import_module("model_transition").step
        wire.write(json.dumps({"ready": True}) + "\n")
        wire.flush()
    except BaseException as exc:
        wire.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
        wire.flush()
        raise SystemExit(1) from None
    controller: Any = None
    for line in sys.stdin:
        request = json.loads(line)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = handle(request)
            payload = json.dumps({"id": request["id"], "result": result}, allow_nan=False)
        except BaseException as exc:
            # Only submitted-code paths go back; no trusted traceback/credentials.
            payload = json.dumps({"id": request["id"], "error": f"{type(exc).__name__}: {exc}"})
        wire.write(payload + "\n")
        wire.flush()
