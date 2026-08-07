"""Verify your world model against every recorded transition.

Two checks over ``../data/transitions.jsonl``:

- **coherence** - ``render(parse(o)) == o`` over distinct observations;
- **transition accuracy** - ``render(step(parse(o), a)) == o'`` plus exact
  reward/done, over distinct ``(o, a)`` pairs (null while ``step`` is not
  implemented yet - build parse/render first).

Reports the model's code complexity and points at the exact transitions where
the model fails, so you can load and study them:

    python world_model/verify.py [--max-obs N] [--max-failures K] [--json]

Also importable from your own scripts:

    from verify import load_transitions
    bad = load_transitions()[17]        # the transition a failure points at

Diff details read ``rendered != observed`` (coherence) or
``predicted != observed`` (transition). Stdlib only; edit freely.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from pathlib import Path
from typing import Any

MAX_OBS_DEFAULT = int("__MAX_OBS__")
MAX_FAILURES_DEFAULT = int("__MAX_FAILURES__")

_DIR = Path(__file__).resolve().parent
_DATA = _DIR.parent / "data" / "transitions.jsonl"
_EXCLUDED_FROM_COMPLEXITY = {"model_notes.py", Path(__file__).name}


def load_transitions(path: Path | None = None) -> list[dict[str, Any]]:
    """All recorded transitions ``{o, a, r, o2, done}``, in file order.

    Skips blank/torn lines (a crash can truncate the last one).
    """
    source = path if path is not None else _DATA
    transitions: list[dict[str, Any]] = []
    if not source.exists():
        return transitions
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            transitions.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return transitions


def _normalize(value: Any) -> Any:
    """Round-trip through JSON so tuple-vs-list and similar artifacts cannot differ."""
    return json.loads(json.dumps(value))


def _diff(observed: Any, rendered: Any, path: str = "obs") -> str:
    """The first difference between two JSON trees (shape-aware), or ''."""
    if type(observed) is not type(rendered):
        return f"{path}: type {type(rendered).__name__} != {type(observed).__name__}"
    if isinstance(observed, dict):
        for key in observed:
            if key not in rendered:
                return f"{path}.{key}: missing"
            found = _diff(observed[key], rendered[key], f"{path}.{key}")
            if found:
                return found
        extra = sorted(set(rendered) - set(observed))
        return f"{path}: unexpected keys {extra}" if extra else ""
    if isinstance(observed, list):
        if len(observed) != len(rendered):
            return f"{path}: length {len(rendered)} != {len(observed)}"
        for index, (obs_item, ren_item) in enumerate(zip(observed, rendered, strict=True)):
            found = _diff(obs_item, ren_item, f"{path}[{index}]")
            if found:
                return found
        return ""
    if observed != rendered:
        return f"{path}: {rendered!r} != {observed!r}"
    return ""


def _select_obs(
    transitions: list[dict[str, Any]], max_obs: int
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    """Distinct obs to test (each ``o``, plus terminal ``o2``), most recent first,
    capped at ``max_obs``. Returns ``(picked, n_distinct_skipped)``."""
    candidates: list[tuple[int, dict[str, Any]]] = []
    for index, transition in enumerate(transitions):
        candidates.append((index, transition["o"]))
        if transition.get("done"):
            candidates.append((index, transition["o2"]))
    seen: set[str] = set()
    picked: list[tuple[int, dict[str, Any]]] = []
    skipped = 0
    for index, obs in reversed(candidates):
        key = json.dumps(obs, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        if len(picked) < max_obs:
            picked.append((index, obs))
        else:
            skipped += 1
    picked.reverse()
    return picked, skipped


def _load_model() -> tuple[Any, Any, Any]:
    """Import the sibling model modules; return ``(parse, render, step)``.

    ``step`` is ``None`` when ``model_transition.py`` is absent (a pre-v3
    workdir) - the transition check is then skipped, not failed.
    """
    sys.path.insert(0, str(_DIR))  # so model modules can import each other by name
    parser_module = importlib.import_module("model_parser")
    render_module = importlib.import_module("model_render")
    try:
        step = importlib.import_module("model_transition").step
    except ModuleNotFoundError:
        step = None
    return parser_module.parse, render_module.render, step


def _select_transitions(
    transitions: list[dict[str, Any]], max_transitions: int
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    """Distinct ``(o, a)`` transitions to test (deterministic env: one occurrence
    carries all the information), most recent first, capped. Returns
    ``(picked, n_distinct_skipped)``."""
    seen: set[str] = set()
    picked: list[tuple[int, dict[str, Any]]] = []
    skipped = 0
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        key = json.dumps([transition["o"], transition["a"]], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        if len(picked) < max_transitions:
            picked.append((index, transition))
        else:
            skipped += 1
    picked.reverse()
    return picked, skipped


def _check_transitions(
    tested: list[tuple[int, dict[str, Any]]], parse: Any, render: Any, step: Any
) -> tuple[int | None, list[dict[str, Any]]]:
    """Score ``render(step(parse(o), a)) == o'`` + reward/done over ``tested``.

    Returns ``(n_correct, failures)``; ``n_correct`` is ``None`` when ``step``
    is still the unimplemented stub (the metric is not earned yet).
    """
    failures: list[dict[str, Any]] = []
    correct = 0
    for index, transition in tested:
        try:
            state, reward, done = step(parse(transition["o"]), transition["a"])
            predicted = _normalize(render(state))
        except NotImplementedError:
            return None, []
        except Exception as exc:
            failures.append(
                {
                    "transition": index,
                    "check": "transition",
                    "kind": "error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        detail = _diff(_normalize(transition["o2"]), predicted)
        if not detail and reward != transition["r"]:
            detail = f"reward: {reward!r} != {transition['r']!r}"
        if not detail and bool(done) != bool(transition["done"]):
            detail = f"done: {bool(done)!r} != {bool(transition['done'])!r}"
        if detail:
            failures.append(
                {"transition": index, "check": "transition", "kind": "mismatch", "detail": detail}
            )
        else:
            correct += 1
    return correct, failures


def _complexity() -> dict[str, int]:
    """Total AST node count over the model's code (notes and this script excluded)."""
    total = 0
    for path in sorted(_DIR.glob("*.py")):
        if path.name in _EXCLUDED_FROM_COMPLEXITY:
            continue
        total += sum(1 for _ in ast.walk(ast.parse(path.read_text(encoding="utf-8"))))
    return {"ast_nodes": total}


def run(max_obs: int, max_failures: int) -> dict[str, Any]:
    transitions = load_transitions()
    tested, skipped = _select_obs(transitions, max_obs)
    parse, render, step = _load_model()
    failures: list[dict[str, Any]] = []
    coherent = 0
    for index, obs in tested:
        observed = _normalize(obs)
        try:
            rendered = _normalize(render(parse(obs)))
        except Exception as exc:
            failures.append(
                {
                    "transition": index,
                    "check": "representation",
                    "kind": "error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        detail = _diff(observed, rendered)
        if detail:
            failures.append(
                {
                    "transition": index,
                    "check": "representation",
                    "kind": "mismatch",
                    "detail": detail,
                }
            )
        else:
            coherent += 1

    tested_tr: list[tuple[int, dict[str, Any]]] = []
    skipped_tr = 0
    n_correct: int | None = None
    if step is not None:
        tested_tr, skipped_tr = _select_transitions(transitions, max_obs)
        n_correct, transition_failures = _check_transitions(tested_tr, parse, render, step)
        if n_correct is not None:
            failures.extend(transition_failures)
    return {
        "coherence": coherent / len(tested) if tested else 0.0,
        "n_obs_tested": len(tested),
        "n_obs_skipped": skipped,
        "transition_accuracy": (
            n_correct / len(tested_tr) if n_correct is not None and tested_tr else None
        ),
        "n_transitions_tested": len(tested_tr) if n_correct is not None else 0,
        "n_transitions_skipped": skipped_tr if n_correct is not None else 0,
        "n_transitions": len(transitions),
        "n_failures": len(failures),
        "complexity": _complexity(),
        "failures": failures[:max_failures],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-obs", type=int, default=MAX_OBS_DEFAULT)
    parser.add_argument("--max-failures", type=int, default=MAX_FAILURES_DEFAULT)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    try:
        report = run(args.max_obs, args.max_failures)
    except Exception as exc:  # e.g. a model module that does not import
        print(f"verify: cannot evaluate the model: {type(exc).__name__}: {exc}")
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(
        f"coherence: {report['coherence']:.4f} "
        f"({report['n_obs_tested']} distinct obs tested, "
        f"{report['n_obs_skipped']} skipped by --max-obs, "
        f"{report['n_transitions']} transitions on disk)"
    )
    if report["transition_accuracy"] is None:
        print("transition accuracy: n/a (implement world_model/model_transition.py to earn it)")
    else:
        print(
            f"transition accuracy: {report['transition_accuracy']:.4f} "
            f"({report['n_transitions_tested']} distinct (o, a) tested, "
            f"{report['n_transitions_skipped']} skipped by --max-obs)"
        )
    print(f"complexity: {report['complexity']['ast_nodes']} AST nodes")
    if report["n_failures"]:
        shown = len(report["failures"])
        print(f"failures ({shown} of {report['n_failures']} shown; model != observed):")
        for failure in report["failures"]:
            print(
                f"  transition {failure['transition']} "
                f"[{failure['check']}/{failure['kind']}]: {failure['detail']}"
            )
        print("load one with: from verify import load_transitions; load_transitions()[<n>]")
    else:
        print("no failures: the model reproduces every tested observation and transition exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
