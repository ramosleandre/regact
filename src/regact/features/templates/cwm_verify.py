"""Verify your world model against every recorded transition.

Scores your model against the data in ``../data/transitions.jsonl`` (deduplicated)
as four staged checks - each a precondition for the next, so a check below an
unmet one reads "fix the previous point first" and you always have one clear next
priority. It ends with a validity verdict.

    python world_model/verify.py [--max-used N] [--max-incoherences K] [--json]

Also importable from your own scripts (run from the workdir root, where solution.py is):

    from world_model.verify import load_transitions
    t = load_transitions()[17]        # the transition an incoherence points at

Diff details read ``predicted != observed``. Stdlib only; edit freely.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib
import json
import sys
from pathlib import Path
from typing import Any

MAX_UNIQUE_USED_DEFAULT = int("__MAX_UNIQUE_USED__")
MAX_INCOHERENCES_DEFAULT = int("__MAX_INCOHERENCES__")
_DIFF_EXAMPLES = 6  # differing leaves shown per incoherence (e.g. grid cells)
_EXAMPLE_IDS = 10  # transition indices listed per incoherence pattern

_DIR = Path(__file__).resolve().parent
_DATA = _DIR.parent / "data" / "transitions.jsonl"
_EXCLUDED_FROM_COMPLEXITY = {"model_notes.py", Path(__file__).name}


def load_transitions(path: Path | None = None) -> list[dict[str, Any]]:
    """All recorded transitions ``{o, a, r, o2, done}``, in file (play) order.

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


def _top_field(example: str) -> str:
    """The top-level obs field an example path points at: ``obs.frame[0][3]: ...`` -> ``frame``."""
    path = example.split(":", 1)[0]
    path = path[4:] if path.startswith("obs.") else path
    for sep in (".", "["):
        cut = path.find(sep)
        if cut >= 0:
            path = path[:cut]
    return path or "obs"


def _collect_diffs(observed: Any, rendered: Any, path: str, out: list[str]) -> int:
    """Count leaf differences between two JSON trees, appending up to ``_DIFF_EXAMPLES``
    example strings to ``out``. A structural mismatch (type/length/missing key) counts as
    one and does not descend - the shapes no longer line up."""
    if type(observed) is not type(rendered):
        if len(out) < _DIFF_EXAMPLES:
            out.append(f"{path}: type {type(rendered).__name__} != {type(observed).__name__}")
        return 1
    if isinstance(observed, dict):
        total = 0
        for key in observed:
            if key not in rendered:
                if len(out) < _DIFF_EXAMPLES:
                    out.append(f"{path}.{key}: missing")
                total += 1
            else:
                total += _collect_diffs(observed[key], rendered[key], f"{path}.{key}", out)
        for key in rendered:
            if key not in observed:
                if len(out) < _DIFF_EXAMPLES:
                    out.append(f"{path}.{key}: unexpected")
                total += 1
        return total
    if isinstance(observed, list):
        if len(observed) != len(rendered):
            if len(out) < _DIFF_EXAMPLES:
                out.append(f"{path}: length {len(rendered)} != {len(observed)}")
            return 1
        total = 0
        for index, (obs_item, ren_item) in enumerate(zip(observed, rendered, strict=True)):
            total += _collect_diffs(obs_item, ren_item, f"{path}[{index}]", out)
        return total
    if observed != rendered:
        if len(out) < _DIFF_EXAMPLES:
            out.append(f"{path}: {rendered!r} != {observed!r}")
        return 1
    return 0


def _diff(observed: Any, rendered: Any) -> tuple[str, int, str]:
    """``(summary, n_diffs, top_field)``. Empty summary => identical. For a grid, ``n_diffs``
    is the number of differing cells and the summary shows the first few."""
    examples: list[str] = []
    n = _collect_diffs(observed, rendered, "obs", examples)
    if n == 0:
        return "", 0, ""
    shown = "; ".join(examples)
    more = f" (+{n - len(examples)} more)" if n > len(examples) else ""
    return shown + more, n, _top_field(examples[0])


def _state_key(state: Any) -> str:
    """A canonical serialization of a state, for injectivity + size. Uses the state's OWN fields -
    never render(), which would trivially equal the obs. dataclass -> ``__dict__`` -> repr."""
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        return json.dumps(dataclasses.asdict(state), sort_keys=True, default=repr)
    try:
        return json.dumps(vars(state), sort_keys=True, default=repr)
    except TypeError:
        return repr(state)


def _exc(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _incoherence(
    index: int, check: str, kind: str, part: str, detail: str, n: int = 1
) -> dict[str, Any]:
    return {
        "transition": index,
        "check": check,
        "kind": kind,
        "part": part,
        "detail": detail,
        "n_diffs": n,
    }


def _group(incoherences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse incoherences with the same (check, kind, part, detail) into one pattern listing
    every transition it hits - a systematic bug corrupts every observation identically, so 8
    identical lines become one. Grouped collisions pass through. Order preserved (recent first)."""
    patterns: dict[tuple[str, ...], dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for item in incoherences:
        if "transitions" in item:  # a collision, already a pattern
            passthrough.append(item)
            continue
        key = (item["check"], item["kind"], item["part"], item["detail"])
        pattern = patterns.get(key)
        if pattern is None:
            patterns[key] = {
                **{k: v for k, v in item.items() if k != "transition"},
                "transitions": [item["transition"]],
            }
        else:
            pattern["transitions"].append(item["transition"])
    grouped = passthrough + list(patterns.values())
    for pattern in grouped:  # a terminal o2 shares its transition index -> dedup the load targets
        pattern["transitions"] = sorted(set(pattern["transitions"]))
    return grouped


def _select_obs(
    transitions: list[dict[str, Any]], max_used: int
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    """Distinct obs to check (each ``o``, plus terminal ``o2`` so the last state of an episode is
    covered), most recent first, capped at ``max_used``. Returns ``(picked, n_obs_total)`` where
    n_obs_total is every obs occurrence in the data (before dedup)."""
    candidates: list[tuple[int, dict[str, Any]]] = []
    for index, transition in enumerate(transitions):
        candidates.append((index, transition["o"]))
        if transition.get("done"):
            candidates.append((index, transition["o2"]))
    seen: set[str] = set()
    picked: list[tuple[int, dict[str, Any]]] = []
    for index, obs in reversed(candidates):
        key = json.dumps(obs, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        if len(picked) < max_used:
            picked.append((index, obs))
    picked.reverse()
    return picked, len(candidates)


def _select_transitions(
    transitions: list[dict[str, Any]], max_used: int
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    """Distinct transitions to check - deduped by the FULL tuple ``(o, a, r, o2, done)`` so a
    repeated step collapses, but the SAME ``(o, a)`` yielding a different outcome is KEPT. Most
    recent first, capped. Returns ``(picked, n_conflicting_pairs)`` (an ``(o, a)`` with more than
    one outcome is a determinism / hidden-state violation)."""
    seen: set[str] = set()
    outcomes: dict[str, set[str]] = {}
    picked: list[tuple[int, dict[str, Any]]] = []
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        pair = json.dumps([transition["o"], transition["a"]], sort_keys=True)
        outcome = json.dumps(
            [transition["o2"], transition.get("r"), transition.get("done")], sort_keys=True
        )
        outcomes.setdefault(pair, set()).add(outcome)
        full = pair + outcome
        if full in seen:
            continue
        seen.add(full)
        if len(picked) < max_used:
            picked.append((index, transition))
    picked.reverse()
    conflicts = sum(1 for outs in outcomes.values() if len(outs) > 1)
    return picked, conflicts


def _load_model() -> tuple[Any, Any, Any]:
    """Import the sibling model modules; return ``(parse, render, step)``. ``step`` is ``None`` when
    ``model_transition.py`` is absent - the transition check is then skipped, not failed."""
    sys.path.insert(0, str(_DIR))  # so model modules can import each other by name
    parse = importlib.import_module("model_parser").parse
    render = importlib.import_module("model_render").render
    try:
        step = importlib.import_module("model_transition").step
    except ModuleNotFoundError:
        step = None
    return parse, render, step


def _complexity() -> dict[str, int]:
    """AST node count per model module (notes and this script excluded), plus the total."""
    per_file: dict[str, int] = {}
    for path in sorted(_DIR.glob("*.py")):
        if path.name in _EXCLUDED_FROM_COMPLEXITY:
            continue
        nodes = sum(1 for _ in ast.walk(ast.parse(path.read_text(encoding="utf-8"))))
        per_file[path.stem.replace("model_", "")] = nodes
    per_file["total"] = sum(per_file.values())
    return per_file


def _size_stats(state_sizes: list[int], obs_sizes: list[int]) -> dict[str, Any]:
    if not state_sizes:
        return {"avg": 0, "min": 0, "max": 0, "avg_obs": 0, "ratio": None}
    avg_state = sum(state_sizes) / len(state_sizes)
    avg_obs = sum(obs_sizes) / len(obs_sizes) if obs_sizes else 0
    return {
        "avg": round(avg_state),
        "min": min(state_sizes),
        "max": max(state_sizes),
        "avg_obs": round(avg_obs),
        "ratio": round(avg_state / avg_obs, 4) if avg_obs else None,
    }


# --------------------------------------------------------------------------- #
# The staged checks
# --------------------------------------------------------------------------- #
def _parse_states(
    tested_obs: list[tuple[int, dict[str, Any]]], parse: Any
) -> tuple[list[tuple[int, dict[str, Any], Any, str]], list[dict[str, Any]], bool]:
    """Parse each tested obs -> ``(parsed, crashes, not_implemented)``; a parsed item is
    ``(index, obs, state, state_key)``. A stub parse (``NotImplementedError``) short-circuits."""
    parsed: list[tuple[int, dict[str, Any], Any, str]] = []
    crashes: list[dict[str, Any]] = []
    for index, obs in tested_obs:
        try:
            state = parse(obs)
        except NotImplementedError:
            return [], [], True
        except Exception as exc:
            crashes.append(_incoherence(index, "injectivity", "error", "parse", _exc(exc)))
            continue
        parsed.append((index, obs, state, _state_key(state)))
    return parsed, crashes, False


def _injectivity(
    parsed: list[tuple[int, dict[str, Any], Any, str]], n_tested: int
) -> tuple[float, list[dict[str, Any]], int, int]:
    """``(fraction, collisions, n_colliding_obs, n_collision_states)`` - fraction of tested obs
    mapping to a DISTINCT state (obs that crashed produced no state, so they lower it too);
    n_colliding_obs is how many observations share a state with another, over n_collision_states
    distinct shared states."""
    groups: dict[str, list[int]] = {}
    for index, _obs, _state, key in parsed:
        groups.setdefault(key, []).append(index)
    fraction = len(groups) / n_tested if n_tested else 1.0
    colliding = [idxs for idxs in groups.values() if len(idxs) > 1]
    collisions = [
        {
            "check": "injectivity",
            "kind": "collision",
            "part": "parse",
            "detail": f"{len(idxs)} observations parse to one state",
            "transitions": sorted(set(idxs)),
        }
        for idxs in colliding
    ]
    return fraction, collisions, sum(len(idxs) for idxs in colliding), len(colliding)


def _render_check(
    parsed: list[tuple[int, dict[str, Any], Any, str]], render: Any, n_tested: int
) -> tuple[float, list[dict[str, Any]]]:
    """``(coherence, incoherences)`` for ``render(parse(o)) == o`` over the parsed obs."""
    coherent = 0
    incoherences: list[dict[str, Any]] = []
    for index, obs, state, _key in parsed:
        try:
            rendered = _normalize(render(state))
        except Exception as exc:
            incoherences.append(_incoherence(index, "representation", "error", "render", _exc(exc)))
            continue
        detail, n_diffs, part = _diff(_normalize(obs), rendered)
        if detail:
            incoherences.append(
                _incoherence(index, "representation", "mismatch", part, detail, n_diffs)
            )
        else:
            coherent += 1
    return (coherent / n_tested if n_tested else 0.0), incoherences


def _transition_check(
    tested_tr: list[tuple[int, dict[str, Any]]], parse: Any, render: Any, step: Any
) -> tuple[float | None, list[dict[str, Any]], bool]:
    """``(accuracy, incoherences, not_implemented)`` for ``render(step(parse(o), a)) == o2``."""
    correct = 0
    incoherences: list[dict[str, Any]] = []
    for index, transition in tested_tr:
        try:
            predicted = _normalize(render(step(parse(transition["o"]), transition["a"])))
        except NotImplementedError:
            return None, [], True
        except Exception as exc:
            incoherences.append(_incoherence(index, "transition", "error", "step", _exc(exc)))
            continue
        detail, n_diffs, part = _diff(_normalize(transition["o2"]), predicted)
        if detail:
            incoherences.append(
                _incoherence(index, "transition", "mismatch", part, detail, n_diffs)
            )
        else:
            correct += 1
    return (correct / len(tested_tr) if tested_tr else 0.0), incoherences, False


def _blank_report(coverage: dict[str, int], conflicts: int) -> dict[str, Any]:
    return {
        "coverage": coverage,
        "parser_injectivity": None,
        "injectivity_collisions": {"n_observations": 0, "n_states": 0},
        "representation_coherence": None,
        "transition_accuracy": None,
        "complexity": _complexity(),
        "state_size": {"avg": 0, "min": 0, "max": 0, "avg_obs": 0, "ratio": None},
        "n_conflicting_transitions": conflicts,
        "valid": False,
        "next_priority": None,
        "active_point": None,
        "incoherences": [],
        "n_incoherences": 0,
        "n_patterns": 0,
    }


def _attach(report: dict[str, Any], incoherences: list[dict[str, Any]], cap: int) -> None:
    patterns = _group(incoherences)
    report["incoherences"] = patterns[:cap]
    report["n_patterns"] = len(patterns)
    report["n_incoherences"] = sum(len(p["transitions"]) for p in patterns)


def run(max_used: int, max_incoherences: int) -> dict[str, Any]:
    transitions = load_transitions()
    tested_obs, obs_total = _select_obs(transitions, max_used)
    tested_tr, conflicts = _select_transitions(transitions, max_used)
    coverage = {
        "n_unique_obs": len(tested_obs),
        "n_obs_total": obs_total,
        "n_unique_transitions": len(tested_tr),
        "n_transitions_total": len(transitions),
    }
    report = _blank_report(coverage, conflicts)
    if not tested_obs:
        report["next_priority"] = "gathering data (play the game)"
        return report

    parse, render, step = _load_model()
    parsed, crashes, parse_not_impl = _parse_states(tested_obs, parse)
    state_sizes = [len(key) for _i, _o, _s, key in parsed]
    obs_sizes = [len(json.dumps(obs)) for _i, obs, _s, _key in parsed]
    report["state_size"] = _size_stats(state_sizes, obs_sizes)

    if parse_not_impl:
        report["active_point"] = 1
        report["next_priority"] = "implementing the parser"
        return report

    # Compute all three numbers (so the agent always sees its progress on every component); the
    # staging below only decides the single next priority + which point's incoherences to detail.
    injectivity, collisions, n_coll_obs, n_coll_states = _injectivity(parsed, len(tested_obs))
    report["parser_injectivity"] = injectivity
    report["injectivity_collisions"] = {"n_observations": n_coll_obs, "n_states": n_coll_states}
    coherence, repr_incoh = _render_check(parsed, render, len(tested_obs))
    report["representation_coherence"] = coherence
    accuracy, tr_incoh, step_not_impl = _transition_check(tested_tr, parse, render, step)
    report["transition_accuracy"] = accuracy  # None while step is the stub

    if injectivity < 1.0:
        report["active_point"] = 1
        report["next_priority"] = "fixing parser injectivity"
        _attach(report, crashes + collisions, max_incoherences)
    elif coherence < 1.0:
        report["active_point"] = 2
        report["next_priority"] = "fixing representation coherence"
        _attach(report, repr_incoh, max_incoherences)
    elif step_not_impl:
        report["active_point"] = 3
        report["next_priority"] = "implementing the rule of motion"
    elif accuracy is not None and accuracy < 1.0:
        report["active_point"] = 3
        report["next_priority"] = "fixing the rule of motion"
        _attach(report, tr_incoh, max_incoherences)
    else:
        report["valid"] = True
    return report


# --------------------------------------------------------------------------- #
# Human-readable rendering
# --------------------------------------------------------------------------- #
def _print_incoherences(report: dict[str, Any], header: str) -> None:
    print(f"   {header}")
    for pattern in report["incoherences"]:
        ids = pattern["transitions"][:_EXAMPLE_IDS]
        if pattern["kind"] == "collision":
            print(f"     - observations {ids} parse to one state")
        else:
            print(
                f"     - [{pattern['part']}] x{len(pattern['transitions'])} "
                f"e.g. {ids}: {pattern['detail']}"
            )


def _render_report(report: dict[str, Any]) -> None:
    cov = report["coverage"]
    print(
        f"Verify results ({cov['n_unique_obs']} unique observations used, {cov['n_obs_total']} "
        f"total on disk, {cov['n_unique_transitions']} unique transitions used, "
        f"{cov['n_transitions_total']} total on disk):"
    )
    print()
    if cov["n_transitions_total"] == 0:
        print("No transitions recorded yet - play the game to gather data, then verify.")
        return

    inj, rep, tra, active = (
        report["parser_injectivity"],
        report["representation_coherence"],
        report["transition_accuracy"],
        report["active_point"],
    )

    q1 = "1) Parser injectivity (do different observations always parse to different states?): "
    print(
        q1
        + ("not implemented (write parse)" if inj is None else ("True" if inj == 1.0 else "False"))
    )
    if inj is not None and inj < 1.0:
        c = report["injectivity_collisions"]
        if c["n_states"]:
            print(
                f"   {c['n_observations']} observations parse to a state another observation also "
                f"parses to; those {c['n_observations']} observations collapsed into "
                f"{c['n_states']} distinct states."
            )
        _print_incoherences(
            report,
            "Fix this first - the parser does not give a distinct state to every observation:",
        )
    print()

    q2 = "2) Representation coherence (fraction where render(parse(o)) == o): "
    print(q2 + ("n/a" if rep is None else f"{rep:.4f}"))
    if active == 2:
        _print_incoherences(
            report, "Fix this - the model cannot rebuild these observations exactly:"
        )
    print()

    q3 = "3) Transition coherence (does step(state, action) reach the real next state?): "
    unwritten = "not implemented (write step in model_transition.py)"
    print(q3 + (unwritten if tra is None else f"{tra:.4f}"))
    if active == 3 and report["incoherences"]:
        _print_incoherences(report, "Fix this - the model mispredicts these transitions:")
    print()

    comp, size = report["complexity"], report["state_size"]
    print("4) World model complexity:")
    print(
        f"     parser {comp.get('parser', 0)}, render {comp.get('render', 0)}, "
        f"state class {comp.get('state', 0)}, step {comp.get('transition', 0)} AST nodes"
    )
    print(
        f"     average state size {size['avg']} B versus average observation "
        f"{size['avg_obs']} B (ratio {size['ratio']})"
    )
    print()

    if report["n_conflicting_transitions"]:
        n = report["n_conflicting_transitions"]
        print(
            f"Note: {n} (o,a) pair{'s' if n != 1 else ''} gave different outcomes "
            "in the data (the game"
        )
        print(
            "looked non-deterministic here - these can never be predicted, "
            "and are not counted against you)."
        )
        print()
    if report["incoherences"]:
        print(
            "Index reference: an index i is the i-th observation/transition you "
            "recorded, in play order;"
        )
        print("load it with load_transitions()[i].")
        print()

    if report["valid"]:
        print("Validity of your Code World Model given current data: True")
        print(
            "   Your model reproduces every observation and transition you have gathered. "
            "You can now use it"
        )
        print(
            "   as a simulator: pick a target state, plan how to reach it INSIDE the model "
            "(chaining step, at"
        )
        print(
            "   no real cost), then run that plan in the real environment to confirm "
            "- and keep verifying."
        )
    else:
        print("Validity of your Code World Model given current data: False")
        print(f"   Your next priority: {report['next_priority']}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-used", type=int, default=MAX_UNIQUE_USED_DEFAULT)
    parser.add_argument("--max-incoherences", type=int, default=MAX_INCOHERENCES_DEFAULT)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    try:
        report = run(args.max_used, args.max_incoherences)
    except Exception as exc:  # e.g. a model module that does not import
        print(f"verify: cannot evaluate the model: {type(exc).__name__}: {exc}")
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _render_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
