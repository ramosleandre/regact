"""Trusted comparisons: worker outputs are data, never validation verdicts."""

from __future__ import annotations

from typing import Any

from regact.protocols.cwm import limits as budgets
from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.store import ExperienceStore, canonical
from regact.protocols.cwm.worker import Worker, WorkerError


def check_observation(obs: Any) -> dict[str, Any]:
    if not isinstance(obs, dict) or set(obs) != {
        "frame",
        "reward",
        "is_done",
        "available_actions",
        "info",
    }:
        raise WorkerError("render must return the complete observation dictionary")
    if (
        type(obs["is_done"]) is not bool
        or not isinstance(obs["available_actions"], list)
        or not isinstance(obs["info"], dict)
    ):
        raise WorkerError("render returned invalid observation fields")
    canonical(obs)
    return obs


def differences(expected: Any, actual: Any, limit: int | None, path: str = "") -> dict[str, Any]:
    """Bounded structural evidence plus the exact count of omitted evidence entries.

    A changed list length is one entry; shared elements are compared separately.
    This works for unequal/ragged shapes without broadcasting or padding grids.
    """
    out: list[dict[str, Any]] = []
    total = 0

    def add(p: str, a: Any, b: Any) -> None:
        nonlocal total
        total += 1
        if not budgets.reached(len(out), limit):
            out.append({"path": p, "predicted": str(a)[:120], "observed": str(b)[:120]})

    def visit(a: Any, b: Any, p: str) -> None:
        if type(a) is dict and type(b) is dict:
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    add(f"{p}.{key}".lstrip("."), a.get(key, "<missing>"), b.get(key, "<missing>"))
                else:
                    visit(a[key], b[key], f"{p}.{key}".lstrip("."))
        elif type(a) is list and type(b) is list:
            if len(a) != len(b):
                add(f"{p}.length".lstrip("."), len(a), len(b))
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{p}[{i}]")
        elif type(a) is not type(b):
            add(f"{p}.type".lstrip("."), type(a).__name__, type(b).__name__)
        elif canonical(a) != canonical(b):
            add(p, a, b)

    visit(expected, actual, path)
    result: dict[str, Any] = {"differences": out}
    if total > len(out):
        result["differences_omitted"] = total - len(out)
    return result


def add_state_size(summary: dict[str, Any], oid: int, state: Any, obs: Any) -> None:
    """Accumulate bytes and representative extremes without retaining extra states."""
    state_bytes = len(canonical(state).encode())
    observation_bytes = len(canonical(obs).encode())
    summary["state_bytes"] = summary.get("state_bytes", 0) + state_bytes
    summary["observation_bytes"] = summary.get("observation_bytes", 0) + observation_bytes
    item = {
        "observation_id": oid,
        "state_bytes": state_bytes,
        "observation_bytes": observation_bytes,
        "ratio": state_bytes / max(1, observation_bytes),
    }
    for key, sign in (("smallest_state", 1), ("largest_state", -1)):
        if key not in summary or sign * state_bytes < sign * summary[key]["state_bytes"]:
            summary[key] = item


def validate(
    worker: Worker, store: ExperienceStore, config: CwmConfig
) -> tuple[dict[str, Any], dict[int, Any]]:
    states: dict[int, Any] = {}
    by_state: dict[str, int] = {}
    summary: dict[str, Any] = {
        "accepted": False,
        "complete": False,
        "dataset_version": store.version,
        "observations_checked": 0,
        "transitions_checked": 0,
        "failures": {},
        "counterexamples": [],
        "state_bytes": 0,
        "observation_bytes": 0,
    }

    def failure(
        kind: str,
        data: dict[str, Any],
        predicted: Any = None,
        actual: Any = None,
        outputs: tuple[Any, Any] | None = None,
    ) -> None:
        summary["failures"][kind] = summary["failures"].get(kind, 0) + 1
        examples = summary["counterexamples"]
        if budgets.reached(len(examples), config.feedback.max_counterexamples):
            # Prefer a first example of a new failure kind over another of the same kind.
            kinds = [entry["kind"] for entry in examples]
            replace = next(
                (i for i in reversed(range(len(kinds))) if kinds.count(kinds[i]) > 1), None
            )
            if kind in kinds or replace is None:
                return
            examples.pop(replace)
        entry = {"kind": kind, **data}
        if predicted is not None:
            entry.update(differences(predicted, actual, config.feedback.max_diff_items))
            entry["diagnostic_id"] = store.diagnostic(
                {"predicted": predicted, "observed": actual, **entry}
            )
        elif outputs is not None:
            entry["diagnostic_id"] = store.diagnostic(
                {**entry, "first_output": outputs[0], "second_output": outputs[1]}
            )
        examples.append(entry)

    def finish_counts() -> None:
        summary["counterexamples_omitted"] = sum(summary["failures"].values()) - len(
            summary["counterexamples"]
        )

    try:
        for oid in store.observation_ids():
            obs = store.observation(oid)
            state = worker.call("parse", obs=obs)
            repeated = worker.call("parse", obs=obs)
            if canonical(repeated) != canonical(state):
                failure(
                    "non_deterministic_model",
                    {"observation_id": oid, "callback": "parse"},
                    outputs=(state, repeated),
                )
            text = canonical(state)
            if text in by_state and by_state[text] != oid:
                failure("parser_collision", {"observation_ids": [by_state[text], oid]})
            by_state[text] = oid
            states[oid] = state
            reconstructed = check_observation(worker.call("render", state=state))
            repeated = worker.call("render", state=state)
            if canonical(repeated) != canonical(reconstructed):
                failure(
                    "non_deterministic_model",
                    {"observation_id": oid, "callback": "render"},
                    outputs=(reconstructed, repeated),
                )
            if canonical(reconstructed) != canonical(obs):
                failure("reconstruction_mismatch", {"observation_id": oid}, reconstructed, obs)
            add_state_size(summary, oid, state, obs)
            summary["observations_checked"] += 1
        ratio = summary["state_bytes"] / max(1, summary["observation_bytes"])
        summary["state_obs_size_ratio"] = ratio
        if ratio >= config.threshold_max_state_obs_size_ratio:
            failure(
                "compression_ratio",
                {"ratio": ratio, "required_below": config.threshold_max_state_obs_size_ratio},
            )
        for tid in store.transition_ids():
            transition = store.transition(tid)
            state = states[transition["before_obs_id"]]
            successor = worker.call("step", state=state, action=transition["action"])
            repeated = worker.call("step", state=state, action=transition["action"])
            if canonical(repeated) != canonical(successor):
                failure(
                    "non_deterministic_model",
                    {"transition_id": tid, "callback": "step"},
                    outputs=(successor, repeated),
                )
            predicted = check_observation(worker.call("render", state=successor))
            if canonical(predicted) != canonical(transition["o_next"]):
                failure(
                    "prediction_mismatch",
                    {
                        "transition_id": tid,
                        "before_obs_id": transition["before_obs_id"],
                        "after_obs_id": transition["after_obs_id"],
                    },
                    predicted,
                    transition["o_next"],
                )
            summary["transitions_checked"] += 1
    except WorkerError as exc:
        summary["error"] = budgets.truncate_error(str(exc), config.feedback.max_error_chars)
        summary["error_type"] = exc.kind
        summary["error_context"] = exc.context
        finish_counts()
        return summary, states
    finish_counts()
    summary["complete"] = True
    summary["accepted"] = not summary["failures"]
    return summary, states
