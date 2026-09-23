"""Trusted comparisons: worker outputs are data, never validation verdicts."""

from __future__ import annotations

from typing import Any

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


def differences(expected: Any, actual: Any, limit: int, path: str = "") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def visit(a: Any, b: Any, p: str) -> None:
        if len(out) >= limit:
            return
        if type(a) is dict and type(b) is dict:
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    out.append(
                        {
                            "path": f"{p}.{key}".lstrip("."),
                            "expected": str(a.get(key, "<missing>"))[:120],
                            "observed": str(b.get(key, "<missing>"))[:120],
                        }
                    )
                else:
                    visit(a[key], b[key], f"{p}.{key}".lstrip("."))
                if len(out) >= limit:
                    break
        elif type(a) is list and type(b) is list and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                visit(x, y, f"{p}[{i}]")
                if len(out) >= limit:
                    break
        elif canonical(a) != canonical(b):
            out.append({"path": p, "expected": str(a)[:120], "observed": str(b)[:120]})

    visit(expected, actual, path)
    return out[:limit]


def validate(
    worker: Worker, store: ExperienceStore, config: CwmConfig
) -> tuple[dict[str, Any], dict[int, Any]]:
    states: dict[int, Any] = {}
    by_state: dict[str, int] = {}
    summary: dict[str, Any] = {
        "accepted": False,
        "complete": False,
        "dataset_revision": store.revision,
        "observations_checked": 0,
        "transitions_checked": 0,
        "failures": {},
        "counterexamples": [],
        "state_bytes": 0,
        "observation_bytes": 0,
    }

    def failure(kind: str, data: dict[str, Any], predicted: Any = None, actual: Any = None) -> None:
        summary["failures"][kind] = summary["failures"].get(kind, 0) + 1
        if len(summary["counterexamples"]) < config.feedback.max_counterexamples:
            entry = {"kind": kind, **data}
            if predicted is not None:
                entry["differences"] = differences(
                    predicted, actual, config.feedback.max_diff_items
                )
                entry["diagnostic_id"] = store.diagnostic(
                    {"predicted": predicted, "observed": actual, **entry}
                )
            summary["counterexamples"].append(entry)

    try:
        for oid in store.observation_ids():
            obs = store.observation(oid)
            state = worker.call("parse", obs=obs)
            if canonical(worker.call("parse", obs=obs)) != canonical(state):
                failure("non_deterministic_model", {"observation_id": oid})
            text = canonical(state)
            if text in by_state and by_state[text] != oid:
                failure("parser_collision", {"observation_ids": [by_state[text], oid]})
            by_state[text] = oid
            states[oid] = state
            reconstructed = check_observation(worker.call("render", state=state))
            if canonical(worker.call("render", state=state)) != canonical(reconstructed):
                failure("non_deterministic_model", {"observation_id": oid})
            if canonical(reconstructed) != canonical(obs):
                failure("reconstruction_mismatch", {"observation_id": oid}, reconstructed, obs)
            summary["state_bytes"] += len(text.encode())
            summary["observation_bytes"] += len(canonical(obs).encode())
            summary["observations_checked"] += 1
        ratio = summary["state_bytes"] / max(1, summary["observation_bytes"])
        summary["state_obs_size_ratio"] = ratio
        if ratio >= config.threshold_state_obs_size_ratio:
            failure(
                "compression_ratio",
                {"ratio": ratio, "required_below": config.threshold_state_obs_size_ratio},
            )
        for tid in store.transition_ids():
            transition = store.transition(tid)
            state = states[transition["before_obs_id"]]
            successor = worker.call("step", state=state, action=transition["action"])
            if canonical(
                worker.call("step", state=state, action=transition["action"])
            ) != canonical(successor):
                failure("non_deterministic_model", {"transition_id": tid})
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
        summary["error"] = str(exc)[: config.feedback.max_error_chars]
        summary["error_type"] = exc.kind
        return summary, states
    summary["complete"] = True
    summary["accepted"] = not summary["failures"]
    return summary, states
