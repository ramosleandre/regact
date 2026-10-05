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
    largest = summary.get("largest_ratio_state")
    if largest is None or item["ratio"] > largest["ratio"]:
        summary["largest_ratio_state"] = item


def compactness_failure(sizes: dict[str, Any], threshold: float) -> dict[str, Any] | None:
    """Both the aggregate ratio and the largest single state must stay under the threshold, so a
    State that grows along an episode (visited sets, logs) fails even when the average is low."""
    ratio = sizes.get("state_bytes", 0) / max(1, sizes.get("observation_bytes", 0))
    largest = sizes.get("largest_ratio_state")
    if ratio >= threshold:
        return {"ratio": ratio, "required_below": threshold}
    if largest is not None and largest["ratio"] >= threshold:
        return {"largest_state": largest, "required_below": threshold}
    return None


_REPEAT_EVERY = 10  # repeatability is checked on episode starts and every 10th step


def validate(
    worker: Worker, store: ExperienceStore, config: CwmConfig, *, live_episode: int | None = None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Replay every recorded episode in time order: get_initial_state on its first observation,
    then step carries the State, and render is checked against every recorded observation.

    Returns the summary and, when the live episode was predicted to its end, the running state
    there (``{"state", "hash"}``) for single_instance exploration to continue from.
    """
    summary: dict[str, Any] = {
        "accepted": False,
        "complete": False,
        "dataset_version": store.version,
        "episodes_checked": 0,
        "steps_checked": 0,
        "failures": {},
        "counterexamples": [],
        "state_bytes": 0,
        "observation_bytes": 0,
    }
    sizes: dict[str, Any] = {}
    repeated = store.repeated_histories()
    validated: dict[str, Any] = {}  # repeated history hash -> State there, validated once
    diverged: set[str] = set()  # repeated history hashes where a failure is already reported

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

    def call(op: str, repeat: bool, where: dict[str, Any], **args: Any) -> Any:
        value = worker.call(op, **args)
        if repeat:
            again = worker.call(op, **args)
            if canonical(again) != canonical(value):
                failure(
                    "non_deterministic_model", {**where, "callback": op}, outputs=(value, again)
                )
        return value

    def explains(state: Any, obs: Any, kind: str, where: dict[str, Any], repeat: bool) -> bool:
        rendered = check_observation(call("render", repeat, where, state=state))
        if canonical(rendered) != canonical(obs):
            failure(kind, where, rendered, obs)
            return False
        add_state_size(sizes, where["observation_id"], state, obs)
        return True

    def reached(history: str, state: Any, ok: bool = True) -> bool:
        if history in repeated:
            if ok:
                validated[history] = state
            else:
                diverged.add(history)
        return ok

    live: dict[str, Any] | None = None
    try:
        for episode in store.episodes():
            episode_id = episode["episode_id"]
            where = {
                "episode_id": episode_id,
                "step": 0,
                "observation_id": episode["initial_obs_id"],
            }
            if episode["start_hash"] in validated:
                state, alive = validated[episode["start_hash"]], True
            elif episode["start_hash"] in diverged:
                state, alive = None, False
            else:
                start = store.observation(episode["initial_obs_id"])
                state = call("get_initial_state", True, where, obs=start)
                alive = reached(
                    episode["start_hash"],
                    state,
                    explains(state, start, "reconstruction_mismatch", where, True),
                )
            for item in store.episode_steps(episode_id) if alive else []:
                summary["steps_checked"] += 1
                if item["history_hash"] in validated:
                    state = validated[item["history_hash"]]
                    continue
                if item["history_hash"] in diverged:
                    alive = False
                    break
                where = {
                    "episode_id": episode_id,
                    "step": item["step_index"] + 1,
                    "observation_id": item["after_obs_id"],
                    "transition_id": item["transition_id"],
                }
                repeat = item["step_index"] % _REPEAT_EVERY == 0
                state = call("step", repeat, where, state=state, action=item["action"])
                observed = store.observation(item["after_obs_id"])
                ok = explains(state, observed, "prediction_mismatch", where, repeat)
                if not reached(item["history_hash"], state, ok):
                    alive = False  # the rest of this episode cannot be predicted from here
                    break
            if episode_id == live_episode and alive:
                live = {"state": state, "hash": store.last_hash(episode_id)}
            summary["episodes_checked"] += 1
        summary.update({k: sizes[k] for k in ("state_bytes", "observation_bytes") if k in sizes})
        for key in ("smallest_state", "largest_state", "largest_ratio_state"):
            if key in sizes:
                summary[key] = sizes[key]
        summary["state_obs_size_ratio"] = sizes.get("state_bytes", 0) / max(
            1, sizes.get("observation_bytes", 0)
        )
        if size_failure := compactness_failure(sizes, config.threshold_max_state_obs_size_ratio):
            failure("compression_ratio", size_failure)
    except WorkerError as exc:
        summary["error"] = budgets.truncate_error(str(exc), config.feedback.max_error_chars)
        summary["error_type"] = exc.kind
        summary["error_context"] = exc.context
        _finish_counts(summary)
        return summary, None
    _finish_counts(summary)
    summary["complete"] = True
    summary["accepted"] = not summary["failures"]
    return summary, live


def _finish_counts(summary: dict[str, Any]) -> None:
    summary["counterexamples_omitted"] = sum(summary["failures"].values()) - len(
        summary["counterexamples"]
    )
