"""Trusted comparisons: worker outputs are data, never validation verdicts."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from typing import Any

from regact.protocols.cwm import limits as budgets
from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.differences import summarize
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


def diagnostic_pointer(diagnostic_id: int) -> dict[str, Any]:
    """How a counterexample names its diagnostic record."""
    return {
        "diagnostic_id": diagnostic_id,
        "details": f"data_api.describe_diagnostic({diagnostic_id})",
    }


# What a running exploration carries over from its validation to keep checking compactness.
SIZE_KEYS = (
    "state_bytes",
    "observation_bytes",
    "n_states",
    "largest_ratio_state",
    "largest_oversized_state",
)


def add_state_size(
    summary: dict[str, Any],
    oid: int,
    state: Any,
    obs: Any,
    threshold: float,
    allowance: int | None = None,
) -> None:
    """Accumulate bytes and representative extremes without retaining extra states. A State of at
    most ``allowance`` bytes never becomes the largest-ratio State: it always passes. Of the States
    at or above ``threshold`` times their own observation, only the largest is kept: it alone
    decides the per-State check (see compactness_failure)."""
    state_bytes = len(canonical(state).encode())
    observation_bytes = len(canonical(obs).encode())
    summary["state_bytes"] = summary.get("state_bytes", 0) + state_bytes
    summary["observation_bytes"] = summary.get("observation_bytes", 0) + observation_bytes
    summary["n_states"] = summary.get("n_states", 0) + 1
    item = {
        "observation_id": oid,
        "state_bytes": state_bytes,
        "observation_bytes": observation_bytes,
        "ratio": state_bytes / max(1, observation_bytes),
    }
    for key, sign in (("smallest_state", 1), ("largest_state", -1)):
        if key not in summary or sign * state_bytes < sign * summary[key]["state_bytes"]:
            summary[key] = item
    if allowance is not None and state_bytes <= allowance:
        return
    largest = summary.get("largest_ratio_state")
    if largest is None or item["ratio"] > largest["ratio"]:
        summary["largest_ratio_state"] = item
    oversized = summary.get("largest_oversized_state")
    if item["ratio"] >= threshold and (oversized is None or state_bytes > oversized["state_bytes"]):
        summary["largest_oversized_state"] = item


def compactness_failure(
    sizes: dict[str, Any], threshold: float, allowance: int | None = None
) -> dict[str, Any] | None:
    """Both the aggregate ratio and every single State must stay under the threshold, so a State
    that grows along an episode (visited sets, logs) fails even when the average is low. A single
    State is measured against the larger of its own observation and the mean observation: a
    near-empty observation (an ARC game over) must not demand a State smaller than the game needs.
    States of at most ``allowance`` bytes pass whatever the ratio: on games with small
    observations the ratio alone would refuse even a minimal State."""
    ratio = sizes.get("state_bytes", 0) / max(1, sizes.get("observation_bytes", 0))
    small = allowance is not None and sizes.get("state_bytes", 0) <= allowance * max(
        1, sizes.get("n_states", 0)
    )
    if ratio >= threshold and not small:
        return {"ratio": ratio, "required_below": threshold}
    oversized = sizes.get("largest_oversized_state")
    mean = sizes.get("observation_bytes", 0) / max(1, sizes.get("n_states", 0))
    if oversized is not None and oversized["state_bytes"] >= threshold * mean:
        return {
            "largest_state": oversized,
            "mean_observation_bytes": round(mean),
            "required_below": threshold,
        }
    return None


_REPEAT_EVERY = 10  # repeatability is checked on episode starts and every 10th step
_SURPRISES = 5  # most expensive steps listed in a loss report
_BUCKETS = (0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0)  # upper edges of the calibration buckets
_ALPHABET = 64  # values a component is assumed able to take, for the no-model reference


def advance(
    worker: Worker, explainer: Worker | None, state: Any, action: Any, obs: Any, config: CwmConfig
) -> dict[str, Any]:
    """Explain one real step with a stochastic CWM: the cheapest values of step's random choices
    under which render gives ``obs``. ``explainer`` is a second process running the same model,
    so what ``explain`` reads in the observation reaches ``step`` only as choice values, each
    paid for. The search runs here: the process running ``step`` never sees ``obs``."""
    pins = explainer.call("explain", state=state, action=action, obs=obs) if explainer else {}
    target = canonical(obs)
    queue: list[tuple[float, int, list[Any]]] = [(0.0, 0, [])]
    nodes, first, tie = 0, None, 0
    while queue and not budgets.reached(nodes, config.max_nodes_per_explanation):
        _, _, prefix = heapq.heappop(queue)
        run = worker.call("run_step", state=state, action=action, pins=pins, prefix=prefix)
        nodes += 1
        first = first or run
        if run["possible"]:
            rendered = worker.call("render", state=run["state"])
            if canonical(rendered) == target:
                bits = sum(-math.log2(p) for _, _, p, _, _ in run["choices"])
                return {**run, "explained": True, "bits": bits, "nodes": nodes}
        cost, k = 0.0, 0
        for _, value, p, pinned, ranked in run["choices"]:
            if pinned:
                continue
            if k >= len(prefix):
                taken = [v for _, v, _, pin, _ in run["choices"] if not pin][:k]
                for other, q in ranked:
                    if other != value:
                        tie += 1
                        heapq.heappush(queue, (cost - math.log2(q), tie, [*taken, other]))
            cost -= math.log2(p)
            k += 1
    assert first is not None
    return {
        **first,
        "explained": False,
        "bits": None,
        "nodes": nodes,
        "exhausted": not queue,
        "impossible": [name for name, _, p, _, _ in first["choices"] if p == 0],
    }


def choice_group(name: str) -> str:
    """`cow@3,4` and `cow@5,1` are two uses of the choice `cow`."""
    return name.split("@", 1)[0]


class Loss:
    """Bits paid by a stochastic CWM for its explanations, per choice group, with two references
    computed here: the same explanations priced with frequencies learned online (KT estimator),
    and a model-free coder predicting each observation component from its previous value."""

    def __init__(self) -> None:
        self.steps = 0
        self.total = 0.0
        self.adaptive = 0.0
        self.no_model = 0.0
        self.groups: dict[str, dict[str, Any]] = {}
        self.surprises: list[dict[str, Any]] = []
        self.calibration = [[0, 0.0, 0] for _ in _BUCKETS]  # options, expected, happened
        self._contexts: dict[tuple[str, str], dict[str, int]] = {}

    def add(self, result: dict[str, Any], where: dict[str, Any], before: Any, after: Any) -> None:
        self.steps += 1
        self.total += result["bits"]
        worst = None
        for name, value, p, _pinned, ranked in result["choices"]:
            for other, q in ranked:  # reliability: do things given probability q happen q of the time?
                bucket = self.calibration[min(len(_BUCKETS) - 1, sum(q > edge for edge in _BUCKETS[:-1]))]
                bucket[0] += 1
                bucket[1] += q
                bucket[2] += other == value
            group = self.groups.setdefault(
                choice_group(name), {"uses": 0, "bits": 0.0, "values": {}, "declared": {}}
            )
            key = canonical(value)
            seen, n = group["values"].get(key, 0), group["uses"]
            k = max(2, len(group["values"]) + (seen == 0))
            self.adaptive -= math.log2((seen + 0.5) / (n + k / 2))
            group["uses"] += 1
            group["bits"] -= math.log2(p)
            group["values"][key] = seen + 1
            group["declared"][key] = group["declared"].get(key, 0.0) + p
            if worst is None or p < worst[1]:
                worst = (name, p)
        self.no_model += self.no_model_bits(before, after)
        if worst is not None:
            self.surprises.append(
                {
                    "episode_id": where["episode_id"],
                    "step": where["step"],
                    "bits": round(result["bits"], 2),
                    "least_likely_choice": worst[0],
                    "probability": round(worst[1], 5),
                }
            )
            self.surprises = sorted(self.surprises, key=lambda item: -item["bits"])[:_SURPRISES]

    def no_model_bits(self, before: Any, after: Any) -> float:
        """What the model-free coder pays for one step, learning from it as it goes."""
        old, bits = dict(_leaves(before)), 0.0
        for path, value in _leaves(after):
            counts = self._contexts.setdefault((path, canonical(old.get(path))), {})
            key, n = canonical(value), sum(counts.values())
            bits -= math.log2((counts.get(key, 0) + 0.5) / (n + _ALPHABET / 2))
            counts[key] = counts.get(key, 0) + 1
        return bits

    def report(self) -> dict[str, Any]:
        steps = max(1, self.steps)
        groups = []
        for name, g in sorted(self.groups.items(), key=lambda item: -item[1]["bits"]):
            n = g["uses"]
            entropy = -sum(c / n * math.log2(c / n) for c in g["values"].values())
            groups.append(
                {
                    "choice": name,
                    "uses": n,
                    "bits": round(g["bits"], 1),
                    "excess_bits": round(g["bits"] - n * entropy, 1),
                    # for each value: how often it happened, and the probability it had then
                    "observed": _top({k: c / n for k, c in g["values"].items()}),
                    "declared": {
                        k: round(g["declared"][k] / g["values"][k], 3)
                        for k in _top({k: c / n for k, c in g["values"].items()})
                    },
                }
            )
        return {
            "steps": self.steps,
            "total": round(self.total, 1),
            "per_step": round(self.total / steps, 3),
            "references_per_step": {
                "adaptive_frequencies": round(self.adaptive / steps, 3),
                "no_model": round(self.no_model / steps, 3),
            },
            "calibration": [
                {
                    "declared_up_to": edge,
                    "options": n,
                    "expected_to_happen": round(expected, 1),
                    "happened": happened,
                }
                for edge, (n, expected, happened) in zip(_BUCKETS, self.calibration, strict=True)
                if n
            ],
            "by_choice": groups,
            "largest_surprises": self.surprises,
        }


def _top(shares: dict[str, float], n: int = 6) -> dict[str, float]:
    ranked = sorted(shares.items(), key=lambda item: -item[1])
    return {key: round(value, 3) for key, value in ranked[:n]}


def _leaves(value: Any, path: str = "") -> Any:
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _leaves(value[key], f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _leaves(item, f"{path}[{i}]")
    else:
        yield path, value


def validate(
    worker: Worker,
    store: ExperienceStore,
    config: CwmConfig,
    *,
    live_episode: int | None = None,
    progress: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    explained_until: int | None = None,
    explainer: Worker | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Replay every recorded episode in time order: get_initial_state on its first observation,
    then step carries the State, and render is checked against every recorded observation.

    ``progress`` names the problem's progress at an observation, for counterexamples.
    ``explained_until`` is the last step event the previously accepted version predicted.

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
    loss = Loss() if config.stochastic else None
    # a stochastic State depends on what happened before, not only on the episode's own history
    repeated = set() if config.stochastic else store.repeated_histories()
    previous_final: Any = None  # the State at the end of the previous episode, for reset()
    validated: dict[str, Any] = {}  # repeated history hash -> State there, validated once
    diverged: set[str] = set()  # repeated history hashes where a failure is already reported

    def failure(
        kind: str,
        data: dict[str, Any],
        predicted: Any = None,
        actual: Any = None,
        outputs: tuple[Any, Any] | None = None,
        states: dict[str, Any] | None = None,
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
        if progress is not None and "from_observation_id" in data:
            entry.update(progress(store.observation(data["from_observation_id"])))
        diagnostic = None
        if predicted is not None:
            entry["differences"] = summarize(predicted, actual, config.feedback)
            diagnostic = store.diagnostic(
                {**entry, "predicted": predicted, "observed": actual, **(states or {})}
            )
        elif outputs is not None:
            diagnostic = store.diagnostic(
                {**entry, "first_output": outputs[0], "second_output": outputs[1]}
            )
        if diagnostic is not None:
            entry.update(diagnostic_pointer(diagnostic))
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

    def explains(
        state: Any, obs: Any, kind: str, where: dict[str, Any], repeat: bool, before: Any = None
    ) -> bool:
        rendered = check_observation(call("render", repeat, where, state=state))
        if canonical(rendered) != canonical(obs):
            failure(kind, where, rendered, obs, states={"state_before": before, "state": state})
            return False
        add_state_size(
            sizes,
            where["observation_id"],
            state,
            obs,
            config.threshold_max_state_obs_size_ratio,
            config.max_state_bytes_always_allowed,
        )
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
                if config.stochastic:
                    state = worker.call("start_state", previous=previous_final, obs=start)
                else:
                    state = call("get_initial_state", True, where, obs=start)
                alive = reached(
                    episode["start_hash"],
                    state,
                    explains(state, start, "reconstruction_mismatch", where, True),
                )
            before = episode["initial_obs_id"]
            for item in store.episode_steps(episode_id) if alive else []:
                summary["steps_checked"] += 1
                if item["history_hash"] in validated:
                    state = validated[item["history_hash"]]
                    before = item["after_obs_id"]
                    continue
                if item["history_hash"] in diverged:
                    alive = False
                    break
                where = {
                    "episode_id": episode_id,
                    "step": item["step_index"] + 1,
                    "from_observation_id": before,
                    "action": item["action"],
                    "observation_id": item["after_obs_id"],
                    "transition_id": item["transition_id"],
                }
                if explained_until is not None:
                    where["explained_by_previous_version"] = (
                        item["step_event_id"] <= explained_until
                    )
                before = item["after_obs_id"]
                repeat = item["step_index"] % _REPEAT_EVERY == 0
                before_state = state
                observed = store.observation(item["after_obs_id"])
                if loss is None:
                    state = call("step", repeat, where, state=state, action=item["action"])
                else:
                    advanced = advance(worker, explainer, state, item["action"], observed, config)
                    state = advanced["state"]
                    where = {**where, **explanation_note(advanced)}
                ok = explains(state, observed, "prediction_mismatch", where, repeat, before_state)
                if ok and loss is not None and not advanced["explained"]:
                    # the right observation, through a choice the model gives probability 0
                    states = {"state_before": before_state, "state": state}
                    failure("prediction_mismatch", where, observed, observed, states=states)
                    ok = False
                if ok and loss is not None:
                    loss.add(advanced, where, store.observation(where["from_observation_id"]), observed)
                if not reached(item["history_hash"], state, ok):
                    alive = False  # the rest of this episode cannot be predicted from here
                    break
            if episode_id == live_episode and alive:
                live = {"state": state, "hash": store.last_hash(episode_id)}
            previous_final = state if alive else None
            summary["episodes_checked"] += 1
        for key in (*SIZE_KEYS, "smallest_state", "largest_state"):
            if key in sizes:
                summary[key] = sizes[key]
        if loss is not None:
            summary["bits"] = loss.report()
        summary["state_obs_size_ratio"] = sizes.get("state_bytes", 0) / max(
            1, sizes.get("observation_bytes", 0)
        )
        if size_failure := compactness_failure(
            sizes, config.threshold_max_state_obs_size_ratio, config.max_state_bytes_always_allowed
        ):
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


def explanation_note(advanced: dict[str, Any]) -> dict[str, Any]:
    """Why a stochastic step was not explained, for its counterexample."""
    if advanced["explained"]:
        return {}
    note: dict[str, Any] = {"explanation_search_steps": advanced["nodes"]}
    if advanced.get("impossible"):
        note["choices_given_probability_0"] = advanced["impossible"][:10]
    if not advanced.get("exhausted"):
        note["search_budget_reached"] = True
    return {"explanation": note}


def _finish_counts(summary: dict[str, Any]) -> None:
    summary["counterexamples_omitted"] = sum(summary["failures"].values()) - len(
        summary["counterexamples"]
    )
