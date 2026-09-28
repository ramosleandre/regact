"""Agent-facing CWM feedback. Full execution/provenance records stay in the store."""

from __future__ import annotations

import json
from typing import Any

from regact.config.schema import LimitsConfig
from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.ids import format_id_ranges

FAILURES = {
    "reconstruction_mismatch": "parse then render does not reproduce a recorded observation",
    "prediction_mismatch": "the predicted next observation differs from recorded reality",
    "parser_collision": "distinct observations map to the same state",
    "compression_ratio": "the serialized states are too large",
    "non_deterministic_model": "a callback returned different outputs for identical input",
}
ENDINGS = {
    "solved": "The full game is solved; the task has ended.",
    "interrupted": "The task was interrupted; no further work will run.",
    "observation_determinism_violation": (
        "The same recorded observation and action produced different real "
        "outcomes. This protocol requires observation-deterministic dynamics; the"
        " task has stopped."
    ),
    "initial_observation_mismatch": (
        "A fresh reset did not reproduce the fixed starting observation; the task has stopped."
    ),
    "environment_or_storage_failure": (
        "The environment or experience recording failed. History may be "
        "incomplete; the task has stopped to avoid repeating uncertain actions."
    ),
    "framework_failure": "A framework failure stopped the task. This is not a controller failure.",
    "initial_collection_failure": "Dataset preparation failed before the agent could start.",
}
REASONS = {
    "plan_exhausted": "The action list finished.",
    "controller_done": "The controller requested the end of this exploration.",
    "goal_achieved": "The controller's goal predicate is true in the real environment.",
    "environment_done": "The real environment ended the episode.",
    "no_predicted_novelty": (
        "Simulating the exploration controller in the CWM found no new observations "
        "outside the current dataset. Produce an exploration controller that explores "
        "new and relevant states. No real episode was started."
    ),
    "prediction_mismatch": (
        "Reality contradicted a predicted observation. Inspect the counterexample."
    ),
    "reconstruction_mismatch": (
        "The CWM cannot reconstruct a newly observed observation. Inspect the counterexample."
    ),
    "parser_collision": (
        "A new observation shares its state with a different observation. "
        "Preserve the missing distinction."
    ),
    "compression_ratio": (
        "The state-size ratio exceeded the acceptance threshold on new experience."
    ),
    "controller_error": (
        "The controller failed. Recorded experience is retained and this "
        "exploration receives worst metrics; repair exploration.py."
    ),
    "model_error": (
        "The CWM failed while processing real experience. Inspect the error and validate a repair."
    ),
    "episode_time_limit": (
        "The real episode ran out of time before all checks finished. Experience "
        "is retained and must be validated. This does not by itself establish a CWM contradiction."
    ),
}


def format_feedback(value: dict[str, Any]) -> str:
    """Readable tool text, shared by native delivery and the terminal bridge."""
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)


def budget(reason: str, c: CwmConfig, limits: LimitsConfig) -> dict[str, Any] | None:
    mapping = {
        "max_actions_per_exploration": (
            "protocol.max_actions_per_exploration",
            c.max_actions_per_exploration,
            "actions per episode; simulation and real have separate allowances",
        ),
        "max_actions_per_episode": (
            "limits.max_actions_per_episode",
            limits.max_actions_per_episode,
            "real actions per episode",
        ),
        "real_action_limit": (
            "limits.max_actions_per_task",
            limits.max_actions_per_task,
            "real actions across the task, including dataset preparation",
        ),
        "walltime_limit": (
            "limits.max_seconds_per_task",
            limits.max_seconds_per_task,
            "seconds per task",
        ),
        "episode_time_limit": (
            "protocol.execution.max_seconds_per_episode",
            c.execution.max_seconds_per_episode,
            "seconds per episode; simulation and real have separate allowances",
        ),
        "search_memory_limit": (
            "protocol.execution.max_memory_mb",
            c.execution.max_memory_mb / 4 if c.execution.max_memory_mb is not None else None,
            "MiB of serialized search records (one quarter of max_memory_mb)",
        ),
    }
    if reason.startswith("max_") and hasattr(c.planner, reason):
        value = getattr(c.planner, reason)
        unit = (
            "seconds"
            if "seconds" in reason
            else "CWM callbacks"
            if "calls" in reason
            else "states"
            if "nodes" in reason
            else "actions of search depth"
        )
        mapping[reason] = ("protocol.planner." + reason, value, unit + " per planner call")
    if reason not in mapping:
        return None
    key, value, unit = mapping[reason]
    if reason == "search_memory_limit":
        return {
            "parameter": key,
            "value": c.execution.max_memory_mb,
            "unit": "MiB per isolated Python process",
            "search_queue_max_bytes": c.execution.max_memory_mb * 1024 * 1024 // 4
            if c.execution.max_memory_mb is not None
            else None,
        }
    return {"parameter": key, "value": value, "unit": unit}


def present(name: str, r: dict[str, Any], c: CwmConfig, limits: LimitsConfig) -> dict[str, Any]:
    """Project trusted results into the small public interface; never mutate stored records."""
    out: dict[str, Any]
    error = r.get("error")
    if name == "UpdateCodeWorldModel":
        status = (
            "Accepted" if r.get("accepted") else "Refused" if r.get("complete") else "Incomplete"
        )
        message = (
            "CWM accepted against the current dataset."
            if status == "Accepted"
            else (
                "CWM refused: " + "; ".join(FAILURES.get(k, k) for k in r.get("failures", {})) + "."
                if status == "Refused"
                else "Validation did not finish. Correct the reported problem and submit again."
            )
        )
        out = {
            "status": status,
            "message": message,
            "checked": {
                "observations": r.get("observations_checked", 0),
                "transitions": r.get("transitions_checked", 0),
            },
        }
        if r.get("cwm_version") is not None:
            out["cwm_version" if status == "Accepted" else "previously_accepted_cwm_version"] = r[
                "cwm_version"
            ]
        if r.get("state_obs_size_ratio") is not None:
            out["state_size"] = {
                "state_bytes": r["state_bytes"],
                "observation_bytes": r["observation_bytes"],
                "ratio": r["state_obs_size_ratio"],
                "required_below": c.threshold_max_state_obs_size_ratio,
            }
            if r["state_obs_size_ratio"] >= c.threshold_max_state_obs_size_ratio:
                for key in ("smallest_state", "largest_state"):
                    if key in r:
                        out["state_size"][key] = r[key]
        for key in ("failures", "counterexamples", "counterexamples_omitted"):
            if r.get(key):
                out[key] = r[key]
    elif name == "PlanInCWM":
        found = r.get("candidate_found", False)
        out = {"status": "Incomplete" if error else "Plan found" if found else "No plan found"}
        out["message"] = (
            "Planning failed; inspect the error."
            if error
            else (
                f"Planner found a plan as a {r['n_actions']}-length action list "
                f"in the Code World Model. Saved at {r.get('path')}. "
                "Submit an ExplorationControllerFromListActions instance using this plan "
                "with SubmitExplorationController to apply it in the real environment."
            )
            if found
            else (
                "No path with a new predicted observation was found within this search. "
                "Try another goal or a direct exploration controller."
            )
        )
        for key in ("path", "cwm_version", "elapsed_seconds", "n_states_searched"):
            if key in r:
                out[key] = round(r[key], 5) if key == "elapsed_seconds" else r[key]
        if found:
            out.update(
                goal_achieved=r["achieved"],
                utility=r["utility"],
                optimality_proven=r["optimality_proven"],
            )
            if not r["achieved"]:
                out["message"] += (
                    " Warning: this is a partial plan; it does not reach the requested goal."
                )
            if not r["optimality_proven"]:
                out["message"] += " Warning: search stopped before optimality could be established."
        if r.get("search_stop_reason"):
            out["search_stop_reason"] = r["search_stop_reason"]
    else:
        reason = r.get("stop_reason", "")
        out = {
            "status": "Incomplete"
            if error
            else "Refused"
            if reason == "no_predicted_novelty"
            else "Completed",
            "message": REASONS.get(
                reason,
                "Exploration finished." if not error else "Exploration failed; inspect the error.",
            ),
        }
        if reason:
            out["stop_reason"] = reason
        for key in (
            "exploration_id",
            "cwm_version",
            "simulation_actions",
            "real_actions",
            "predicted_novel_observations",
            "actual_novel_observations",
            "episode_id",
            "diagnostic_id",
            "counterexample",
            "differences",
            "differences_omitted",
            "new_milestones",
            "stage",
        ):
            if key in r and (key != "new_milestones" or r[key]):
                out[key] = r[key]
        if r.get("new_milestones"):
            out["message"] = (
                "New real milestones: "
                + "; ".join(f"{m['name']} ({m['kind']})" for m in r["new_milestones"])
                + ". "
                + out["message"]
            )
        for key in ("observation_ids", "transition_ids"):
            if key in r:
                out[key] = format_id_ranges(r[key])
        if r.get("image_observation_ids"):
            out["observation_images"] = "tmp/images/obs_id_<n>.png for <n> in " + format_id_ranges(
                r["image_observation_ids"]
            )
        if r.get("image_preview_error"):
            out["image_preview_error"] = r["image_preview_error"]
        if r.get("objective_reached") is not None:
            out["goal_achieved"] = r["objective_reached"]
        if "metrics" in r:
            out["metrics"] = r["metrics"]
        if r.get("history_complete") is False:
            out["history_complete"] = False
            out["real_actions_recorded"] = out.pop("real_actions", 0)
            out.pop("metrics", None)
            out["message"] = (
                "History is incomplete. The recorded action count may undercount actions "
                "that actually happened. The task has stopped; do not retry real "
                "execution."
            )
    if error:
        callback = r.get("error_context", {}).get("callback")
        if r.get("error_type") == "InvalidActionError":
            error = (
                "Invalid action from the exploration controller. The real environment rejected it: "
                + error
            )
        elif callback in ("parse", "render", "step"):
            error = f"CWM {callback} failed: {error}"
        elif callback in ("act", "is_done", "controller_init"):
            error = f"Exploration controller {callback} failed: {error}"
        out["error"] = {
            "type": "execution_stopped"
            if r.get("error_type") == "worker_exit"
            else r.get("error_type", "error"),
            "message": error,
            **r.get("error_context", {}),
        }
        if r.get("diagnostic_id"):
            out["diagnostic_id"] = r["diagnostic_id"]
    reason = r.get("search_stop_reason") or r.get("stop_reason") or ""
    if limit := budget(reason, c, limits):
        out["budget"] = limit
    if r.get("exit_reason"):
        reason = r["exit_reason"]
        out["task_stop"] = {
            "reason": reason,
            "message": ENDINGS.get(reason, "The task has ended: " + reason.replace("_", " ") + "."),
        }
        if limit := budget(reason, c, limits):
            out["task_stop"]["budget"] = limit
        if reason == "interrupted":
            out["message"] = "Operation interrupted; partial results are not an acceptance verdict."
    if r.get("error_type") == "request_conflict":
        # Retry identifiers and conflicts belong to transport logs, not agent work.
        out = {
            "status": "Incomplete",
            "message": "The command could not be started because of an internal delivery error. No new execution occurred.",
        }
    return out


def render_feedback(name: str, r: dict[str, Any], c: CwmConfig, limits: LimitsConfig) -> str:
    """One completed result, with simulation screening and real execution clearly separated."""
    out = present(name, r, c, limits)
    if name != "SubmitExplorationController" or r.get("stage") not in ("simulation", "real"):
        return format_feedback(out)
    out.pop("stage", None)
    if r["stage"] == "simulation":
        for key in (
            "real_actions",
            "actual_novel_observations",
            "metrics",
            "new_milestones",
            "goal_achieved",
        ):
            out.pop(key, None)
        return "Running in Code World Model... Failure. Stopping here.\n" + format_feedback(out)
    for key in ("simulation_actions", "predicted_novel_observations"):
        out.pop(key, None)
    novelty = r.get("predicted_novel_observations", 0)
    observation_word = "observation" if novelty == 1 else "observations"
    ending = "Stopped." if r.get("error") else "Done."
    return (
        f"Running in Code World Model... Success. Found {novelty} new {observation_word}.\n\n"
        f"Running in actual env... {ending}\n" + format_feedback(out)
    )
