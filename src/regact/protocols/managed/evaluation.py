"""Frozen evaluation: when a task ends, its final controller plays one episode on each of the
problem's held-out tasks (``BaseProblem.evaluation_tasks``), with the accepted CWM and no agent.
The final controller is ``final_controller.py`` or, without one, the controller of the last
RunController call. Nothing here is shown to the agent or recorded as experience."""

from __future__ import annotations

import time
from contextlib import ExitStack
from typing import Any

from regact.env.session import EnvSession
from regact.envclient.errors import InvalidActionError
from regact.protocols.cwm import limits as budgets
from regact.protocols.cwm.bundle import snapshot
from regact.protocols.cwm.store import atomic_json, canonical
from regact.protocols.cwm.validation import advance, check_observation
from regact.protocols.cwm.worker import WorkerError

# Exits after which the task may still be resumed or its recorded state is not trusted.
SKIPPED_EXITS = (
    "interrupted",
    "dry_run",
    "environment_or_storage_failure",
    "framework_failure",
    "observation_determinism_violation",
    "initial_observation_mismatch",
)


FINAL_CONTROLLER = "final_controller.py"


class _Unexplained(Exception):
    pass


def _final_controller(coordinator: Any) -> tuple[Any, Any, str | None]:
    """The bundle to run, the CWM bundle beside it, and where the controller came from."""
    bundles = coordinator.root / "bundles"
    accepted = coordinator.accepted
    if (coordinator.workdir / FINAL_CONTROLLER).is_file() and (
        accepted or not coordinator.uses_model
    ):
        model = bundles / accepted["bundle"] if accepted else None
        try:
            return (
                snapshot(coordinator.workdir, bundles, [FINAL_CONTROLLER], model=model)[0],
                model,
                FINAL_CONTROLLER,
            )
        except (OSError, ValueError, SyntaxError):
            pass  # unreadable: the last controller that ran stands in
    record = coordinator.store.last_record("exploration", "completed") or {}
    if not record.get("bundle") or (coordinator.uses_model and not record.get("model_bundle")):
        return None, None, None
    model = bundles / record["model_bundle"] if record.get("model_bundle") else None
    return bundles / record["bundle"], model, "last RunController call"


def evaluate(coordinator: Any) -> dict[str, Any] | None:
    """Play the held-out episodes and write ``cwm/evaluation.json``; None when the problem has
    none. A task that never ran a controller scores its episodes as empty."""
    problem = coordinator.problem
    tasks = problem.evaluation_tasks(coordinator.task)
    if not tasks:
        return None
    controller, model, source = _final_controller(coordinator)
    coordinator.event("evaluation_started", tasks=tasks, controller=source)
    module = "final_controller" if source == FINAL_CONTROLLER else "controller"
    episodes = []
    for name in tasks:
        if controller is None:
            episodes.append(
                {"task": name, "stop_reason": "no_controller", "actions": 0, "info": {}}
            )
        else:
            episodes.append(_episode(coordinator, name, controller, model, module))
    result = {
        "controller": source,
        "controller_bundle": controller.name if controller else None,
        "model_bundle": model.name if model else None,
        "max_actions_per_episode": problem.evaluation_max_actions(),
        "metrics": problem.evaluation_metrics(episodes),
        "episodes": episodes,
    }
    atomic_json(coordinator.root / "evaluation.json", result)
    coordinator.event("evaluation_finished", metrics=result["metrics"])
    return result


def _episode(
    coordinator: Any, name: str, controller_bundle: Any, model: Any, module: str
) -> dict[str, Any]:
    from regact.orchestration.task import _lifecycle_policy

    config, options, problem = coordinator.config, coordinator.options, coordinator.problem
    stochastic = bool(getattr(options, "stochastic", False))
    cap = problem.evaluation_max_actions()
    # submitted code gets the seconds per action a RunController call grants
    call_seconds, call_actions = (
        options.execution.max_seconds_per_RunController,
        options.max_actions_per_RunController,
    )
    seconds = (
        call_seconds * max(1.0, cap / call_actions)
        if call_seconds and call_actions and cap
        else None
    )
    clock = budgets.AgentClock(seconds)
    session = EnvSession(
        make_native=lambda: problem.make_env(name),
        key=name,
        renderer=problem.obs_renderer(name, mode=config.problem.obs_mode),
        lifecycle=_lifecycle_policy(config.problem.lifecycle),
        milestone_detector=problem.milestone_detector(name),
    )
    out: dict[str, Any] = {"task": name, "actions": 0, "action_list": [], "info": {}}
    regrounded, bits, explained = 0, 0.0, 0
    started = time.monotonic()
    try:
        obs = session.first_obs().to_json()
        out["info"] = obs["info"]
        with ExitStack() as stack:

            def open_worker(bundle: Any) -> Any:
                return stack.enter_context(
                    coordinator.worker(bundle, seconds, "per evaluation episode", clock=clock)
                )

            worker = open_worker(model) if model is not None else None
            explainer = open_worker(model) if model is not None and stochastic else None
            controller = open_worker(controller_bundle)

            def ground(obs: dict[str, Any]) -> Any:
                """A State built from one observation alone, as at the start of an episode."""
                if worker is None:
                    return obs
                op = "start_state" if stochastic else "get_initial_state"
                state = worker.call(op, obs=obs)
                if canonical(check_observation(worker.call("render", state=state))) != canonical(
                    obs
                ):
                    raise _Unexplained
                return state

            def follow(state: Any, action: Any, obs: dict[str, Any]) -> Any:
                nonlocal regrounded, bits, explained
                if worker is None:
                    return obs
                try:
                    if stochastic:
                        advanced = advance(worker, explainer, state, action, obs, options)
                        if advanced["explained"]:
                            bits, explained = bits + advanced["bits"], explained + 1
                            return advanced["state"]
                    else:
                        after = worker.call("step", state=state, action=action)
                        rendered = check_observation(worker.call("render", state=after))
                        if canonical(rendered) == canonical(obs):
                            return after
                except WorkerError as exc:
                    if exc.kind == "operation_timeout":
                        raise
                # nobody can repair the model here: forget and carry on from what is visible
                regrounded += 1
                return ground(obs)

            controller.call("controller_init", module=module)
            state = ground(obs)
            reason = "max_actions"
            for _ in budgets.action_indices(cap):
                if obs["is_done"]:
                    reason = "environment_done"
                    break
                if controller.call("is_done", state=state):
                    reason = "controller_done"
                    break
                action = controller.call("act", state=state)
                problem.validate_controller_action(action)
                assert session.live is not None
                obs = session.live.step(action).to_json()
                out["actions"] += 1
                out["action_list"].append(action)
                out["info"] = obs["info"]
                state = follow(state, action, obs)
            out["stop_reason"] = reason
    except _Unexplained:
        out["stop_reason"] = "model_cannot_represent_observation"
    except (WorkerError, InvalidActionError) as exc:
        timeout = getattr(exc, "kind", None) == "operation_timeout"
        out["stop_reason"] = "time_limit" if timeout else "code_error"
        out["error"] = budgets.truncate_error(str(exc), options.feedback.max_error_chars)
        out["error_context"] = dict(getattr(exc, "context", {}))
    finally:
        session.close()
    out["seconds"] = round(time.monotonic() - started, 2)
    out["submitted_code_seconds"] = round(clock.used, 2)
    if model is not None:
        out["regrounded"] = regrounded
        if explained:
            out["bits_per_step"] = round(bits / explained, 3)
    return out
