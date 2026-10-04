"""One trusted CWM coordinator per task: phases, real actions, and immutable versions."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import random
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from regact.config.schema import Lifecycle, RunConfig
from regact.env.session import EnvSession
from regact.envclient.errors import InvalidActionError
from regact.envclient.obs import Obs
from regact.obs.errors import LogComponent
from regact.orchestration.signals import StopSignal
from regact.problems.base import BaseProblem
from regact.protocols.base import ProtocolContext, ProtocolSession
from regact.protocols.cwm import limits as budgets
from regact.protocols.cwm.bundle import description, snapshot
from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.ids import expand_id_ranges
from regact.protocols.cwm.images import clear_images, save_preview, select_preview_ids
from regact.protocols.cwm.store import ExperienceStore, atomic_json, canonical, digest
from regact.protocols.cwm.validation import differences
from regact.protocols.cwm.worker import Worker, WorkerError
from regact.protocols.managed.prompting import reset_commands
from regact.security.runtime import SandboxRuntime
from regact.tools.base import Tool, ToolContext, ToolOutput

MODELING = "CWM Modeling"
EXPLORATION = "Active Exploration"

MODEL_FILES = [
    f"world_model/model_{part}.py" for part in ("state", "parser", "render", "transition")
]


class TaskStopped(Exception):
    """A known task limit reached before an environment action; history stays complete."""


class ManagedCoordinator:
    """Own the live environment and execute fresh isolated controllers.

    Subclasses specialize validation, prediction and available commands. Vanilla
    uses observations directly; CWM adds model checks through the same loop.
    """

    protocol_name = "vanilla"
    initial_phase = EXPLORATION
    uses_model = False

    def __init__(
        self,
        config: RunConfig,
        options: CwmConfig,
        env: EnvSession,
        problem: BaseProblem,
        task: str,
        output: Path,
        workdir: Path,
    ) -> None:
        self.config, self.options, self.env, self.problem, self.task = (
            config,
            options,
            env,
            problem,
            task,
        )
        self.output, self.workdir = output, workdir
        self.root = output / "cwm"
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "experience.sqlite3").exists():
            raise RuntimeError(
                "CWM task directory already has experience; automatic resume is not supported"
            )
        self.store = ExperienceStore(self.root / "experience.sqlite3")
        self.lock = threading.RLock()
        self.phase = self.initial_phase
        self.phase_changes: list[dict[str, str]] = []  # this command's, reported in its result
        self._seen_milestones: set[str] = set()
        self.milestones: list[dict[str, Any]] = []
        self.terminal: str | None = None
        self.deadline = float("inf")
        self.step_timings: dict[str, float] = {}  # per RunController call; see _note_timing
        self.context: ProtocolContext | None = None
        self.accepted: dict[str, Any] | None = None
        self.states: dict[int, Any] = {}
        self.latest: dict[str, Any] | None = None
        self.best: dict[str, Any] | None = None
        self.closed = False
        self.episode: int | None = None
        self.active_request: int | None = None
        self.initial: dict[str, Any] = {}
        self.initial_id = 0
        self.initial_collection: dict[str, Any] | None = None
        self.reset_actions = 0
        try:
            self._reset("initial_collection")
        except BaseException:
            self.env.close()
            self.store.close()
            raise
        assert self.env.live is not None and self.env.live.last_obs is not None
        self.initial = self.env.live.last_obs.to_json()
        self.initial_id = self.current_id
        self.persist()

    def event(self, kind: str, **payload: Any) -> None:
        self.store.event(kind, self.phase, payload)
        if self.context is not None:
            self.context.logger.log(
                LogComponent.ORCHESTRATOR, "INFO", kind, phase=str(self.phase), **payload
            )

    def phase_description(self, phase: str | None = None) -> str:
        phase = self.phase if phase is None else phase
        text = (
            "Inspect the current dataset, build or repair world_model/, then validate it with UpdateCodeWorldModel."
            if phase == MODELING
            else "Produce a controller in controller.py aiming to explore new and relevant states, then run it with RunController."
        )
        if phase == EXPLORATION and "PlanInCWM" in self.commands:
            text += " Optionally define a goal in goal.py and use PlanInCWM first."
        return text

    def change_phase(self, phase: str, reason: str, **evidence: Any) -> None:
        if phase == self.phase:
            return
        before, self.phase = self.phase, phase
        self.event("phase_changed", before=before, after=phase, reason=reason, **evidence)
        self.phase_changes.append(
            {"from": before, "to": phase, "next_step": self.phase_description(phase)}
        )

    def persist(self) -> None:
        atomic_json(
            self.root / "status.json",
            {
                "protocol": self.protocol_name,
                "phase": self.phase,
                "exit_reason": self.terminal,
                "initial_observation_id": self.initial_id,
                "current_observation_id": self.current_id,
                "episode_id": self.episode,
                "reset_actions": self.reset_actions,
                "initial_collection": self.initial_collection,
                "milestones": self.milestones,
                **self.store.summary(),
                "accepted_cwm": self.accepted,
                "latest_exploration": self.latest,
                "best_exploration": self.best,
                "config": dataclasses.asdict(self.options),
            },
        )

    def _limit(self) -> str | None:
        if self.terminal:
            return self.terminal
        if time.monotonic() >= self.deadline:
            self.terminal = "walltime_limit"
        budget = self.config.limits.max_actions_per_task
        if (
            budget is not None
            and self.store.summary()["n_total_transitions"] + self.reset_actions >= budget
        ):
            self.terminal = self.terminal or "real_action_limit"
        return self.terminal

    def collect_initial(self, stop: StopSignal | None = None) -> None:
        """Trusted, bounded random prefill. No agent runs during this operation."""
        with self.lock:
            if self.initial_collection is not None:
                return
            started = time.monotonic()
            # Runs before the session clock starts (on_start), so only the task's budget from now
            # and the experiment deadline bound it.
            budget = self.config.limits.seconds_left()
            if budget is not None:
                self.deadline = min(self.deadline, started + budget)
            seed = self.config.problem.seed if self.config.problem.seed is not None else 0
            rng = random.Random(seed)
            deadline = min(
                self.deadline,
                budgets.deadline(self.options.max_seconds_per_initial_collection, start=started),
            )
            steps = 0
            reason = "target_reached"
            self.event("initial_collection_started", seed=seed, policy="uniform_complete_actions")
            try:
                while (
                    self.store.summary()["n_unique_observations"]
                    < self.options.n_unique_observations_in_initial_collection
                ):
                    if stop is not None and stop.is_set():
                        self.terminal = "interrupted"
                    if self._limit():
                        reason = self.terminal or "interrupted"
                        break
                    if budgets.reached(steps, self.options.max_actions_per_initial_collection):
                        reason = "action_cap"
                        break
                    if time.monotonic() >= deadline:
                        reason = "time_cap"
                        break
                    assert self.env.live is not None and self.env.live.last_obs is not None
                    cap = self.config.limits.max_actions_per_episode
                    if self.env.live.last_obs.is_done:
                        reason = "environment_done"
                        break
                    if cap is not None and self.env.live.episode_action_count >= cap:
                        reason = "max_actions_per_episode"
                        break
                    choices = list(self.problem.enumerate_actions(self.env.live.last_obs))
                    if not choices:
                        reason = "no_available_actions"
                        break
                    obs, _ = self._step(rng.choice(choices))
                    steps += 1
                    metrics = self.problem.compute_episode_metrics(
                        Obs.from_json(obs), steps=self.env.live.action_count
                    )
                    if self.problem.is_perfect(self.problem.aggregate_episode_metrics([metrics])):
                        self.terminal = self.terminal or "solved"
                    self._limit()
                    self.persist()
                    if self.terminal:
                        reason = self.terminal
                        break
            except Exception as exc:
                self.terminal = self.terminal or "initial_collection_failure"
                reason = self.terminal
                self.event(
                    "initial_collection_error",
                    error=budgets.truncate_error(
                        f"{type(exc).__name__}: {exc}", self.options.feedback.max_error_chars
                    ),
                )
            if not self.terminal and self.env.live.last_obs.is_done:
                reason = "environment_done"
            elapsed = time.monotonic() - started
            self.initial_collection = {
                "policy": "uniform_complete_actions",
                "seed": seed,
                "target_unique_observations": self.options.n_unique_observations_in_initial_collection,
                "target_reached": self.store.summary()["n_unique_observations"]
                >= self.options.n_unique_observations_in_initial_collection,
                "real_actions": steps,
                "stop_reason": reason,
                "elapsed_seconds": elapsed,
            }
            if self.env.live is not None and self.env.live.last_obs.is_done:
                self.store.finish_episode(self.episode, "environment_done", self.initial_collection)
            self.event("dataset_prepared", **self.initial_collection)
            self.persist()

    def _note_timing(self, phase: str, seconds: float) -> None:
        """Total and slowest duration per phase, so a call that overruns its deadline shows
        which uninterruptible part (native step, durable store commit) took the time."""
        total, slowest = f"{phase}_seconds", f"{phase}_max_seconds"
        self.step_timings[total] = self.step_timings.get(total, 0.0) + seconds
        self.step_timings[slowest] = max(self.step_timings.get(slowest, 0.0), seconds)

    def _reset(self, purpose: str, **metadata: Any) -> dict[str, Any]:
        if self.episode is not None:
            self.store.finish_episode(self.episode, "reset", {})
        env = (
            self.env.live if self.single_instance and self.env.live is not None else self.env.make()
        )
        initial = env.reset_explicit("environment", seed=self.config.problem.seed)
        obs = initial.to_json()
        self.episode, self.current_id = self.store.start_episode(
            obs, purpose, {"seed": self.config.problem.seed, "task": self.task, **metadata}
        )
        return obs

    def _step(self, action: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._limit():
            raise TaskStopped(self.terminal)
        if self.env.live is None or self.env.live.last_obs is None or self.episode is None:
            raise ValueError("no active episode; reset first")
        if self.env.live.last_obs.is_done:
            raise ValueError("environment is terminal; start a fresh episode")
        cap = self.config.limits.max_actions_per_episode
        if cap is not None and self.env.live.episode_action_count >= cap:
            raise ValueError("max_actions_per_episode")
        self.problem.validate_controller_action(action)
        before = json.loads(canonical(self.env.live.last_obs.to_json()))
        try:
            started = time.monotonic()
            after = self.env.live.step(action).to_json()
            stepped = time.monotonic()
            evidence = self.store.record_step(self.episode, before, action, after)
            self._note_timing("native_step", stepped - started)
            self._note_timing("record_step", time.monotonic() - stepped)
        except InvalidActionError:
            raise
        except Exception:
            self.terminal = "environment_or_storage_failure"
            # A real action may have happened. Never retry or continue uncertain history.
            raise
        self.current_id = evidence["after_obs_id"]
        new_milestones = []
        for milestone in after["info"].get("milestones", []):
            if milestone not in self._seen_milestones:
                self._seen_milestones.add(milestone)
                item = {
                    "name": milestone,
                    "kind": self.problem.milestone_kind(milestone),
                    "observation_id": self.current_id,
                    "transition_id": evidence["transition_id"],
                    "episode_id": evidence["episode_id"],
                    "event_id": evidence["event_id"],
                }
                self.milestones.append(item)
                new_milestones.append(item)
                self.event("milestone_observed", milestone=item)
        evidence["new_milestones"] = new_milestones
        if self.uses_model and evidence["conflicting_witnesses"]:
            self.terminal = "observation_determinism_violation"
            self.event("observation_determinism_violation", **evidence)
        return after, evidence

    def public_environment(self, op: str, body: dict[str, Any]) -> dict[str, Any]:
        # Record denials even if the caller catches/suppresses the HTTP error.
        with self.lock:
            self.store.event(
                "flagged_tool_call",
                self.phase,
                {"reason": "cwm_direct_environment_access", "operation": op},
            )
            if self.context is not None:
                self.context.experiment.flagged_tool_calls += 1
                self.context.logger.log(
                    LogComponent.AGENT,
                    "WARNING",
                    "flagged_tool_call",
                    reason="cwm_direct_environment_access",
                    operation=op,
                    flags=["Direct environment access is forbidden; use RunController"],
                )
        # Deny even during bootstrap and even when callers bypass workspace helpers.
        raise HTTPException(
            409,
            detail={
                "code": "cwm_direct_environment_disabled",
                "message": ("Direct environment access is unavailable; use RunController."),
            },
        )

    def worker(
        self,
        bundle: Path,
        seconds: float | None,
        budget_key: str = "protocol.execution.max_seconds_per_RunController",
        *,
        clock: budgets.AgentClock | None = None,
    ) -> Worker:
        """``seconds`` bounds the worker's wall time, or with a ``clock`` its submitted-code time."""
        # Game modules can live inside the interpreter prefix; carve them out.
        from regact.orchestration.task import _secret_module_paths

        return Worker(
            bundle,
            self.options.execution,
            deadline=min(self.deadline, budgets.deadline(None if clock else seconds)),
            runtime=SandboxRuntime(self.config.sandbox_opts.get("backend", "auto")),
            deny_read=_secret_module_paths(self.problem.secret_modules()),
            task_deadline=lambda: self.deadline,
            budget_key=budget_key,
            budget_seconds=seconds,
            load_model=self.uses_model,
            clock=clock,
        )

    def data(self, body: dict[str, Any]) -> Any:
        with self.lock:
            value: Any
            op = body.get("op")
            ids = body.get("ids", [])
            if isinstance(ids, str):
                ids = expand_id_ranges(ids, max_items=self.options.data_api.max_items)
            if not isinstance(ids, list) or any(type(i) is not int or i < 1 for i in ids):
                raise ValueError("ids must be a list of positive integers")
            limit = body.get("limit", self.options.data_api.max_items)
            after_id = body.get("after_id", 0)
            if (
                (limit is not None and (type(limit) is not int or limit < 1))
                or type(after_id) is not int
                or after_id < 0
            ):
                raise ValueError(
                    "limit must be a positive integer or null; after_id must be a nonnegative integer"
                )
            if self.options.data_api.max_items is not None and (
                len(ids) > self.options.data_api.max_items
                or limit is None
                or limit > self.options.data_api.max_items
            ):
                raise ValueError(
                    f"query supports at most {self.options.data_api.max_items} items; "
                    "use pagination"
                )
            if op == "summary":
                value = {
                    "phase": self.phase,
                    "initial_observation_id": self.initial_id,
                    "current_observation_id": self.current_id,
                    "controller_start_observation_id": self.current_id
                    if self.single_instance
                    else self.initial_id,
                    "episode_id": self.episode,
                    "reset_actions": self.reset_actions,
                    **{k: v for k, v in self.store.summary().items() if k != "dataset_version"},
                    **({"milestones": self.milestones} if self.milestones else {}),
                }
            elif op == "list_observation_ids":
                value = [
                    oid
                    for oid in self.store.observation_ids()
                    if oid > int(body.get("after_id") or 0)
                ][:limit]
            elif op == "list_transition_ids":
                value = self.store.transition_ids(int(body.get("after_id") or 0), limit)
            elif op == "observations":
                value = [self.store.observation(int(i)) for i in ids]
            elif op == "transitions":
                value = [self.store.transition(int(i)) for i in ids]
            elif op == "diagnostic":
                value = self.store.get_diagnostic(int(body["id"]))
            elif op == "image":
                sources = [
                    key
                    for key in ("observation_id", "transition_id", "diagnostic_id")
                    if body.get(key) is not None
                ]
                if len(sources) != 1:
                    raise ValueError(
                        "save_image requires exactly one observation_id, transition_id or diagnostic_id"
                    )
                source = sources[0]
                identifier = body[source]
                if type(identifier) is not int or identifier < 1:
                    raise ValueError(f"{source} must be a positive integer")
                which = body.get("which")
                if source == "diagnostic_id":
                    which = which or "observed"
                    if which not in ("observed", "predicted"):
                        raise ValueError("diagnostic image side must be observed or predicted")
                    diagnostic = self.store.get_diagnostic(identifier)
                    obs = diagnostic.get(which)
                    if not isinstance(obs, dict) or "frame" not in obs:
                        raise ValueError(
                            f"diagnostic {identifier} has no {which} observation to render; use load_diagnostic to inspect its evidence"
                        )
                elif source == "observation_id":
                    if which is not None:
                        raise ValueError(
                            "which applies only to transitions or diagnostics; omit it for an observation"
                        )
                    obs = self.store.observation(identifier)
                else:
                    which = which or "after"
                    if which not in ("before", "after"):
                        raise ValueError("transition image side must be before or after")
                    transition = self.store.transition(identifier)
                    obs = transition["o" if which == "before" else "o_next"]
                from regact.protocols.cwm.viewer import png

                value = {"png_base64": base64.b64encode(png(self.problem, obs)).decode()}
            else:
                raise ValueError("unknown data operation")
            # Keep individual records/images readable; the byte guard bounds bulk queries.
            if (
                op in ("observations", "transitions", "list_observation_ids", "list_transition_ids")
                and len(value) > 1
                and self.options.data_api.max_response_bytes is not None
                and len(canonical(value).encode()) > self.options.data_api.max_response_bytes
            ):
                raise ValueError(
                    f"response exceeds protocol.data_api.max_response_bytes={self.options.data_api.max_response_bytes} bytes; request a smaller batch"
                )
            return value

    @property
    def single_instance(self):
        return self.config.problem.lifecycle is Lifecycle.SINGLE_INSTANCE

    @property
    def commands(self):
        return {
            "RunController": "Run a fresh controller from controller.py.",
            **reset_commands(self.config, self.problem),
        }

    def reset_environment(self, kind):
        if self._limit():
            raise ValueError(self.terminal)
        before = self.current_id
        env = self.env.live
        assert env is not None
        try:
            obs = env.reset_explicit(kind, seed=self.config.problem.seed).to_json()
            self.reset_actions += 1
            if self.episode is not None:
                self.store.finish_episode(self.episode, "reset_" + kind, {})
            self.episode, self.current_id = self.store.start_episode(
                obs, "reset_" + kind, {"previous_observation_id": before}
            )
        except Exception:
            self.terminal = "environment_or_storage_failure"
            raise
        self.after_reset()
        return {
            "stop_reason": "reset_" + kind,
            "episode_id": self.episode,
            "current_observation_id": self.current_id,
            "observation_ids": [self.current_id],
            "real_actions": 1,
        }

    def after_reset(self):
        pass

    def exploration_model(self):
        return None

    def open_model(self, stack, model, clock):
        return None

    def controller_input(self, worker, obs):
        return obs

    def predict(self, worker, state, action):
        return None

    def check_prediction(self, predicted, obs, evidence):
        pass

    def after_valid_run(self):
        pass

    def explore(self) -> dict[str, Any]:
        model = self.exploration_model()
        bundle, manifest = snapshot(
            self.workdir, self.root / "bundles", ["controller.py"], model=model
        )
        goal = description(bundle / "controller.py")
        known = self.store.observation_hashes()
        rid = self.store.record("exploration", "running", {"goal": goal, "bundle": bundle.name})
        self.active_request = rid
        result: dict[str, Any] = {
            "exploration_id": rid,
            "goal": goal,
            "bundle": bundle.name,
            "manifest": manifest,
            "model_bundle": model.name if model else None,
            **({"cwm_version": self.accepted["cwm_version"]} if self.accepted else {}),
            "dataset_version": self.store.version,
            "initial_observation_id": self.initial_id,
            "current_observation_id": self.current_id,
            "episode_id": self.episode,
            "reset_actions": self.reset_actions,
            "stage": "real",
            "new_milestones": [],
            "real_actions": 0,
            "actual_novel_observations": 0,
            "objective_reached": None,
        }
        # Predict immediately before each real action; no preliminary simulation.
        action = None
        if self.single_instance:
            assert self.env.live is not None and self.env.live.last_obs is not None
            obs = self.env.live.last_obs.to_json()
        else:
            obs = self._reset("exploration", exploration_id=rid, bundle=bundle.name)
        result["initial_observation_id"] = self.current_id
        result["observation_sequence"] = [self.current_id]
        result["transition_sequence"] = []
        result["episode_id"] = self.episode
        self.store.update_record(rid, "running", result)
        assert self.episode is not None and self.env.live is not None
        if (
            self.uses_model
            and not self.single_instance
            and canonical(obs) != canonical(self.initial)
        ):
            self.terminal = "initial_observation_mismatch"
            result["stop_reason"] = self.terminal
            result["diagnostic_id"] = self.store.diagnostic(
                {
                    "kind": self.terminal,
                    "predicted": self.initial,
                    "observed": obs,
                    "observation_id": self.current_id,
                    "initial_observation_id": self.initial_id,
                    "current_observation_id": self.current_id,
                    "episode_id": self.episode,
                    "reset_actions": self.reset_actions,
                }
            )
        else:
            reason = "max_actions_per_RunController"
            evidence: dict[str, Any] = {}
            role = "model"
            call_started = time.monotonic()
            self.step_timings = {}
            worker = controller = None
            clock = budgets.AgentClock(self.options.execution.max_seconds_per_RunController)
            try:
                with ExitStack() as stack:
                    worker = self.open_model(stack, model, clock)
                    controller = stack.enter_context(
                        self.worker(
                            bundle,
                            self.options.execution.max_seconds_per_RunController,
                            clock=clock,
                        )
                    )
                    role = "controller"
                    controller.call("controller_init")
                    for _ in budgets.action_indices(self.options.max_actions_per_RunController):
                        if obs["is_done"]:
                            reason = "environment_done"
                            break
                        role = "model" if self.uses_model else "controller"
                        state = self.controller_input(worker, obs)
                        role = "controller"
                        result["objective_reached"] = controller.call("objective", state=state)
                        if self._limit():
                            reason = self.terminal or "interrupted"
                            break
                        if obs["is_done"]:
                            reason = "environment_done"
                            break
                        if controller.call("is_done", state=state):
                            reason = controller.call("completion_reason", state=state)
                            break
                        cap = self.config.limits.max_actions_per_episode
                        if cap is not None and self.env.live.episode_action_count >= cap:
                            reason = "max_actions_per_episode"
                            break
                        action = controller.call("act", state=state)
                        role = "model"
                        predicted = self.predict(worker, state, action)
                        role = "environment"
                        obs, evidence = self._step(action)
                        result["real_actions"] += 1
                        result["observation_sequence"].append(self.current_id)
                        result["transition_sequence"].append(evidence["transition_id"])
                        result["new_milestones"].extend(evidence.get("new_milestones", []))
                        fingerprint = digest(obs)
                        if fingerprint not in known:
                            known.add(fingerprint)
                            result["actual_novel_observations"] += 1
                        if self.terminal == "observation_determinism_violation":
                            reason = self.terminal or "interrupted"
                            result["counterexample"] = evidence
                            result["diagnostic_id"] = self.store.diagnostic(
                                {"kind": reason, "observed": obs, **evidence}
                            )
                            break
                        role = "model"
                        self.check_prediction(predicted, obs, evidence)
                        # Always check the final real observation, even at an action cap.
                        state = self.controller_input(worker, obs)
                        role = "controller"
                        result["objective_reached"] = controller.call("objective", state=state)
                    else:
                        reason = (
                            "environment_done" if obs["is_done"] else "max_actions_per_RunController"
                        )
            except TaskStopped:
                # The limit raced with prediction computation; no real action was attempted.
                reason = self.terminal or "interrupted"
            except ModelMismatch as exc:
                reason = exc.kind
                diagnostic = {
                    "kind": exc.kind,
                    "predicted": exc.predicted,
                    "observed": exc.actual,
                    **({"cwm_version": self.accepted["cwm_version"]} if self.accepted else {}),
                    **exc.evidence,
                }
                result["diagnostic_id"] = self.store.diagnostic(diagnostic)
                result["counterexample"] = {
                    k: v for k, v in diagnostic.items() if k not in ("predicted", "observed")
                }
                if exc.predicted is not None:
                    result.update(
                        differences(exc.predicted, exc.actual, self.options.feedback.max_diff_items)
                    )
                self.change_phase(MODELING, reason, diagnostic_id=result["diagnostic_id"])
            except (WorkerError, InvalidActionError) as exc:
                reason = (
                    "controller_error"
                    if role == "controller" or isinstance(exc, InvalidActionError)
                    else "model_error"
                )
                if getattr(exc, "kind", None) == "operation_timeout":
                    reason = self._limit() or "controller_call_time_limit"
                result["error"] = budgets.truncate_error(
                    str(exc), self.options.feedback.max_error_chars
                )
                result["error_type"] = getattr(exc, "kind", type(exc).__name__)
                result["error_context"] = dict(getattr(exc, "context", {}))
                if reason == "controller_call_time_limit":
                    result["error_context"]["budget"] = {
                        "parameter": "protocol.execution.max_seconds_per_RunController",
                        "value": self.options.execution.max_seconds_per_RunController,
                        "unit": "seconds of submitted-code execution per RunController call",
                    }
                    result["error"] = (
                        "This controller call used up its submitted-code time limit."
                    )

                if (
                    isinstance(exc, InvalidActionError)
                    or result["error_context"].get("callback") == "step"
                ):
                    result["error_context"]["action"] = action
                if self.uses_model and reason == "model_error":
                    self.change_phase(MODELING, reason)
            except Exception as exc:
                # Environment/storage faults are framework failures, never controller blame.
                self.terminal = self.terminal or "framework_failure"
                reason = self.terminal
                result["error"] = budgets.truncate_error(
                    f"{type(exc).__name__}: {exc}", self.options.feedback.max_error_chars
                )
                result["error_type"] = "framework_failure"
                result["history_complete"] = False
            result["elapsed_seconds"] = time.monotonic() - call_started
            result["timings"] = {
                "submitted_code_seconds": clock.used,
                **self.step_timings,
                **{
                    f"{name}_close_seconds": proc.close_seconds
                    for name, proc in (("model_worker", worker), ("controller_worker", controller))
                    if getattr(proc, "close_seconds", None) is not None
                },
            }
            result["stop_reason"] = reason
        metrics = self.problem.compute_episode_metrics(
            Obs.from_json(obs), steps=result["real_actions"]
        )
        if result["stop_reason"] == "controller_error":
            result["observed_metrics_before_error"] = metrics
            metrics = self.problem.failure_metrics(steps=result["real_actions"])
        result["metrics"] = metrics
        result["aggregate"] = self.problem.aggregate_episode_metrics([metrics])
        if self.problem.is_perfect(result["aggregate"]) and self.terminal not in (
            "observation_determinism_violation",
            "environment_or_storage_failure",
            "framework_failure",
            "initial_observation_mismatch",
        ):
            self.terminal = "solved"
        self._limit()
        if self.phase == EXPLORATION and not result.get("error") and not self.terminal:
            self.after_valid_run()
        self.latest = self._feedback(result)

        # Problem-provided aggregate keys are not universally orderable. Keep best by the
        # existing primary success/progress fields; preserve every full metric vector too.
        def score(r: dict[str, Any]) -> float:
            return self.problem.exploration_score(r.get("aggregate", {}))

        if self.best is None or score(result) > score(self.best):
            self.best = self.latest
        result["current_observation_id"] = self.current_id
        if obs["is_done"]:
            self.store.finish_episode(self.episode, "environment_done", result)
        elif not self.single_instance:
            self.store.finish_episode(self.episode, result["stop_reason"], result)
        self.store.update_record(rid, "completed", result)
        self.active_request = None
        return self._feedback(result)

    @staticmethod
    def _feedback(result: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in result.items() if k != "manifest"}

    def exploration_evidence(self, result: dict[str, Any], *, previews: bool) -> None:
        if result.get("episode_id") is None:
            return
        observations = result.get("observation_sequence", [])
        transitions = result.get("transition_sequence", [])
        result["observation_ids"] = sorted(set(observations))
        result["transition_ids"] = sorted(set(transitions))
        if not previews or not self.options.n_tmp_images_saved_per_exploration:
            return
        selected = select_preview_ids(observations, self.options.n_tmp_images_saved_per_exploration)
        saved = []
        try:
            from regact.protocols.cwm.viewer import png

            for oid in selected:
                save_preview(self.workdir, oid, png(self.problem, self.store.observation(oid)))
                saved.append(oid)
        except Exception as exc:
            result["image_preview_error"] = budgets.truncate_error(
                f"Could not save all observation previews: {exc}",
                self.options.feedback.max_error_chars,
            )
            self.event("image_preview_error", error=result["image_preview_error"])
        if saved:
            result["image_observation_ids"] = saved

    def tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.closed:
                return {"error": "Task is closed", "exit_reason": self.terminal}
            if name not in self.commands:
                return {
                    "error": f"{name} is not enabled for this task.",
                    "error_type": "command_unavailable",
                }
            rid = args.get("request_id")
            request = {"tool": name, "args": args}
            if rid:
                try:
                    previous = self.store.request_result(str(rid), request)
                except ValueError as exc:
                    return {"error": str(exc), "error_type": "request_conflict"}
                if previous is not None:
                    return {**previous, "replayed": True}
            previews = False
            preview_error = None
            self.phase_changes = []
            try:
                if set(args) - {"request_id"}:
                    raise ValueError(
                        "these tools use fixed workspace files; only request_id is accepted"
                    )
                if name == "RunController":
                    try:
                        clear_images(self.workdir)
                        previews = True
                    except (OSError, ValueError) as exc:
                        preview_error = budgets.truncate_error(
                            f"Could not prepare tmp/images safely: {exc}",
                            self.options.feedback.max_error_chars,
                        )
                        self.event("image_preview_error", error=preview_error)
                if self._limit():
                    raise ValueError(self.terminal)
                if name in reset_commands(self.config, self.problem):
                    response = self.reset_environment(
                        "level" if name == "ResetLevel" else "environment"
                    )
                elif name == "UpdateCodeWorldModel" and self.phase in (MODELING, EXPLORATION):
                    response = self.update()
                elif name == "PlanInCWM" and self.phase == EXPLORATION:
                    response = self.plan()
                elif name == "RunController" and self.phase == EXPLORATION:
                    response = self.explore()
                else:
                    raise ValueError(
                        f"{name} is unavailable during {self.phase}. Build or repair world_model/ and call UpdateCodeWorldModel first."
                    )
            except (WorkerError, ValueError, ImportError, SyntaxError, FileNotFoundError) as exc:
                response = {
                    "error": budgets.truncate_error(
                        str(exc), self.options.feedback.max_error_chars
                    ),
                    "error_type": getattr(exc, "kind", type(exc).__name__),
                    "error_context": getattr(exc, "context", {}),
                }
                if getattr(exc, "evidence", None):
                    response["diagnostic_id"] = self.store.diagnostic(exc.evidence)
                if name == "UpdateCodeWorldModel" and self.accepted:
                    response["cwm_version"] = self.accepted["cwm_version"]
                if self.active_request is not None:
                    response = {
                        **self._feedback(self.store.record_payload(self.active_request)),
                        **response,
                    }
                    self.store.update_record(self.active_request, "failed", response)
                    self.active_request = None
            except Exception as exc:
                self.terminal = self.terminal or "framework_failure"
                response = {
                    "error": budgets.truncate_error(
                        f"{type(exc).__name__}: {exc}", self.options.feedback.max_error_chars
                    ),
                    "error_type": "framework_failure",
                }
            if self.active_request is not None:
                self.store.update_record(self.active_request, "failed", response)
                self.active_request = None
            if name == "RunController":
                self.exploration_evidence(response, previews=previews)
                if preview_error:
                    response["image_preview_error"] = preview_error
                if response.get("exploration_id") is not None:
                    self.store.update_record(
                        response["exploration_id"],
                        "failed" if response.get("error") else "completed",
                        response,
                    )
                if self.latest and self.latest.get("exploration_id") == response.get(
                    "exploration_id"
                ):
                    self.latest.update(response)
            self._limit()
            response.update(
                phase=self.phase, exit_reason=self.terminal, dataset_version=self.store.version
            )
            if self.phase_changes:
                response["phase_change"] = {
                    **self.phase_changes[-1],
                    "from": self.phase_changes[0]["from"],
                }
            self.event(name, result=response)
            if rid:
                self.store.remember_request(str(rid), request, response)
            self.persist()
            return response

    async def shutdown(self, reason: str | None) -> None:
        # Signal active workers before waiting for the coordinator lock. A disconnected
        # HTTP client must not leave real exploration running until its ordinary deadline.
        self.deadline = min(self.deadline, time.monotonic())
        self.terminal = self.terminal or reason or "interrupted"
        await asyncio.to_thread(self.close, self.terminal)

    def close(self, reason: str | None) -> None:
        with self.lock:
            if self.closed:
                return
            self.terminal = self.terminal or reason or "interrupted"
            try:
                if self.episode is not None:
                    self.store.finish_episode(self.episode, self.terminal, {}, status="interrupted")
                    self.episode = None
                if self.active_request is not None:
                    self.store.update_record(
                        self.active_request, "interrupted", {"stop_reason": self.terminal}
                    )
                    self.active_request = None
                self.event("cwm_run_finished", reason=self.terminal)
                self.persist()
            finally:
                try:
                    self.env.close()
                finally:
                    self.store.close()
                    self.closed = True


class ModelMismatch(Exception):
    def __init__(self, kind: str, predicted: Any, actual: Any, evidence: dict[str, Any]) -> None:
        super().__init__(kind)
        self.kind, self.predicted, self.actual, self.evidence = kind, predicted, actual, evidence


class ManagedTool(Tool):
    def __init__(self, name: str, coordinator: ManagedCoordinator) -> None:
        self._name, self.coordinator = name, coordinator

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self.coordinator.commands[self.name]

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    async def call(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        from regact.protocols.cwm.feedback import format_feedback, render_feedback

        if args:
            return ToolOutput(
                data=format_feedback(
                    {
                        "status": "Refused",
                        "message": (
                            "PlanInCWM takes no arguments; it reads goal.py and the accepted CWM from world_model/*."
                            if self.name == "PlanInCWM"
                            else f"{self.name} takes no arguments."
                        ),
                    }
                ),
                is_error=True,
            )
        internal_args: dict[str, Any] = {}
        if context.detail.get("request_id"):
            internal_args["request_id"] = context.detail["request_id"]
        job = asyncio.create_task(
            asyncio.to_thread(self.coordinator.tool, self.name, internal_args)
        )
        try:
            result = await asyncio.shield(job)
        except asyncio.CancelledError:
            # Never release ownership while its real execution is still running.
            self.coordinator.deadline = min(self.coordinator.deadline, time.monotonic())
            while not job.done():
                try:
                    await asyncio.shield(job)
                except asyncio.CancelledError:
                    continue
            raise
        return ToolOutput(
            data=render_feedback(
                self.name, result, self.coordinator.options, self.coordinator.config.limits
            ),
            is_error=bool(result.get("error")),
        )


@dataclasses.dataclass(kw_only=True)
class ManagedSession(ProtocolSession):
    coordinator: ManagedCoordinator

    def stop_reason(self) -> str | None:
        return self.coordinator.terminal

    def reminder(self, reminders: int) -> str:
        phase = self.coordinator.phase
        return (
            "Continue your work until the game is fully solved.\n"
            f"Current phase is {phase}: {self.coordinator.phase_description(phase)}"
        )

    def interaction_guidance(self) -> str:
        return ""

    async def prepare(self, stop: StopSignal | None = None) -> None:
        context = self.coordinator.context
        if context is not None:
            context.experiment.save(str(self.coordinator.output / "logs/experiment_state.json"))
        await asyncio.to_thread(self.coordinator.collect_initial, stop)

    def on_start(self, start: float) -> None:
        budget = self.coordinator.config.limits.seconds_left()
        self.coordinator.deadline = start + budget if budget is not None else float("inf")

    async def close(self) -> None:
        context = self.coordinator.context
        await self.coordinator.shutdown(context.experiment.exit_reason if context else None)
