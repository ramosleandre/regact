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

from regact.config.schema import RunConfig
from regact.env.session import EnvSession
from regact.envclient.errors import InvalidActionError
from regact.envclient.obs import Obs
from regact.obs.errors import LogComponent
from regact.orchestration.signals import StopSignal
from regact.problems.base import BaseProblem
from regact.protocols.base import ProtocolContext, ProtocolSession
from regact.protocols.cwm import limits as budgets
from regact.protocols.cwm.bundle import description, snapshot, write_plan
from regact.protocols.cwm.commands import COMMANDS
from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.ids import expand_id_ranges
from regact.protocols.cwm.images import clear_images, save_preview, select_preview_ids
from regact.protocols.cwm.planner import plan
from regact.protocols.cwm.store import ExperienceStore, atomic_json, canonical, digest
from regact.protocols.cwm.validation import add_state_size, check_observation, differences, validate
from regact.protocols.cwm.worker import Worker, WorkerError
from regact.security.runtime import SandboxRuntime
from regact.tools.base import Tool, ToolContext, ToolOutput

MODELING = "CWM Modeling"
EXPLORATION = "Active Exploration"
PHASE_DESCRIPTIONS = {
    MODELING: (
        "Inspect the current dataset, build or repair world_model/, "
        "then validate it with UpdateCodeWorldModel."
    ),
    EXPLORATION: (
        "Produce an exploration controller in exploration.py aiming to explore new and "
        "relevant states, then submit it with SubmitExplorationController. "
        "Optionally define a goal in goal.py and use PlanInCWM first."
    ),
}

MODEL_FILES = [
    f"world_model/model_{part}.py" for part in ("state", "parser", "render", "transition")
]


class TaskStopped(Exception):
    """A known task limit reached before an environment action; history stays complete."""


class Coordinator:
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
        self.phase = MODELING
        self._messages: list[str] = []
        self._seen_milestones: set[str] = set()
        self.milestones: list[dict[str, Any]] = []
        self.terminal: str | None = None
        self.deadline = float("inf")
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

    def change_phase(self, phase: str, reason: str, **evidence: Any) -> None:
        """Record transitions centrally; the dispatch layer delivers queued notices in order."""
        if phase == self.phase:
            return
        before, self.phase = self.phase, phase
        self.event("phase_changed", before=before, after=phase, reason=reason, **evidence)
        self._messages.append(
            f"Phase transition: {before} --> {phase}.\n{PHASE_DESCRIPTIONS[phase]}"
        )

    def drain_messages(self) -> list[str]:
        with self.lock:
            messages, self._messages = self._messages, []
            return messages

    def execute_request(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        # Pair notices with their completed operation under the same lock. Concurrent
        # HTTP callers cannot consume each other's transitions; replay creates none.
        with self.lock:
            result = self.tool(name, args)
            return result, self.drain_messages()

    def persist(self) -> None:
        atomic_json(
            self.root / "status.json",
            {
                "protocol": "cwm",
                "phase": self.phase,
                "exit_reason": self.terminal,
                "initial_observation_id": self.initial_id,
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
        if budget is not None and self.store.summary()["n_total_transitions"] >= budget:
            self.terminal = self.terminal or "real_action_limit"
        return self.terminal

    def collect_initial(self, stop: StopSignal | None = None) -> None:
        """Trusted, bounded random prefill. No agent runs during this operation."""
        with self.lock:
            if self.initial_collection is not None:
                return
            started = time.monotonic()
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
                    needs_reset = self.env.live.last_obs.is_done or (
                        cap is not None and self.env.live.episode_action_count >= cap
                    )
                    if needs_reset and canonical(self._reset("initial_collection")) != canonical(
                        self.initial
                    ):
                        self.terminal = reason = "initial_observation_mismatch"
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
            self.initial_collection = {
                "policy": "uniform_complete_actions",
                "seed": seed,
                "target_unique_observations": self.options.n_unique_observations_in_initial_collection,
                "target_reached": self.store.summary()["n_unique_observations"]
                >= self.options.n_unique_observations_in_initial_collection,
                "real_actions": steps,
                "stop_reason": reason,
                "elapsed_seconds": time.monotonic() - started,
            }
            if self.episode is not None:
                self.store.finish_episode(
                    self.episode,
                    "initial_collection_" + reason,
                    self.initial_collection,
                    status="interrupted"
                    if self.terminal and self.terminal != "solved"
                    else "completed",
                )
                self.episode = None
            self.event("dataset_prepared", **self.initial_collection)
            self.persist()

    def _reset(self, purpose: str, **metadata: Any) -> dict[str, Any]:
        if self.episode is not None:
            self.store.finish_episode(self.episode, "reset", {})
        env = self.env.make()
        initial = env.reset(seed=self.config.problem.seed)
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
        before = json.loads(canonical(self.env.live.last_obs.to_json()))
        try:
            after = self.env.live.step(action).to_json()
            evidence = self.store.record_step(self.episode, before, action, after)
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
        if evidence["conflicting_witnesses"]:
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
                    flags=["Direct environment access is forbidden in the CWM protocol"],
                )
        # Deny even during bootstrap and even when callers bypass workspace helpers.
        raise HTTPException(
            409,
            detail={
                "code": "cwm_direct_environment_disabled",
                "message": ("Direct environment access is unavailable in CWM."),
            },
        )

    def worker(
        self,
        bundle: Path,
        seconds: float | None,
        budget_key: str = "protocol.execution.max_seconds_per_episode",
    ) -> Worker:
        # Game modules can live inside the interpreter prefix; carve them out.
        from regact.orchestration.task import _secret_module_paths

        return Worker(
            bundle,
            self.options.execution,
            deadline=min(self.deadline, budgets.deadline(seconds)),
            runtime=SandboxRuntime(self.config.sandbox_opts.get("backend", "auto")),
            deny_read=_secret_module_paths(self.problem.secret_modules()),
            task_deadline=lambda: self.deadline,
            budget_key=budget_key,
            budget_seconds=seconds,
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

    def update(self) -> dict[str, Any]:
        rid = self.store.record("validation", "running", {"dataset_version": self.store.version})
        self.active_request = rid
        bundle, manifest = snapshot(self.workdir, self.root / "bundles", MODEL_FILES)
        self.store.update_record(
            rid,
            "running",
            {"bundle": bundle.name, "manifest": manifest, "dataset_version": self.store.version},
        )
        with self.worker(
            bundle,
            self.options.execution.max_seconds_per_UpdateCodeWorldModel,
            "protocol.execution.max_seconds_per_UpdateCodeWorldModel",
        ) as worker:
            summary, states = validate(worker, self.store, self.options)
        record = {"bundle": bundle.name, "manifest": manifest, "validation": summary}
        self.store.update_record(rid, "accepted" if summary["accepted"] else "rejected", record)
        self.active_request = None
        if summary["accepted"]:
            self.accepted = {
                "cwm_version": rid,
                "bundle": bundle.name,
                "dataset_version": self.store.version,
                "validation": summary,
            }
            self.states = states
            self.change_phase(EXPLORATION, "model_accepted")
            self.event("cwm_accepted", cwm_version=rid)
        return {**summary, "cwm_version": self.accepted["cwm_version"] if self.accepted else None}

    def plan(self) -> dict[str, Any]:
        assert self.accepted is not None
        model = self.root / "bundles" / self.accepted["bundle"]
        bundle, manifest = snapshot(self.workdir, self.root / "bundles", ["goal.py"], model=model)
        goal = description(bundle / "goal.py")
        rid = self.store.record("plan", "running", {"goal": goal, "bundle": bundle.name})
        self.active_request = rid
        self.store.update_record(
            rid,
            "running",
            {
                "goal": goal,
                "bundle": bundle.name,
                "manifest": manifest,
                "cwm_version": self.accepted["cwm_version"],
                "model_bundle": model.name,
                "initial_observation_id": self.initial_id,
                "dataset_version": self.store.version,
            },
        )
        deadline = min(
            self.deadline, budgets.deadline(self.options.planner.max_seconds_per_planner_call)
        )
        with ExitStack() as stack:
            worker = stack.enter_context(
                self.worker(
                    model,
                    self.options.planner.max_seconds_per_planner_call,
                    "protocol.planner.max_seconds_per_planner_call",
                )
            )
            goal_worker = stack.enter_context(
                self.worker(
                    bundle,
                    self.options.planner.max_seconds_per_planner_call,
                    "protocol.planner.max_seconds_per_planner_call",
                )
            )
            worker.deadline = goal_worker.deadline = deadline
            result = plan(
                worker,
                self.initial,
                self.store.observation_hashes(),
                self.options,
                lambda obs: self.problem.enumerate_actions(Obs.from_json(obs)),
                goal_worker=goal_worker,
            )
        metadata = {
            "plan_id": rid,
            "goal": goal,
            "bundle": bundle.name,
            "manifest": manifest,
            "cwm_version": self.accepted["cwm_version"],
            "model_bundle": model.name,
            "initial_observation_id": self.initial_id,
            "dataset_version": self.store.version,
            **result,
        }
        if result["candidate_found"]:
            path = self.workdir / "plans" / f"plan_{rid:03d}.py"
            write_plan(
                self.workdir,
                str(path.relative_to(self.workdir)),
                (
                    '"""Planner candidate; submit an exploration to execute it in real'
                    'ity."""\nACTIONS = '
                )
                + repr(result["actions"])
                + "\nPLAN_ID = "
                + str(rid)
                + "\n",
            )
            metadata["path"] = str(path.relative_to(self.workdir))
        self.store.update_record(rid, "completed", metadata)
        self.active_request = None
        return {
            k: v
            for k, v in metadata.items()
            if k not in ("actions", "predicted_observation_hashes", "manifest")
        }

    def _model_observation(self, worker: Worker, obs: dict[str, Any], oid: int) -> Any:
        state = worker.call("parse", obs=obs)
        restored = check_observation(worker.call("render", state=state))
        if canonical(restored) != canonical(obs):
            raise ModelMismatch("reconstruction_mismatch", restored, obs, {"observation_id": oid})
        text = canonical(state)
        for other, represented in self.states.items():
            if other != oid and canonical(represented) == text:
                raise ModelMismatch(
                    "parser_collision", None, obs, {"observation_ids": [other, oid]}
                )
        if oid not in self.states:
            self.states[oid] = state
        sizes: dict[str, Any] = {}
        for i, represented in self.states.items():
            add_state_size(sizes, i, represented, self.store.observation(i))
        numerator = sizes["state_bytes"]
        denominator = sizes["observation_bytes"]
        if numerator / max(1, denominator) >= self.options.threshold_max_state_obs_size_ratio:
            raise ModelMismatch(
                "compression_ratio",
                None,
                obs,
                {
                    "ratio": numerator / denominator,
                    "required_below": self.options.threshold_max_state_obs_size_ratio,
                    **sizes,
                },
            )
        return state

    def explore(self) -> dict[str, Any]:
        assert self.accepted is not None
        model = self.root / "bundles" / self.accepted["bundle"]
        bundle, manifest = snapshot(
            self.workdir, self.root / "bundles", ["exploration.py"], model=model
        )
        goal = description(bundle / "exploration.py")
        known = self.store.observation_hashes()
        rid = self.store.record("exploration", "running", {"goal": goal, "bundle": bundle.name})
        self.active_request = rid
        result: dict[str, Any] = {
            "exploration_id": rid,
            "goal": goal,
            "bundle": bundle.name,
            "manifest": manifest,
            "model_bundle": model.name,
            "cwm_version": self.accepted["cwm_version"],
            "dataset_version": self.store.version,
            "initial_observation_id": self.initial_id,
            "stage": "simulation",
            "new_milestones": [],
            "simulation_actions": 0,
            "real_actions": 0,
            "predicted_novel_observations": 0,
            "actual_novel_observations": 0,
            "objective_reached": None,
            "simulation_action_sequence": [],
            "simulation_prediction_hashes": [],
        }
        # Simulation and real each get a fresh worker/controller and separate action/time allowances.
        action = None
        self.store.update_record(rid, "running", result)
        try:
            deadline = min(
                self.deadline, budgets.deadline(self.options.execution.max_seconds_per_episode)
            )
            with ExitStack() as stack:
                worker = stack.enter_context(
                    self.worker(model, self.options.execution.max_seconds_per_episode)
                )
                controller = stack.enter_context(
                    self.worker(bundle, self.options.execution.max_seconds_per_episode)
                )
                worker.deadline = controller.deadline = deadline
                controller.call("controller_init")
                state = worker.call("parse", obs=self.initial)
                obs = self.initial
                novel: set[str] = set()
                for _ in budgets.action_indices(self.options.max_actions_per_exploration):
                    if obs["is_done"] or controller.call("is_done", state=state):
                        break
                    action = controller.call("act", state=state)
                    state = worker.call("step", state=state, action=action)
                    obs = check_observation(worker.call("render", state=state))
                    result["simulation_actions"] += 1
                    result["simulation_action_sequence"].append(action)
                    result["simulation_prediction_hashes"].append(digest(obs))
                    if digest(obs) not in known:
                        novel.add(digest(obs))
                    result["predicted_novel_observations"] = len(novel)
        except WorkerError as exc:
            if exc.context.get("callback") == "step":
                exc.context["action"] = action
            raise
        finally:
            self.store.update_record(rid, "running", result)
        if not novel:
            result["stop_reason"] = "no_predicted_novelty"
            self.store.update_record(rid, "rejected", result)
            self.active_request = None
            return self._feedback(result)
        result["stage"] = "real"
        obs = self._reset(
            "exploration",
            cwm_version=self.accepted["cwm_version"],
            exploration_id=rid,
            bundle=bundle.name,
        )
        result["episode_id"] = self.episode
        self.store.update_record(rid, "running", result)
        assert self.episode is not None and self.env.live is not None
        if canonical(obs) != canonical(self.initial):
            self.terminal = "initial_observation_mismatch"
            result["stop_reason"] = self.terminal
            result["diagnostic_id"] = self.store.diagnostic(
                {
                    "kind": self.terminal,
                    "predicted": self.initial,
                    "observed": obs,
                    "observation_id": self.current_id,
                    "initial_observation_id": self.initial_id,
                }
            )
        else:
            reason = "max_actions_per_exploration"
            evidence: dict[str, Any] = {}
            role = "model"
            try:
                deadline = min(
                    self.deadline, budgets.deadline(self.options.execution.max_seconds_per_episode)
                )
                with ExitStack() as stack:
                    worker = stack.enter_context(
                        self.worker(model, self.options.execution.max_seconds_per_episode)
                    )
                    controller = stack.enter_context(
                        self.worker(bundle, self.options.execution.max_seconds_per_episode)
                    )
                    worker.deadline = controller.deadline = deadline
                    role = "controller"
                    controller.call("controller_init")
                    for _ in budgets.action_indices(self.options.max_actions_per_exploration):
                        role = "model"
                        state = self._model_observation(worker, obs, self.current_id)
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
                        predicted_state = worker.call("step", state=state, action=action)
                        predicted = check_observation(worker.call("render", state=predicted_state))
                        role = "environment"
                        obs, evidence = self._step(action)
                        result["real_actions"] += 1
                        result["new_milestones"].extend(evidence.get("new_milestones", []))
                        if digest(obs) not in known:
                            known.add(digest(obs))
                            result["actual_novel_observations"] += 1
                        if self.terminal == "observation_determinism_violation":
                            reason = self.terminal or "interrupted"
                            result["counterexample"] = evidence
                            result["diagnostic_id"] = self.store.diagnostic(
                                {"kind": reason, "observed": obs, **evidence}
                            )
                            break
                        role = "model"
                        if canonical(predicted) != canonical(obs):
                            raise ModelMismatch("prediction_mismatch", predicted, obs, evidence)
                        # Always check the final real observation, even at an action cap.
                        state = self._model_observation(worker, obs, self.current_id)
                        role = "controller"
                        result["objective_reached"] = controller.call("objective", state=state)
                    else:
                        reason = (
                            "environment_done" if obs["is_done"] else "max_actions_per_exploration"
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
                    "cwm_version": self.accepted["cwm_version"],
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
                    reason = self._limit() or "episode_time_limit"
                result["error"] = budgets.truncate_error(
                    str(exc), self.options.feedback.max_error_chars
                )
                result["error_type"] = getattr(exc, "kind", type(exc).__name__)
                result["error_context"] = dict(getattr(exc, "context", {}))
                if (
                    isinstance(exc, InvalidActionError)
                    or result["error_context"].get("callback") == "step"
                ):
                    result["error_context"]["action"] = action
                if reason in ("model_error", "episode_time_limit"):
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
            self.accepted["dataset_version"] = self.store.version
        self.latest = self._feedback(result)

        # Problem-provided aggregate keys are not universally orderable. Keep best by the
        # existing primary success/progress fields; preserve every full metric vector too.
        def score(r: dict[str, Any]) -> float:
            return self.problem.exploration_score(r.get("aggregate", {}))

        if self.best is None or score(result) > score(self.best):
            self.best = self.latest
        self.store.finish_episode(self.episode, result["stop_reason"], result)
        self.episode = None
        self.store.update_record(rid, "completed", result)
        self.active_request = None
        return self._feedback(result)

    @staticmethod
    def _feedback(result: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v
            for k, v in result.items()
            if k not in ("manifest", "simulation_action_sequence", "simulation_prediction_hashes")
        }

    def exploration_evidence(self, result: dict[str, Any], *, previews: bool) -> None:
        if result.get("episode_id") is None:
            return
        observations, transitions = self.store.episode_ids(result["episode_id"])
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
                return {"error": "CWM task is closed", "exit_reason": self.terminal}
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
            try:
                if set(args) - {"request_id"}:
                    raise ValueError(
                        "these tools use fixed workspace files; only request_id is accepted"
                    )
                if name == "SubmitExplorationController":
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
                if name == "UpdateCodeWorldModel" and self.phase in (MODELING, EXPLORATION):
                    response = self.update()
                elif name == "PlanInCWM" and self.phase == EXPLORATION:
                    response = self.plan()
                elif name == "SubmitExplorationController" and self.phase == EXPLORATION:
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
            if name == "SubmitExplorationController":
                self.exploration_evidence(response, previews=previews)
                if preview_error:
                    response["image_preview_error"] = preview_error
                if response.get("exploration_id") is not None:
                    self.store.update_record(
                        response["exploration_id"],
                        "failed"
                        if response.get("error")
                        else "rejected"
                        if response.get("stage") == "simulation"
                        else "completed",
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


class CwmTool(Tool):
    def __init__(self, name: str, coordinator: Coordinator) -> None:
        self._name, self.coordinator = name, coordinator

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return COMMANDS[self.name]

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
                            else f"{self.name} takes no arguments; it reads its fixed workspace files."
                        ),
                    }
                ),
                is_error=True,
            )
        internal_args: dict[str, Any] = {}
        if context.detail.get("request_id"):
            internal_args["request_id"] = context.detail["request_id"]
        job = asyncio.create_task(
            asyncio.to_thread(self.coordinator.execute_request, self.name, internal_args)
        )
        try:
            result, notices = await asyncio.shield(job)
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
            messages=notices,
        )


@dataclasses.dataclass(kw_only=True)
class CwmSession(ProtocolSession):
    coordinator: Coordinator

    def stop_reason(self) -> str | None:
        return self.coordinator.terminal

    def reminder(self, reminders: int) -> str:
        phase = self.coordinator.phase
        return (
            "Continue your work until the game is fully solved.\n"
            f"Current phase is {phase}: {PHASE_DESCRIPTIONS[phase]}"
        )

    def interaction_guidance(self) -> str:
        return ""

    async def prepare(self, stop: StopSignal | None = None) -> None:
        context = self.coordinator.context
        if context is not None:
            context.experiment.save(str(self.coordinator.output / "logs/experiment_state.json"))
        await asyncio.to_thread(self.coordinator.collect_initial, stop)

    def on_start(self, start: float) -> None:
        budget = self.coordinator.config.limits.max_seconds_per_task
        self.coordinator.deadline = start + budget if budget is not None else float("inf")

    async def close(self) -> None:
        context = self.coordinator.context
        await self.coordinator.shutdown(context.experiment.exit_reason if context else None)
