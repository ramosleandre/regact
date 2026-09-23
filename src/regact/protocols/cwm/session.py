"""One trusted CWM coordinator per task: phases, real actions, and immutable revisions."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
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
from regact.problems.base import BaseProblem
from regact.protocols.base import ProtocolContext, ProtocolSession
from regact.protocols.cwm.bundle import description, snapshot, write_plan
from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.planner import plan
from regact.protocols.cwm.store import ExperienceStore, atomic_json, canonical, digest
from regact.protocols.cwm.validation import check_observation, differences, validate
from regact.protocols.cwm.worker import Worker, WorkerError
from regact.security.runtime import SandboxRuntime
from regact.tools.base import Tool, ToolContext, ToolOutput

MODEL_FILES = [
    f"world_model/model_{part}.py" for part in ("state", "parser", "render", "transition")
]


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
        self.phase = 0
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
        try:
            self._reset("initial_collection")
        except BaseException:
            self.env.close()
            self.store.close()
            raise
        assert self.env.live is not None and self.env.live.last_obs is not None
        self.initial = self.env.live.last_obs.to_json()
        self.initial_id = self.current_id
        self._phase_boundary()
        self.persist()

    def event(self, kind: str, **payload: Any) -> None:
        self.store.event(kind, self.phase, payload)
        if self.context is not None:
            self.context.logger.log(
                LogComponent.ORCHESTRATOR, "INFO", kind, phase=str(self.phase), **payload
            )

    def persist(self) -> None:
        atomic_json(
            self.root / "status.json",
            {
                "protocol": "cwm",
                "phase": self.phase,
                "exit_reason": self.terminal,
                "initial_observation_id": self.initial_id,
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
        budget = self.config.limits.max_real_actions_per_task
        if budget is not None and self.store.summary()["n_step_events"] >= budget:
            self.terminal = self.terminal or "real_action_limit"
        return self.terminal

    def _phase_boundary(self) -> None:
        if (
            self.phase == 0
            and self.store.summary()["n_observations"] >= self.options.initial_unique_observations
        ):
            self.phase = 1
            if self.episode is not None:
                self.store.finish_episode(self.episode, "initial_collection_complete", {})
                self.episode = None
            self.event("phase_changed", before=0, after=1, reason="initial_unique_observations")

    def _reset(self, purpose: str, **metadata: Any) -> dict[str, Any]:
        if self.episode is not None:
            self.store.finish_episode(self.episode, "reset", {})
        env = self.env.make()
        obs = env.reset(seed=self.config.problem.seed).to_json()
        self.episode, self.current_id = self.store.start_episode(
            obs, purpose, {"seed": self.config.problem.seed, "task": self.task, **metadata}
        )
        return obs

    def _step(self, action: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._limit():
            raise ValueError(self.terminal)
        if self.env.live is None or self.env.live.last_obs is None or self.episode is None:
            raise ValueError("no active episode; reset first")
        if self.env.live.last_obs.is_done:
            raise ValueError("environment is terminal; start a fresh episode")
        cap = self.config.limits.max_actions_per_env
        if cap is not None and self.env.live.action_count >= cap:
            raise ValueError("max_actions_per_env")
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
        if evidence["conflicting_witnesses"]:
            self.terminal = "observation_determinism_violation"
            self.event("observation_determinism_violation", **evidence)
        return after, evidence

    def public_environment(self, op: str, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.closed:
                raise HTTPException(409, detail="CWM task is closed")
            rid = body.get("request_id")
            payload = {"op": op, "body": body}
            if rid:
                try:
                    cached = self.store.request_result(str(rid), payload)
                except ValueError as exc:
                    raise HTTPException(409, detail=str(exc)) from exc
                if cached is not None:
                    return cached
            if self.phase != 0 or self._limit():
                raise HTTPException(
                    409,
                    detail={
                        "code": "cwm_phase_restriction",
                        "message": (
                            f"{op} is unavailable in CWM phase {self.phase}. "
                            "No real action was executed. Read recorded data; use "
                            "UpdateCodeWorldModel in phase 1 or planning/exploration "
                            "tools in phase 2."
                        ),
                    },
                )
            try:
                assert self.env.live is not None and self.env.live.last_obs is not None
                if op == "reset":
                    if body.get("seed") not in (None, self.config.problem.seed):
                        raise ValueError("CWM resets use the fixed problem.seed for this task")
                    obs = self._reset("initial_collection")
                elif op == "step":
                    obs, _evidence = self._step(body.get("action"))
                    metrics = self.problem.compute_episode_metrics(
                        Obs.from_json(obs), steps=self.env.live.action_count
                    )
                    aggregate = self.problem.aggregate_episode_metrics([metrics])
                    if self.problem.is_perfect(aggregate) and not self.terminal:
                        self.terminal = "solved"
                elif op in ("current", "last-step"):
                    obs = self.env.live.last_obs.to_json()
                elif op == "stop":
                    if self.episode is not None:
                        self.store.finish_episode(self.episode, "collector_stopped", {})
                        self.episode = None
                    # Keep observation available; a following reset creates the next episode.
                    response = {"ok": True}
                    if rid:
                        self.store.remember_request(str(rid), payload, response)
                    self.persist()
                    return response
                else:
                    raise ValueError("unsupported environment operation")
                self._phase_boundary()
                self._limit()
                reply = {
                    "obs": obs,
                    "action_count": self.env.live.action_count,
                    "cwm_phase": self.phase,
                    "notice": "Initial collection complete; phase 1: repair the CWM."
                    if self.phase == 1
                    else None,
                }
                if rid:
                    self.store.remember_request(str(rid), payload, reply)
                self.persist()
                return reply
            except HTTPException:
                raise
            except Exception as exc:
                if self.terminal:
                    self.persist()
                code = (
                    "invalid_action"
                    if isinstance(exc, InvalidActionError)
                    else "cwm_environment_error"
                )
                raise HTTPException(422, detail={"code": code, "message": str(exc)}) from exc

    def worker(self, bundle: Path, seconds: float) -> Worker:
        # Game modules can live inside the interpreter prefix; carve them out.
        from regact.orchestration.task import _secret_module_paths

        return Worker(
            bundle,
            self.options.execution,
            deadline=min(self.deadline, time.monotonic() + seconds),
            runtime=SandboxRuntime(self.config.sandbox_opts.get("backend", "auto")),
            deny_read=_secret_module_paths(self.problem.secret_modules()),
            task_deadline=lambda: self.deadline,
        )

    def data(self, body: dict[str, Any]) -> Any:
        with self.lock:
            value: Any
            op = body.get("op")
            ids = body.get("ids", [])
            if not isinstance(ids, list) or any(type(i) is not int or i < 1 for i in ids):
                raise ValueError("ids must be a list of positive integers")
            limit = int(body.get("limit", self.options.data_api.max_items))
            if (
                len(ids) > self.options.data_api.max_items
                or not 1 <= limit <= self.options.data_api.max_items
            ):
                raise ValueError(
                    f"query supports at most {self.options.data_api.max_items} items; "
                    "use pagination"
                )
            if op == "summary":
                value = {
                    "phase": self.phase,
                    "initial_observation_id": self.initial_id,
                    **self.store.summary(),
                }
            elif op == "list_observations":
                value = [
                    oid
                    for oid in self.store.observation_ids()
                    if oid > int(body.get("after_id") or 0)
                ][:limit]
            elif op == "list_transitions":
                value = self.store.transition_ids(int(body.get("after_id") or 0), limit)
            elif op == "observations":
                value = [self.store.observation(int(i)) for i in ids]
            elif op == "transitions":
                value = [self.store.transition(int(i)) for i in ids]
            elif op == "diagnostic":
                value = self.store.get_diagnostic(int(body["id"]))
            elif op == "image":
                if body.get("diagnostic_id") is not None:
                    diagnostic = self.store.get_diagnostic(int(body["diagnostic_id"]))
                    which = body.get("which", "observed")
                    if which not in ("observed", "predicted"):
                        raise ValueError("image side must be observed or predicted")
                    obs = diagnostic[which]
                elif body.get("observation_id") is not None:
                    obs = self.store.observation(int(body["observation_id"]))
                else:
                    transition = self.store.transition(int(body["transition_id"]))
                    which = body.get("which", "after")
                    if which not in ("before", "after"):
                        raise ValueError("image side must be before or after")
                    obs = transition["o" if which == "before" else "o_next"]
                from regact.protocols.cwm.viewer import png

                value = {"png_base64": base64.b64encode(png(self.problem, obs)).decode()}
            else:
                raise ValueError("unknown data operation")
            if len(canonical(value).encode()) > self.options.data_api.max_response_bytes:
                raise ValueError(
                    "response exceeds protocol.data_api.max_response_bytes; request a smaller batch"
                )
            return value

    def update(self) -> dict[str, Any]:
        rid = self.store.record("validation", "running", {"dataset_revision": self.store.revision})
        self.active_request = rid
        bundle, manifest = snapshot(self.workdir, self.root / "bundles", MODEL_FILES)
        self.store.update_record(
            rid,
            "running",
            {"bundle": bundle.name, "manifest": manifest, "dataset_revision": self.store.revision},
        )
        with self.worker(bundle, self.options.execution.max_seconds_per_validation) as worker:
            summary, states = validate(worker, self.store, self.options)
        record = {"bundle": bundle.name, "manifest": manifest, "validation": summary}
        self.store.update_record(rid, "accepted" if summary["accepted"] else "rejected", record)
        self.active_request = None
        if summary["accepted"]:
            before = self.phase
            self.accepted = {
                "cwm_revision": rid,
                "bundle": bundle.name,
                "dataset_revision": self.store.revision,
                "validation": summary,
            }
            self.states = states
            self.phase = 2
            self.event("cwm_accepted", before=before, after=2, cwm_revision=rid)
        return {**summary, "cwm_revision": self.accepted["cwm_revision"] if self.accepted else None}

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
                "cwm_revision": self.accepted["cwm_revision"],
                "model_bundle": model.name,
                "initial_observation_id": self.initial_id,
                "dataset_revision": self.store.revision,
            },
        )
        deadline = min(
            self.deadline, time.monotonic() + self.options.planner.max_seconds_per_planner_call
        )
        with ExitStack() as stack:
            worker = stack.enter_context(
                self.worker(model, self.options.planner.max_seconds_per_planner_call)
            )
            goal_worker = stack.enter_context(
                self.worker(bundle, self.options.planner.max_seconds_per_planner_call)
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
            "cwm_revision": self.accepted["cwm_revision"],
            "model_bundle": model.name,
            "initial_observation_id": self.initial_id,
            "dataset_revision": self.store.revision,
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
        numerator = sum(len(canonical(s).encode()) for s in self.states.values())
        denominator = sum(len(canonical(self.store.observation(i)).encode()) for i in self.states)
        if numerator / max(1, denominator) >= self.options.threshold_state_obs_size_ratio:
            raise ModelMismatch("compression_ratio", None, obs, {"ratio": numerator / denominator})
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
            "cwm_revision": self.accepted["cwm_revision"],
            "dataset_revision": self.store.revision,
            "initial_observation_id": self.initial_id,
            "dream_actions": 0,
            "real_actions": 0,
            "predicted_novel_observations": 0,
            "actual_novel_observations": 0,
            "objective_reached": None,
            "dream_action_sequence": [],
            "dream_prediction_hashes": [],
        }
        # Dream and real each get a fresh worker/controller and separate action/time allowances.
        self.store.update_record(rid, "running", result)
        try:
            deadline = min(
                self.deadline, time.monotonic() + self.options.execution.max_seconds_per_rollout
            )
            with ExitStack() as stack:
                worker = stack.enter_context(
                    self.worker(model, self.options.execution.max_seconds_per_rollout)
                )
                controller = stack.enter_context(
                    self.worker(bundle, self.options.execution.max_seconds_per_rollout)
                )
                worker.deadline = controller.deadline = deadline
                controller.call("controller_init")
                state = worker.call("parse", obs=self.initial)
                obs = self.initial
                novel: set[str] = set()
                for _ in range(self.options.max_actions_per_exploration):
                    if obs["is_done"] or controller.call("is_done", state=state):
                        break
                    action = controller.call("act", state=state)
                    state = worker.call("step", state=state, action=action)
                    obs = check_observation(worker.call("render", state=state))
                    result["dream_actions"] += 1
                    result["dream_action_sequence"].append(action)
                    result["dream_prediction_hashes"].append(digest(obs))
                    if digest(obs) not in known:
                        novel.add(digest(obs))
                result["predicted_novel_observations"] = len(novel)
        finally:
            self.store.update_record(rid, "running", result)
        if not novel:
            result["stop_reason"] = "no_predicted_novelty"
            self.store.update_record(rid, "rejected", result)
            self.active_request = None
            return self._feedback(result)
        obs = self._reset(
            "exploration",
            cwm_revision=self.accepted["cwm_revision"],
            exploration_id=rid,
            bundle=bundle.name,
        )
        result["episode_id"] = self.episode
        self.store.update_record(rid, "running", result)
        assert self.episode is not None and self.env.live is not None
        if canonical(obs) != canonical(self.initial):
            self.terminal = "initial_observation_mismatch"
            result["stop_reason"] = self.terminal
        else:
            reason = "max_actions_per_exploration"
            evidence: dict[str, Any] = {}
            role = "model"
            try:
                deadline = min(
                    self.deadline, time.monotonic() + self.options.execution.max_seconds_per_rollout
                )
                with ExitStack() as stack:
                    worker = stack.enter_context(
                        self.worker(model, self.options.execution.max_seconds_per_rollout)
                    )
                    controller = stack.enter_context(
                        self.worker(bundle, self.options.execution.max_seconds_per_rollout)
                    )
                    worker.deadline = controller.deadline = deadline
                    role = "controller"
                    controller.call("controller_init")
                    for _ in range(self.options.max_actions_per_exploration):
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
                        cap = self.config.limits.max_actions_per_env
                        if cap is not None and self.env.live.action_count >= cap:
                            reason = "max_actions_per_env"
                            break
                        action = controller.call("act", state=state)
                        role = "model"
                        predicted_state = worker.call("step", state=state, action=action)
                        predicted = check_observation(worker.call("render", state=predicted_state))
                        role = "environment"
                        obs, evidence = self._step(action)
                        result["real_actions"] += 1
                        if digest(obs) not in known:
                            known.add(digest(obs))
                            result["actual_novel_observations"] += 1
                        if self.terminal == "observation_determinism_violation":
                            reason = self.terminal or "interrupted"
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
            except ModelMismatch as exc:
                reason = exc.kind
                diagnostic = {
                    "kind": exc.kind,
                    "predicted": exc.predicted,
                    "observed": exc.actual,
                    "cwm_revision": self.accepted["cwm_revision"],
                    **exc.evidence,
                }
                result["diagnostic_id"] = self.store.diagnostic(diagnostic)
                result["counterexample"] = {
                    k: v for k, v in diagnostic.items() if k not in ("predicted", "observed")
                }
                result["differences"] = (
                    differences(exc.predicted, exc.actual, self.options.feedback.max_diff_items)
                    if exc.predicted is not None
                    else []
                )
                self.phase = 1
                self.event(
                    "phase_changed",
                    before=2,
                    after=1,
                    reason=reason,
                    diagnostic_id=result["diagnostic_id"],
                )
            except (WorkerError, InvalidActionError) as exc:
                reason = (
                    "controller_error"
                    if role == "controller" or isinstance(exc, InvalidActionError)
                    else "model_error"
                )
                if getattr(exc, "kind", None) == "operation_timeout":
                    reason = self._limit() or "rollout_time_limit"
                result["error"] = str(exc)[: self.options.feedback.max_error_chars]
                result["error_type"] = getattr(exc, "kind", type(exc).__name__)
                if reason in ("model_error", "rollout_time_limit"):
                    self.phase = 1
                    self.event("phase_changed", before=2, after=1, reason=reason)
            except Exception as exc:
                # Environment/storage faults are framework failures, never controller blame.
                self.terminal = self.terminal or "framework_failure"
                reason = self.terminal
                result["error"] = f"{type(exc).__name__}: {exc}"[
                    : self.options.feedback.max_error_chars
                ]
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
        if self.phase == 2 and not result.get("error") and not self.terminal:
            self.accepted["dataset_revision"] = self.store.revision
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
            if k not in ("manifest", "dream_action_sequence", "dream_prediction_hashes")
        }

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
                    return previous
            try:
                if set(args) - {"request_id"}:
                    raise ValueError(
                        "these tools use fixed workspace files; only request_id is accepted"
                    )
                if self._limit():
                    raise ValueError(self.terminal)
                if name == "UpdateCodeWorldModel" and self.phase in (1, 2):
                    response = self.update()
                elif name == "PlanInCWM" and self.phase == 2:
                    response = self.plan()
                elif name == "SubmitExplorationController" and self.phase == 2:
                    response = self.explore()
                else:
                    raise ValueError(f"{name} is unavailable in phase {self.phase}")
            except (WorkerError, ValueError, ImportError, SyntaxError, FileNotFoundError) as exc:
                response = {
                    "error": str(exc)[: self.options.feedback.max_error_chars],
                    "error_type": getattr(exc, "kind", type(exc).__name__),
                }
                if self.active_request is not None:
                    self.store.update_record(
                        self.active_request,
                        "failed",
                        {**self.store.record_payload(self.active_request), **response},
                    )
                    self.active_request = None
            except Exception as exc:
                self.terminal = self.terminal or "framework_failure"
                response = {
                    "error": f"{type(exc).__name__}: {exc}"[
                        : self.options.feedback.max_error_chars
                    ],
                    "error_type": "framework_failure",
                }
            if self.active_request is not None:
                self.store.update_record(self.active_request, "failed", response)
                self.active_request = None
            response.update(
                phase=self.phase, exit_reason=self.terminal, dataset_revision=self.store.revision
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
        return {
            "UpdateCodeWorldModel": "Validate world_model/ against all real experience.",
            "PlanInCWM": "Plan in the accepted CWM using goal.py; no real actions.",
            "SubmitExplorationController": (
                "Dream-check exploration.py, then execute it in reality if accepted."
            ),
        }[self.name]

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"request_id": {"type": "string"}},
            "additionalProperties": False,
        }

    async def call(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        job = asyncio.create_task(asyncio.to_thread(self.coordinator.tool, self.name, args))
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
            data=json.dumps(result, allow_nan=False), is_error=bool(result.get("error"))
        )


@dataclasses.dataclass(kw_only=True)
class CwmSession(ProtocolSession):
    coordinator: Coordinator

    def stop_reason(self) -> str | None:
        return self.coordinator.terminal

    def reminder(self, reminders: int) -> str:
        c = self.coordinator
        return {
            0: (
                f"Collect initial experience: target {c.options.initial_unique_observations} "
                "unique observations."
            ),
            1: (
                "Phase 1: repair world_model/ using recorded experience, then call"
                " UpdateCodeWorldModel."
            ),
            2: (
                "Phase 2: choose a useful experiment. Write exploration.py and cal"
                "l SubmitExplorationController; PlanInCWM with goal.py is optional"
                "."
            ),
        }[c.phase]

    def on_start(self, start: float) -> None:
        budget = self.coordinator.config.limits.max_seconds_per_task
        self.coordinator.deadline = start + budget if budget is not None else float("inf")

    async def close(self) -> None:
        context = self.coordinator.context
        await self.coordinator.shutdown(context.experiment.exit_reason if context else None)
