"""One trusted CWM coordinator per task: phases, real actions, and immutable versions."""

from __future__ import annotations

import math

import time
from contextlib import ExitStack
from typing import Any

from regact.envclient.obs import Obs
from regact.protocols.cwm import limits as budgets
from regact.protocols.cwm.bundle import description, require_unchanged_model, snapshot, write_plan
from regact.protocols.cwm.commands import enabled_commands
from regact.protocols.cwm.planner import plan
from regact.protocols.cwm.store import canonical
from regact.protocols.cwm.validation import (
    SIZE_KEYS,
    add_state_size,
    advance,
    check_observation,
    compactness_failure,
    explanation_note,
    validate,
)
from regact.protocols.cwm.worker import Worker, WorkerError
from regact.protocols.managed.prompting import reset_commands
from regact.protocols.managed.session import (
    public_evidence,
    EXPLORATION,
    MODEL_FILES,
    MODELING,
    ManagedCoordinator,
    ModelMismatch,
)


class Coordinator(ManagedCoordinator):
    """CWM validation and planning on top of the shared managed environment."""

    protocol_name = "cwm"
    initial_phase = MODELING
    uses_model = True

    online_bits = [0, 0.0]  # real steps explained during acting, and what they cost

    def task_metrics(self):
        metrics = super().task_metrics()
        if metrics is not None and self.options.stochastic and self.online_bits[0]:
            metrics = {**metrics, "online_bits_per_step": round(self.online_bits[1] / self.online_bits[0], 3)}
        return metrics

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # single_instance: the accepted CWM's State at a point of the live episode, carried across
        # RunController calls. None until a CWM is accepted, and after a contradiction.
        self.running: dict[str, Any] | None = None  # {"state", "hash"}
        self.state_sizes: dict[str, Any] = {}  # compactness totals, validation + exploration
        self.current_state: Any = None  # the State the controller sees in this RunController
        self.next_state: Any = None  # step's prediction, adopted once reality matches it

    @property
    def commands(self):
        return {**enabled_commands(self.options), **reset_commands(self.config, self.problem)}

    def exploration_model(self):
        assert self.accepted is not None
        model = self.root / "bundles" / self.accepted["bundle"]
        require_unchanged_model(self.workdir, model, self.model_files())
        return model

    def model_files(self) -> list[str]:
        if not self.options.stochastic:
            return MODEL_FILES
        optional = [f"world_model/model_{part}.py" for part in ("explain", "reset")]
        return [*MODEL_FILES, *(name for name in optional if (self.workdir / name).exists())]

    def open_model(self, stack, model, clock):
        seconds = self.options.execution.max_seconds_per_RunController
        self.explainer = (
            stack.enter_context(self.worker(model, seconds, clock=clock))
            if self.options.stochastic
            else None
        )
        return stack.enter_context(self.worker(model, seconds, clock=clock))

    def after_restore(self):
        # The store's last accepted validation, in case status.json missed it. Steps recorded
        # after it are checked against the CWM when live_state next replays the live episode.
        record = self.store.last_record("validation", "accepted")
        if record is None:
            self.accepted, self.phase = None, MODELING
            return
        summary = record["validation"]
        self.accepted = {
            "cwm_version": record["cwm_version"],
            "bundle": record["bundle"],
            "dataset_version": summary["dataset_version"],
            "validation": summary,
        }
        self.state_sizes = {k: summary[k] for k in SIZE_KEYS if k in summary}

    def controller_input(self, worker, obs):
        if self.current_state is None:
            self.current_state = (
                self.live_state(worker)
                if self.single_instance
                else self._explained(
                    worker, worker.call("get_initial_state", obs=obs), obs, self.current_id
                )
            )
        return self.current_state

    def begin_exploration(self):
        self.current_state = self.next_state = None
        self.call_bits: list[float] = []
        self.surprise: dict[str, Any] | None = None

    def _advance(self, worker, state, action, obs):
        """A stochastic CWM's State after a real step, with the step's explanation."""
        advanced = advance(worker, self.explainer, state, action, obs, self.options)
        return advanced["state"], advanced

    def after_reset(self):
        # the State the life ended in, for the model's reset(); None when it was not explained
        self.previous_final = self.running["state"] if self.running is not None else None

    def predict(self, worker, state, action):
        if self.options.stochastic:
            self._pending = (worker, state, action)  # explained once the real step is known
            return None
        self.next_state = worker.call("step", state=state, action=action)
        return check_observation(worker.call("render", state=self.next_state))

    def controller_started(self, controller):
        self.surprise_cap = self.options.max_surprise_bits_per_choice
        if self.options.stochastic:
            own = controller.call("surprise_bits")
            if own["set"]:
                self.surprise_cap = own["value"]

    def stop_after_step(self):
        return "surprise" if self.surprise is not None else None

    def annotate_run(self, result):
        if self.options.stochastic and self.call_bits:
            self.online_bits = [
                self.online_bits[0] + len(self.call_bits),
                self.online_bits[1] + sum(self.call_bits),
            ]
            result["bits"] = {
                "total": round(sum(self.call_bits), 2),
                "per_step": round(sum(self.call_bits) / len(self.call_bits), 3),
                "largest_step": round(max(self.call_bits), 2),
            }
            if self.surprise is not None:
                result["surprise"] = self.surprise

    def check_prediction(self, predicted, obs, evidence):
        advanced = None
        if self.options.stochastic:
            worker, state, action = self._pending
            self.next_state, advanced = self._advance(worker, state, action, obs)
            predicted = check_observation(worker.call("render", state=self.next_state))
            if canonical(predicted) != canonical(obs):
                evidence = {**evidence, **explanation_note(advanced)}
        if canonical(predicted) != canonical(obs):
            states = {"state_before": self.current_state, "state": self.next_state}
            self.running = self.current_state = None
            raise ModelMismatch(
                "prediction_mismatch", predicted, obs, public_evidence(evidence), states
            )
        self.current_state = self.next_state
        if "explained_until" in self.accepted:
            self.accepted["explained_until"] = evidence["event_id"]
        self._check_size(self.current_state, obs, evidence["after_obs_id"])
        if self.single_instance:
            self.running = {"state": self.current_state, "hash": evidence["history_hash"]}
        if advanced is not None:
            self.call_bits.append(advanced["bits"])
            cap = self.surprise_cap
            worst = min(advanced["choices"], key=lambda item: item[2], default=None)
            if cap is not None and worst is not None and -math.log2(worst[2]) > cap:
                self.surprise = {
                    "bits": round(-math.log2(worst[2]), 2),
                    "threshold": cap,
                    "step_bits": round(advanced["bits"], 2),
                    "least_likely_choice": worst[0],
                    "value": worst[1],
                    "probability": round(worst[2], 6),
                    "observation_id": evidence["after_obs_id"],
                    "transition_id": evidence["transition_id"],
                }

    def after_valid_run(self):
        self.accepted["dataset_version"] = self.store.version

    def live_state(self, worker: Worker) -> Any:
        """The accepted CWM's State at the live episode's current point: the running state, caught
        up through anything recorded since (a reset, steps), each screen checked on the way."""
        position = self.store.last_hash(self.episode)
        if self.running is not None and self.running["hash"] == position:
            return self.running["state"]
        episode = next(e for e in self.store.episodes() if e["episode_id"] == self.episode)
        steps = self.store.episode_steps(self.episode)
        anchor = self.running["hash"] if self.running is not None else None
        hashes = [item["history_hash"] for item in steps]
        done = hashes.index(anchor) + 1 if anchor in hashes else None
        if done is None and anchor != episode["start_hash"]:
            oid = episode["initial_obs_id"]
            start = self.store.observation(oid)
            first = (
                worker.call("start_state", previous=getattr(self, "previous_final", None), obs=start)
                if self.options.stochastic
                else worker.call("get_initial_state", obs=start)
            )
            state = self._explained(worker, first, start, oid)
        else:
            assert self.running is not None
            state = self.running["state"]
        for item in steps[done or 0 :]:
            after = self.store.observation(item["after_obs_id"])
            if self.options.stochastic:
                state, _ = self._advance(worker, state, item["action"], after)
            else:
                state = worker.call("step", state=state, action=item["action"])
            state = self._explained(worker, state, after, item["after_obs_id"])
        self.running = {"state": state, "hash": position}
        return state

    def _explained(self, worker: Worker, state: Any, obs: dict[str, Any], oid: int) -> Any:
        rendered = check_observation(worker.call("render", state=state))
        if canonical(rendered) != canonical(obs):
            self.running = None
            raise ModelMismatch(
                "reconstruction_mismatch",
                rendered,
                obs,
                {"observation_id": oid},
                {"state_before": None, "state": state},
            )
        self._check_size(state, obs, oid)
        return state

    def _check_size(self, state: Any, obs: dict[str, Any], oid: int) -> None:
        allowance = self.options.max_state_bytes_always_allowed
        threshold = self.options.threshold_max_state_obs_size_ratio
        add_state_size(self.state_sizes, oid, state, obs, threshold, allowance)
        failure = compactness_failure(self.state_sizes, threshold, allowance)
        if failure is not None:
            self.running = None
            raise ModelMismatch("compression_ratio", None, obs, {"observation_id": oid, **failure})

    def update(self) -> dict[str, Any]:
        rid = self.store.record("validation", "running", {"dataset_version": self.store.version})
        self.active_request = rid
        bundle, manifest = snapshot(self.workdir, self.root / "bundles", self.model_files())
        self.store.update_record(
            rid,
            "running",
            {"bundle": bundle.name, "manifest": manifest, "dataset_version": self.store.version},
        )
        seconds = self.validation_budget()
        clock = budgets.AgentClock(seconds)
        started = time.monotonic()
        with ExitStack() as stack:
            worker = stack.enter_context(
                self.worker(bundle, seconds, "per UpdateCodeWorldModel", clock=clock)
            )
            explainer = (
                stack.enter_context(
                    self.worker(bundle, seconds, "per UpdateCodeWorldModel", clock=clock)
                )
                if self.options.stochastic
                else None
            )
            summary, live = validate(
                worker,
                self.store,
                self.options,
                live_episode=self.episode if self.single_instance else None,
                progress=self.step_progress,
                explained_until=(self.accepted or {}).get("explained_until"),
                explainer=explainer,
            )
        self.save_diagnostic_images(summary["counterexamples"])
        record = {
            "bundle": bundle.name,
            "manifest": manifest,
            "validation": summary,
            "seconds": time.monotonic() - started,
            "model_seconds": clock.used,
        }
        version = (self.accepted["cwm_version"] if self.accepted else 0) + 1  # 1, 2, 3, ...
        if summary["accepted"]:
            record["cwm_version"] = version
        self.store.update_record(rid, "accepted" if summary["accepted"] else "rejected", record)
        self.active_request = None
        if summary["accepted"]:
            self.accepted = {
                "cwm_version": version,
                "bundle": bundle.name,
                "dataset_version": self.store.version,
                "validation": summary,
                # The last step event this version predicted: every one recorded so far.
                "explained_until": self.store.last_step_event_id(),
            }
            self.running = live
            self.state_sizes = {k: summary[k] for k in SIZE_KEYS if k in summary}
            self.change_phase(EXPLORATION, "model_accepted")
            self.event("cwm_accepted", cwm_version=version)
        return {**summary, "cwm_version": self.accepted["cwm_version"] if self.accepted else None}

    def planning_start(self, worker: Worker, start: dict[str, Any]) -> Any:
        """Where a plan starts: the live state in single_instance, the initial observation's State
        in multi_instance. A screen the accepted CWM cannot explain sends the agent back to
        modelling, as in exploration."""
        try:
            if self.single_instance:
                return self.live_state(worker)
            return self._explained(
                worker, worker.call("get_initial_state", obs=start), start, self.initial_id
            )
        except ModelMismatch as exc:
            error = WorkerError(
                f"The accepted CWM does not explain the planning start ({exc.kind}). "
                "Inspect the diagnostic, repair the CWM and validate it again."
            )
            error.kind = exc.kind
            error.evidence = {
                "kind": exc.kind,
                "predicted": exc.predicted,
                "observed": exc.actual,
                **exc.evidence,
            }
            self.change_phase(MODELING, exc.kind)
            raise error from None

    def validation_budget(self) -> float | None:
        """Seconds of submitted code for one validation: the floor, or a per-1,000-recorded-steps
        rate when the dataset is large enough for that to be more."""
        execution = self.options.execution
        floor, rate = (
            execution.max_seconds_per_UpdateCodeWorldModel,
            execution.max_seconds_per_UpdateCodeWorldModel_per_1000_steps,
        )
        if floor is None or rate is None:
            return None
        return max(floor, rate * self.store.summary()["n_total_transitions"] / 1000)

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
                "initial_observation_id": self.current_id
                if self.single_instance
                else self.initial_id,
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
                    "per PlanInCWM call",
                )
            )
            goal_worker = stack.enter_context(
                self.worker(
                    bundle,
                    self.options.planner.max_seconds_per_planner_call,
                    "per PlanInCWM call",
                )
            )
            worker.deadline = goal_worker.deadline = deadline
            start = (
                self.store.observation(self.current_id) if self.single_instance else self.initial
            )
            start_state = self.planning_start(worker, start)
            result = plan(
                worker,
                start,
                self.store.observation_hashes(),
                self.options,
                lambda obs: self.problem.enumerate_actions(Obs.from_json(obs)),
                goal_worker=goal_worker,
                initial_state=start_state,
            )
        metadata = {
            "plan_id": rid,
            "goal": goal,
            "bundle": bundle.name,
            "manifest": manifest,
            "cwm_version": self.accepted["cwm_version"],
            "model_bundle": model.name,
            "initial_observation_id": self.current_id if self.single_instance else self.initial_id,
            "initial_state": start_state,  # the viewer replays the plan from exactly this State
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


# Stable public imports for callers of the earlier CWM module.
from regact.protocols.managed.session import ManagedSession as CwmSession
from regact.protocols.managed.session import ManagedTool as CwmTool

__all__ = ["Coordinator", "CwmTool", "CwmSession", "MODELING", "EXPLORATION"]
