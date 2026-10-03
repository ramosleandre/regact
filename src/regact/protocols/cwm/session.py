"""One trusted CWM coordinator per task: phases, real actions, and immutable versions."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any

from regact.envclient.obs import Obs
from regact.protocols.cwm import limits as budgets
from regact.protocols.cwm.bundle import description, require_unchanged_model, snapshot, write_plan
from regact.protocols.cwm.commands import enabled_commands
from regact.protocols.cwm.planner import plan
from regact.protocols.cwm.store import canonical
from regact.protocols.cwm.validation import add_state_size, check_observation, validate
from regact.protocols.cwm.worker import Worker
from regact.protocols.managed.prompting import reset_commands
from regact.protocols.managed.session import (
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Incremental views of self.states, so each real step costs O(1), not O(states).
        self.state_owner: dict[str, int] = {}  # canonical state -> its observation id
        self.state_sizes: dict[str, Any] = {}  # add_state_size totals over self.states

    def _index_states(self) -> None:
        self.state_owner, self.state_sizes = {}, {}
        for oid, state in self.states.items():
            self.state_owner.setdefault(canonical(state), oid)
            add_state_size(self.state_sizes, oid, state, self.store.observation(oid))

    @property
    def commands(self):
        return {**enabled_commands(self.options), **reset_commands(self.config, self.problem)}

    def exploration_model(self):
        assert self.accepted is not None
        model = self.root / "bundles" / self.accepted["bundle"]
        require_unchanged_model(self.workdir, model, MODEL_FILES)
        return model

    def open_model(self, stack, model, clock):
        return stack.enter_context(
            self.worker(model, self.options.execution.max_seconds_per_RunController, clock=clock)
        )

    def controller_input(self, worker, obs):
        return self._model_observation(worker, obs, self.current_id)

    def predict(self, worker, state, action):
        predicted = worker.call("step", state=state, action=action)
        return check_observation(worker.call("render", state=predicted))

    def check_prediction(self, predicted, obs, evidence):
        if canonical(predicted) != canonical(obs):
            raise ModelMismatch("prediction_mismatch", predicted, obs, evidence)

    def after_valid_run(self):
        self.accepted["dataset_version"] = self.store.version

    def after_reset(self):
        # The new reset observation must also be validated. Resets themselves
        # are recorded episode boundaries, never CWM transition examples.
        if self.accepted and self.current_id not in self.states:
            self.change_phase(MODELING, "environment_reset")

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
            self._index_states()
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
                self.store.observation(self.current_id) if self.single_instance else self.initial,
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
            "initial_observation_id": self.current_id if self.single_instance else self.initial_id,
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
        other = self.state_owner.get(text, oid)
        if other != oid:
            raise ModelMismatch("parser_collision", None, obs, {"observation_ids": [other, oid]})
        if oid not in self.states:
            self.states[oid] = state
            self.state_owner[text] = oid
            add_state_size(self.state_sizes, oid, state, obs)
        sizes = dict(self.state_sizes)
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


# Stable public imports for callers of the earlier CWM module.
from regact.protocols.managed.session import ManagedSession as CwmSession
from regact.protocols.managed.session import ManagedTool as CwmTool

__all__ = ["Coordinator", "CwmTool", "CwmSession", "MODELING", "EXPLORATION"]
