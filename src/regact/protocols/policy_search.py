"""The existing controller-writing workflow, preserved as the policy_search protocol.

The controller executor, tool implementations, prompts and templates are reused
unchanged. Submission-based completion/reminders live here, not in the shared
agent loop. Legacy optional features remain supported during the CWM migration.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from regact.agent.capabilities import ToolProtocol
from regact.config.schema import Lifecycle, RunConfig
from regact.features.base import Feature, FeatureContext, RunDeps, build_features
from regact.features.controller import Controller
from regact.obs.errors import LogComponent
from regact.obs.logger import RunLogger
from regact.problems.base import BaseProblem
from regact.prompt.builder import PromptBuilder
from regact.protocols.base import ExperimentProtocol, ProtocolContext, ProtocolSession
from regact.session.state import ExperimentState
from regact.workspace.templates import TemplateFile

_KEEP_ALIVE_MESSAGE = (
    "Keep-alive reminder - continue working or finish your work: 1) produce a controller in "
    "solution.py, 2) submit it by running `python framework/control.py SubmitSolution`, 3) if you "
    "are satisfied with your solution, end the task with `python framework/control.py ExitTask`."
)
# Used when the agent cannot end its own run (exit_task_enabled=False): no ExitTask mention.
_KEEP_ALIVE_MESSAGE_NO_EXIT = (
    "Keep-alive reminder - keep improving your controller: 1) refine the controller in "
    "solution.py, 2) submit it via `python framework/control.py SubmitSolution`, 3) keep "
    "iterating until fully solving the task."
)

# Repeating one sentence does not convert a non-submitter: over 43 runs, submitters and
# non-submitters got the same number of reminders (29.1 vs 27.9) and two ignored it 57 times.
_ESCALATE_AFTER_REMINDERS = 10
_KEEP_ALIVE_MESSAGE_ESCALATED = (
    "You have now received {n} reminders and have not submitted anything. Submit NOW, even if "
    "your controller is unfinished: run `python framework/control.py SubmitSolution`. Every "
    "submission is scored and the result is returned to you, so submitting is how you find out "
    "whether your controller works - keep improving it afterwards."
)


def _keep_alive_message(base: str, reminders: int, submissions: int) -> str:
    """The reminder to send. Escalates once a run has been reminded repeatedly and still submitted
    nothing - firing early only costs a stronger sentence, never the run."""
    if submissions == 0 and reminders >= _ESCALATE_AFTER_REMINDERS:
        return _KEEP_ALIVE_MESSAGE_ESCALATED.format(n=reminders)
    return base


def _solved(
    experiment: ExperimentState, is_perfect: Callable[[dict[str, Any]], bool] | None
) -> bool:
    """Whether the latest submission scored perfect, so the run has nothing left to do."""
    last = experiment.last_submission_results
    if not is_perfect or not last or last.get("error"):
        return False
    aggregate = last.get("aggregate", {})
    if not aggregate.get("evaluation_complete") or aggregate.get("n_errors", 0):
        return False
    if any(e.get("error") for e in last.get("episodes", [])):
        return False
    return bool(is_perfect(aggregate))


def _acted_without_submitting(
    reason: str | None, submission_count: int, tool_calls_total: int
) -> bool:
    """Shape-3 tell: the agent ran tools but never submitted and exited on walltime - an unbound
    control channel (native protocol on a subprocess agent -> submit/exit 503s) or a doom loop,
    scoring only via the teardown re-score. turn/tool_calls ratio look healthy, so submission_count
    with env_moves is the only signal - the failure the spinning detector cannot see."""
    return reason == "walltime_limit" and submission_count == 0 and tool_calls_total > 0


def _collect_feature_metrics(
    features: list[Feature], deps: RunDeps, logger: RunLogger
) -> dict[str, Any]:
    """Every loaded feature's own submission numbers, keyed by feature name.

    Empty contributions are dropped so a submission only carries features that
    actually scored something. A faulty contributor is logged and skipped — extra
    metrics must never break the submission that carries them.
    """
    collected: dict[str, Any] = {}
    for feature in features:
        try:
            metrics = feature.submission_metrics(deps)
        except Exception as exc:
            logger.log(
                LogComponent.EVAL,
                "WARNING",
                "feature_metrics_failed",
                feature=feature.name,
                error=f"{type(exc).__name__}: {exc}",
            )
            continue
        if metrics:
            collected[feature.name] = metrics
    return collected


@dataclass(kw_only=True)
class PolicySearchSession(ProtocolSession):
    experiment: ExperimentState
    exit_task_enabled: bool = True
    is_perfect: Callable[[dict[str, Any]], bool] | None = None

    def stop_reason(self) -> str | None:
        # Preserve precedence: a perfect submission wins over ExitTask and budgets.
        if _solved(self.experiment, self.is_perfect):
            return "solved"
        if self.experiment.exit_requested:
            return "agent_exit"
        return None

    def reminder(self, reminders: int) -> str:
        base = _KEEP_ALIVE_MESSAGE if self.exit_task_enabled else _KEEP_ALIVE_MESSAGE_NO_EXIT
        return _keep_alive_message(base, reminders, self.experiment.submission_count)

    def after_teardown(self, reason: str, logger: RunLogger) -> None:
        experiment = self.experiment
        if _acted_without_submitting(
            reason, experiment.submission_count, experiment.tool_calls_total
        ):
            logger.log(
                LogComponent.ORCHESTRATOR,
                "WARNING",
                "acted_without_submitting",
                tool_calls_total=experiment.tool_calls_total,
                env_moves=experiment.env_moves,
            )


class PolicySearchProtocol(ExperimentProtocol):
    name = "policy_search"

    def __init__(self, config: RunConfig) -> None:
        if config.protocol.options:
            raise ValueError(
                "policy_search has no protocol-specific options; keep evaluation settings "
                "under controller.*"
            )
        self.config = config
        self.controller = Controller.from_config(config.controller)
        self.features = build_features(config.features)

    def validate(self) -> None:
        if "cwm" in self.config.features:
            raise ValueError(
                "CWM v3 is retired: replace features=cwm with protocol=cwm features=none; "
                "use protocol.* settings"
            )
        if self.config.problem.lifecycle is Lifecycle.SINGLE_INSTANCE and (
            self.controller.evaluates_on_env
            or any(feature.evaluates_on_env for feature in self.features)
        ):
            raise RuntimeError(
                "single-instance problem with an on-env evaluation (the always-on controller "
                "scores by rolling episodes; an evaluating feature may too): exploration and "
                "evaluation share the same env, so scores would reflect the session, not an "
                "isolated policy"
            )

    def templates(self, ctx: FeatureContext) -> Iterator[TemplateFile]:
        # Preserve the old bootstrap order: write controller files before asking
        # a feature for its templates (a feature can inspect those earlier files).
        yield from self.controller.templates(ctx)
        for feature in self.features:
            yield from feature.templates(ctx)

    def env_wrappers(self, ctx: FeatureContext) -> list[Callable[[Any], Any]]:
        return [wrap for feature in self.features if (wrap := feature.env_wrapper(ctx)) is not None]

    def build_system_prompt(
        self,
        problem: BaseProblem,
        task_name: str,
        *,
        tool_protocol: ToolProtocol,
        tool_names: list[str],
        verbalize_variant: str,
    ) -> str:
        return PromptBuilder().build_system_prompt(
            problem,
            task_name,
            self.features,
            controller=self.controller,
            lifecycle=self.config.problem.lifecycle,
            info_mode=self.config.problem.info_mode,
            obs_mode=self.config.problem.obs_mode,
            tool_protocol=tool_protocol,
            tool_names=tool_names,
            verbalize_variant=verbalize_variant,
        )

    def bind(self, ctx: ProtocolContext) -> PolicySearchSession:
        deps = RunDeps(
            experiment=ctx.experiment,
            env_client=ctx.env_client,
            lifecycle=ctx.lifecycle,
            solution_path=os.path.join(ctx.workdir, "solution.py"),
            submissions_dir=os.path.join(ctx.workdir, "submissions"),
            failure_metrics=ctx.failure_metrics,
            compute_episode_metrics=ctx.compute_episode_metrics,
            aggregate_episode_metrics=ctx.aggregate_episode_metrics,
            sandbox_wrap=ctx.sandbox_wrap,
            render_frame=ctx.render_frame,
            seed=ctx.seed,
        )
        deps.feature_metrics = lambda: _collect_feature_metrics(self.features, deps, ctx.logger)
        tools = [*self.controller.tools(deps), *(t for f in self.features for t in f.tools(deps))]
        hooks = [*self.controller.hooks(deps), *(h for f in self.features for h in f.hooks(deps))]
        return PolicySearchSession(
            experiment=ctx.experiment,
            tools=tools,
            hooks=hooks,
            exit_task_enabled=self.config.controller.exit_task_enabled,
            is_perfect=ctx.is_perfect,
        )
