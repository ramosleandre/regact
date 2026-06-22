"""How a submitted controller is evaluated.

One in-process executor. It loads ``solution.py``, instantiates the controller,
and drives ``controller.act(obs) -> EnvClient.step(action)`` against the env
(already behind HTTP). The stop condition lives here: MULTI_INSTANCE aggregates
over N fresh-env episodes; SINGLE_INSTANCE runs the one shared handle to a level
boundary. Per-episode and run metrics are computed by injected callables (the
problem's, so the problem owns what a score means), defaulting to a generic
success/steps/reward summary. Controller exceptions are caught per episode and
tagged ``agent_solution``; the aggregate is written to ``output_path`` and returned.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
from collections.abc import Callable
from typing import Any

from regact.config.schema import Lifecycle
from regact.controllers.runner import run_controller
from regact.envclient.client import EnvClient
from regact.envclient.obs import Obs
from regact.obs.errors import ErrorCategory
from regact.obs.result import EpisodeResult, EvalResult

# Metric callables: the problem supplies these (it knows what its score means);
# a generic default is used when none is injected.
EpisodeMetrics = Callable[..., dict[str, Any]]
AggregateMetrics = Callable[[list[dict[str, Any]]], dict[str, Any]]


def _default_episode_metrics(final_obs: Obs, *, steps: int) -> dict[str, Any]:
    """Generic per-episode metrics: success requires a positive terminal reward."""
    reward = final_obs.reward or 0.0
    return {"success": bool(final_obs.is_done and reward > 0), "steps": steps, "reward": reward}


def _default_aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Generic aggregate over per-episode metric dicts (errors are counted by the caller)."""
    n = len(episodes)
    if n == 0:
        return {"n_episodes": 0, "success_rate": 0.0, "mean_steps": 0.0}
    return {
        "n_episodes": n,
        "success_rate": sum(bool(e.get("success")) for e in episodes) / n,
        "mean_steps": sum(e.get("steps", 0) for e in episodes) / n,
    }


class ControllerExecutor:
    """Evaluate the controller in ``solution.py`` and return a result.

    Controller-specific (it loads ``solution.py`` and runs a controller), hence
    the ``Controller`` prefix and the ``controllers/`` home — distinct from the
    agnostic ``EnvClient`` it drives. Metric computation is delegated to the
    injected callables so the problem layer owns what a score means.
    """

    def __init__(
        self,
        env: EnvClient,
        *,
        compute_metrics: EpisodeMetrics | None = None,
        aggregate_metrics: AggregateMetrics | None = None,
    ) -> None:
        self._env = env
        self._compute_metrics = compute_metrics or _default_episode_metrics
        self._aggregate_metrics = aggregate_metrics or _default_aggregate

    def run(
        self,
        *,
        task_name: str,
        solution_path: str,
        output_path: str,
        lifecycle: Lifecycle,
        n_episodes: int = 1,
        max_moves: int = 400,
    ) -> EvalResult:
        """Drive the controller via the env client and persist the result."""
        try:
            factory = _load_controller_factory(solution_path)
        except Exception as exc:  # import / attribute / syntax error in agent code
            result = EvalResult(
                task=task_name,
                error=f"{type(exc).__name__}: {exc}",
                error_category=ErrorCategory.AGENT_SOLUTION,
                executor="in_process",
            )
            _write(output_path, result)
            return result

        episode_count = 1 if lifecycle is Lifecycle.SINGLE_INSTANCE else n_episodes
        episodes = [
            self._run_one(index, factory, max_moves=max_moves) for index in range(episode_count)
        ]
        aggregate = self._aggregate_metrics([e.metrics for e in episodes if e.error is None])
        aggregate["n_errors"] = sum(1 for e in episodes if e.error is not None)
        result = EvalResult(
            task=task_name,
            aggregate=aggregate,
            episodes=episodes,
            executor="in_process",
        )
        _write(output_path, result)
        return result

    def _run_one(self, index: int, factory: Any, *, max_moves: int) -> EpisodeResult:
        """Run a single episode on a fresh reset, catching controller faults."""
        self._env.reset()
        try:
            controller = factory()
            summary = run_controller(self._env, controller, max_steps=max_moves)
        except Exception as exc:
            return EpisodeResult(
                episode=index,
                error=f"{type(exc).__name__}: {exc}",
                error_category=ErrorCategory.AGENT_SOLUTION,
            )
        return EpisodeResult(
            episode=index,
            stop_kind=summary.stop_kind,
            stop_reason=summary.stop_reason,
            milestones=[{"step": m.step, "description": m.description} for m in summary.milestones],
            metrics=self._compute_metrics(summary.final_obs, steps=summary.total_steps),
        )


def _load_controller_factory(solution_path: str) -> Any:
    """Import ``solution.py`` in isolation and return its ``get_controller`` callable.

    The solution's directory (the agent workdir) is put on ``sys.path`` while the
    module runs so its sibling packages — ``code_library`` (the controller base)
    and ``framework`` — resolve; it is removed again once the imports have run.
    """
    workdir = os.path.dirname(os.path.abspath(solution_path))
    spec = importlib.util.spec_from_file_location("regact_agent_solution", solution_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load solution from {solution_path!r}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, workdir)
    try:
        spec.loader.exec_module(module)
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(workdir)
    return module.get_controller


def _write(output_path: str, result: EvalResult) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result.to_json(), handle, indent=2)
