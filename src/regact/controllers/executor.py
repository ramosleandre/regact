"""How a submitted controller is evaluated.

One in-process executor. It loads ``solution.py``, instantiates the controller,
and drives ``controller.act(obs) -> EnvClient.step(action)`` against the env
(already behind HTTP, so no subprocess is needed for integrity). The stop
condition lives here: MULTI_INSTANCE aggregates over N fresh-env episodes;
SINGLE_INSTANCE runs the one shared handle to a level boundary. Controller
exceptions are caught per episode and tagged ``agent_solution``; the aggregate is
written to ``output_path`` and returned.
"""

from __future__ import annotations

import importlib.util
import json
import os
from typing import Any

from regact.config.schema import Lifecycle
from regact.controllers.runner import run_controller
from regact.envclient.client import EnvClient
from regact.obs.errors import ErrorCategory
from regact.obs.result import EpisodeResult, EvalResult
from regact.security.policy import default_policy
from regact.security.scan import scan_file


class ControllerExecutor:
    """Evaluate the controller in ``solution.py`` and return a result.

    Controller-specific (it loads ``solution.py`` and runs a controller), hence
    the ``Controller`` prefix and the ``controllers/`` home — distinct from the
    agnostic ``EnvClient`` it drives.
    """

    def __init__(self, env: EnvClient) -> None:
        self._env = env

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
        # Anti-cheat: statically scan the controller before its module body runs,
        # so a solution that imports the game lib / escape hatch never executes.
        violations = scan_file(solution_path, default_policy())
        if violations:
            result = EvalResult(
                task=task_name,
                error="anti-cheat: " + "; ".join(violations),
                error_category=ErrorCategory.AGENT_SOLUTION,
                executor="in_process",
            )
            _write(output_path, result)
            return result

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
        result = EvalResult(
            task=task_name,
            aggregate=_aggregate(episodes),
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
            metrics={
                "steps": summary.total_steps,
                "success": summary.stop_kind == "env_done",
                "reward": self._env.last_reward,
            },
        )


def _load_controller_factory(solution_path: str) -> Any:
    """Import ``solution.py`` in isolation and return its ``get_controller`` callable."""
    spec = importlib.util.spec_from_file_location("regact_agent_solution", solution_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load solution from {solution_path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_controller


def _aggregate(episodes: list[EpisodeResult]) -> dict[str, Any]:
    """Union-of-metrics aggregate: counts + success rate + mean steps."""
    ok = [e for e in episodes if e.error is None]
    successes = sum(1 for e in ok if e.metrics.get("success"))
    return {
        "n_episodes": len(episodes),
        "n_errors": len(episodes) - len(ok),
        "success_rate": (successes / len(ok)) if ok else 0.0,
        "mean_steps": (sum(e.metrics.get("steps", 0) for e in ok) / len(ok)) if ok else 0.0,
    }


def _write(output_path: str, result: EvalResult) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result.to_json(), handle, indent=2)
