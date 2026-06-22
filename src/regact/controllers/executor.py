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
import subprocess
import sys
from collections.abc import Callable
from typing import Any, Protocol

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


class Executor(Protocol):
    """The eval contract the submit/finalize paths depend on (in-process or sandboxed)."""

    def run(
        self,
        *,
        task_name: str,
        solution_path: str,
        output_path: str,
        lifecycle: Lifecycle,
        n_episodes: int = ...,
        max_moves: int = ...,
    ) -> EvalResult: ...


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


def run_episodes_raw(
    env: EnvClient,
    solution_path: str,
    *,
    lifecycle: Lifecycle,
    n_episodes: int,
    max_moves: int,
) -> list[dict[str, Any]]:
    """Run the controller for each episode; return raw per-episode outcomes (no scoring).

    The **untrusted** half: it loads ``solution.py`` and records only what the env returned
    (final obs, steps, milestones, or a per-episode fault) — never a score. Safe to run in a
    sandboxed subprocess. Raises if the controller cannot be loaded (the caller maps that to
    an eval error). Used in-process by :class:`ControllerExecutor` and out-of-process by
    ``regact.controllers.eval_runner``.
    """
    factory = _load_controller_factory(solution_path)
    episode_count = 1 if lifecycle is Lifecycle.SINGLE_INSTANCE else n_episodes
    out: list[dict[str, Any]] = []
    for index in range(episode_count):
        env.reset()
        try:
            summary = run_controller(env, factory(), max_steps=max_moves)
        except Exception as exc:  # a fault inside the agent's controller
            out.append({"episode": index, "error": f"{type(exc).__name__}: {exc}"})
            continue
        out.append(
            {
                "episode": index,
                "stop_kind": summary.stop_kind,
                "stop_reason": summary.stop_reason,
                "milestones": [
                    {"step": m.step, "description": m.description} for m in summary.milestones
                ],
                "final_obs": summary.final_obs.to_json(),
                "steps": summary.total_steps,
            }
        )
    return out


def score_episodes(
    raw_episodes: list[dict[str, Any]],
    *,
    task_name: str,
    compute_metrics: EpisodeMetrics,
    aggregate_metrics: AggregateMetrics,
    executor: str,
) -> EvalResult:
    """Score raw episode outcomes (the **trusted** half): apply the problem's metric callables.

    Kept on the orchestrator side so a sandboxed controller cannot fake its own score.
    """
    episodes: list[EpisodeResult] = []
    for raw in raw_episodes:
        if raw.get("error"):
            episodes.append(
                EpisodeResult(
                    episode=int(raw["episode"]),
                    error=str(raw["error"]),
                    error_category=ErrorCategory.AGENT_SOLUTION,
                )
            )
            continue
        final_obs = Obs.from_json(raw["final_obs"])
        episodes.append(
            EpisodeResult(
                episode=int(raw["episode"]),
                stop_kind=raw.get("stop_kind"),
                stop_reason=raw.get("stop_reason"),
                milestones=raw.get("milestones", []),
                metrics=compute_metrics(final_obs, steps=int(raw["steps"])),
            )
        )
    aggregate = aggregate_metrics([e.metrics for e in episodes if e.error is None])
    aggregate["n_errors"] = sum(1 for e in episodes if e.error is not None)
    return EvalResult(task=task_name, aggregate=aggregate, episodes=episodes, executor=executor)


class ControllerExecutor:
    """Evaluate the controller in ``solution.py`` **in-process** and return a result.

    Controller-specific (it loads ``solution.py`` and runs a controller), hence the
    ``Controller`` prefix and the ``controllers/`` home. Used when the env is in-process
    (the ``scripted`` test backend, no socket for a subprocess to reach); real runs use
    :class:`SandboxedExecutor`. Metric computation is delegated to injected callables so
    the problem layer owns what a score means.
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
            raw = run_episodes_raw(
                self._env,
                solution_path,
                lifecycle=lifecycle,
                n_episodes=n_episodes,
                max_moves=max_moves,
            )
        except Exception as exc:  # import / attribute / syntax error in agent code
            result = EvalResult(
                task=task_name,
                error=f"{type(exc).__name__}: {exc}",
                error_category=ErrorCategory.AGENT_SOLUTION,
                executor="in_process",
            )
            _write(output_path, result)
            return result
        result = score_episodes(
            raw,
            task_name=task_name,
            compute_metrics=self._compute_metrics,
            aggregate_metrics=self._aggregate_metrics,
            executor="in_process",
        )
        _write(output_path, result)
        return result


class SandboxedExecutor:
    """Evaluate the controller in a **sandboxed subprocess**, then score it here (trusted).

    Same ``run`` surface as :class:`ControllerExecutor`, so the controller feature swaps one
    for the other transparently. The subprocess (``regact.controllers.eval_runner``, wrapped
    by ``sandbox_wrap``) drives the controller against the env over HTTP and writes raw
    outcomes; this side applies the metric callables. So untrusted controller code cannot
    fake its score nor reach beyond the box (no game data on disk, no external egress).
    """

    def __init__(
        self,
        *,
        workdir: str,
        sandbox_wrap: Callable[[list[str]], list[str]],
        compute_metrics: EpisodeMetrics | None = None,
        aggregate_metrics: AggregateMetrics | None = None,
    ) -> None:
        self._workdir = workdir
        self._wrap = sandbox_wrap
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
        """Run the eval subprocess, score its raw outcomes here, and persist the result."""
        out_dir = os.path.dirname(output_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        raw_path = os.path.join(out_dir, "episodes_raw.json")
        result = self._spawn_and_score(
            task_name, solution_path, raw_path, lifecycle, n_episodes, max_moves
        )
        _write(output_path, result)
        return result

    def _spawn_and_score(
        self,
        task_name: str,
        solution_path: str,
        raw_path: str,
        lifecycle: Lifecycle,
        n_episodes: int,
        max_moves: int,
    ) -> EvalResult:
        argv = [
            sys.executable,
            "-m",
            "regact.controllers.eval_runner",
            "--solution", os.path.abspath(solution_path),
            "--lifecycle", lifecycle.value,
            "--episodes", str(n_episodes),
            "--max-moves", str(max_moves),
            "--output", os.path.abspath(raw_path),
        ]
        tmp = os.path.join(self._workdir, "tmp")
        os.makedirs(tmp, exist_ok=True)
        env = {**os.environ, "PYTHONPATH": _src_dir(), "TMPDIR": tmp}
        proc = subprocess.run(
            self._wrap(argv), cwd=self._workdir, env=env, capture_output=True, text=True
        )
        payload = _read_json(raw_path)
        if payload is None:  # the subprocess died before writing (e.g. sandbox killed it)
            tail = (proc.stderr or "").strip()[-500:]
            return EvalResult(
                task=task_name,
                error=f"eval subprocess produced no result (rc={proc.returncode}): {tail}",
                error_category=ErrorCategory.EVAL_HARNESS,
                executor="subprocess",
            )
        if payload.get("load_error"):  # solution.py failed to import/load — the agent's fault
            return EvalResult(
                task=task_name,
                error=str(payload["load_error"]),
                error_category=ErrorCategory.AGENT_SOLUTION,
                executor="subprocess",
            )
        return score_episodes(
            payload.get("episodes", []),
            task_name=task_name,
            compute_metrics=self._compute_metrics,
            aggregate_metrics=self._aggregate_metrics,
            executor="subprocess",
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


def _src_dir() -> str:
    """Absolute path of the dir holding the ``regact`` package (PYTHONPATH for the subprocess)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)  # type: ignore[no-any-return]
    except (OSError, json.JSONDecodeError):
        return None
