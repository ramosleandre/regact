"""Run a whole experiment: many tasks of one problem, via the Scheduler.

The single function both entry points (``run_exp`` Hydra, ``run_kaggle`` argparse)
build a :class:`RunConfig` and hand to. It builds the problem from config, expands
the task list, and runs :func:`run_task` per task through the :class:`Scheduler`
(sequential or concurrency-bounded). Returns the per-task exit reason.
"""

from __future__ import annotations

import os
from collections import Counter
from datetime import datetime

from regact.config.schema import RunConfig
from regact.obs.console import configure_console_logging, console
from regact.obs.errors import ErrorCategory, RegactError
from regact.orchestration.scheduler import Scheduler
from regact.orchestration.signals import install_stop_signal
from regact.orchestration.task import run_task
from regact.problems.base import build_problem

_STAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"  # sortable, and legal on every filesystem (no ':')


def _resolve_task_names(config: RunConfig, available: list[str]) -> list[str]:
    """Resolve and validate the experiment's problem-owned task selection."""
    selected = config.problem.tasks or available
    duplicates = sorted({task for task in selected if selected.count(task) > 1})
    if duplicates:
        raise RegactError(
            ErrorCategory.EVAL_HARNESS,
            f"problem.tasks contains duplicate tasks: {duplicates}",
        )
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise RegactError(
            ErrorCategory.EVAL_HARNESS,
            f"problem.tasks contains tasks not exposed by {config.problem.name!r}: {unknown}",
        )
    return list(selected)


def resolve_run_dir(config: RunConfig, *, output_root: str | None = None) -> str:
    """The directory one run owns: ``<output_root>/<experiment_name>/<timestamp>``.

    Every run gets its own timestamped dir, so re-running the same experiment name never
    overwrites or interleaves artifacts (logs open in ``"w"``, the scaffold is rewritten,
    and ``submission_count`` restarts at 0). Runs of one name stay grouped under it, in
    chronological order. ``output_root`` overrides the whole path (tests pass a tmp dir).
    """
    if output_root is not None:
        return os.path.abspath(output_root)
    stamp = datetime.now().strftime(_STAMP_FORMAT)
    return os.path.abspath(os.path.join(config.output_root, config.experiment_name or "run", stamp))


def _link_latest(run_dir: str) -> None:
    """Point ``<parent>/latest`` at this run, so tooling can name it without the stamp."""
    link = os.path.join(os.path.dirname(run_dir), "latest")
    try:
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.basename(run_dir), link)
    except OSError:  # a filesystem without symlinks must not fail the run
        pass


def _attempt_plan(task_names: list[str], n_attempts: int) -> list[tuple[str, int]]:
    """The run order for repeated tasks: every task's attempt A before any task's attempt A+1
    (task1@0, task2@0, ..., taskK@0, task1@1, ...), so a benchmark accrues attempts evenly across
    tasks instead of front-loading all attempts of task1."""
    return [(task, attempt) for attempt in range(max(1, n_attempts)) for task in task_names]


def _run_label(task: str, attempt: int, n_attempts: int) -> str:
    """A run's output-dir label: just the task when it runs once, else ``<task>/attempt_<n>`` (a
    subdir per attempt). The viewer still groups by each run's real ``task_name``, so attempts of a
    task aggregate together."""
    return task if n_attempts <= 1 else f"{task}/attempt_{attempt}"


async def run_experiment(config: RunConfig, *, output_root: str | None = None) -> dict[str, str]:
    """Run every task ``n_attempts_per_task`` times; return ``{run_label: exit_reason}``."""
    root = resolve_run_dir(config, output_root=output_root)
    os.makedirs(root, exist_ok=True)
    # Silence third-party INFO noise AND tee the terminal narration to <run>/run.log, so the whole
    # experiment is reviewable from the run folder (live or after it finishes).
    configure_console_logging(run_log_path=os.path.join(root, "run.log"))
    problem = build_problem(config.problem.name, config.problem.kwargs)
    task_names = _resolve_task_names(config, problem.get_task_names())
    _link_latest(root)

    n_attempts = max(1, config.n_attempts_per_task)
    plan = _attempt_plan(task_names, n_attempts)
    workers = max(1, config.parallel_workers)
    attempts_note = f" x {n_attempts} attempts" if n_attempts > 1 else ""
    console(
        f"{config.agent.name} · {config.problem.name} · {len(task_names)} task(s){attempts_note} · "
        f"sandbox={config.sandbox} · workers={workers}"
    )
    with install_stop_signal() as stop:

        async def unit(item: tuple[str, int]) -> str:
            task, attempt = item
            out_dir = os.path.join(root, _run_label(task, attempt, n_attempts))
            return await run_task(config, problem, task, output_dir=out_dir, stop=stop)

        reasons = await Scheduler(config).run(unit, plan, task_of=lambda it: it[0])
    result = {
        _run_label(task, attempt, n_attempts): str(reason)
        for (task, attempt), reason in zip(plan, reasons, strict=True)
    }
    tally = ", ".join(f"{n}x {reason}" for reason, n in Counter(result.values()).items())
    console(f"complete: {len(result)} run(s) - {tally}")
    return result
