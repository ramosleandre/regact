"""Run a whole experiment: many tasks of one problem, via the Scheduler.

The single function both entry points (``run_exp`` Hydra, ``run_kaggle`` argparse)
build a :class:`RunConfig` and hand to. It builds the problem from config, expands
the task list, and runs :func:`run_task` per task through the :class:`Scheduler`
(sequential or concurrency-bounded). Returns the per-task exit reason.
"""

from __future__ import annotations

import os
from datetime import datetime

from regact.config.schema import RunConfig
from regact.orchestration.scheduler import Scheduler
from regact.orchestration.signals import install_stop_signal
from regact.orchestration.task import run_task
from regact.problems.base import build_problem

_STAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"  # sortable, and legal on every filesystem (no ':')


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


async def run_experiment(config: RunConfig, *, output_root: str | None = None) -> dict[str, str]:
    """Run every task of the configured problem; return ``{task_name: exit_reason}``."""
    problem = build_problem(config.problem.name, config.problem.kwargs)
    task_names = config.task_names or problem.get_task_names()
    root = resolve_run_dir(config, output_root=output_root)
    os.makedirs(root, exist_ok=True)
    _link_latest(root)

    with install_stop_signal() as stop:

        async def unit(task_name: str) -> str:
            out_dir = os.path.join(root, task_name)
            return await run_task(config, problem, task_name, output_dir=out_dir, stop=stop)

        reasons = await Scheduler(config).run(unit, task_names)
    return {task: str(reason) for task, reason in zip(task_names, reasons, strict=True)}
