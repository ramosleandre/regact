"""Run a whole experiment: many tasks of one problem, via the Scheduler.

The single function both entry points (``run_exp`` Hydra, ``run_kaggle`` argparse)
build a :class:`RunConfig` and hand to. It builds the problem from config, expands
the task list, and runs :func:`run_task` per task through the :class:`Scheduler`
(sequential or concurrency-bounded). Returns the per-task exit reason.
"""

from __future__ import annotations

import os

from regact.config.schema import RunConfig
from regact.orchestration.scheduler import Scheduler
from regact.orchestration.signals import install_stop_signal
from regact.orchestration.task import run_task
from regact.problems.base import build_problem


async def run_experiment(config: RunConfig, *, output_root: str | None = None) -> dict[str, str]:
    """Run every task of the configured problem; return ``{task_name: exit_reason}``."""
    problem = build_problem(config.problem.name, config.problem.kwargs)
    task_names = config.task_names or problem.get_task_names()
    root = output_root or os.path.join(config.output_root, config.experiment_name or "run")
    root = os.path.abspath(root)

    with install_stop_signal() as stop:

        async def unit(task_name: str) -> str:
            out_dir = os.path.join(root, task_name)
            return await run_task(config, problem, task_name, output_dir=out_dir, stop=stop)

        reasons = await Scheduler(config).run(unit, task_names)
    return {task: str(reason) for task, reason in zip(task_names, reasons, strict=True)}
