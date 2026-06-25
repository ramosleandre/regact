"""Parallel run scheduler.

Runs many tasks, each fully isolated — its own env server, workdir, and agent
session — so there is no shared mutable state and parallelism is safe. Replaces
Hydra's launcher (Hydra only composes config; this runs it). Sequential when
workers == 1; otherwise an ``asyncio.Semaphore`` bounds concurrency (the shared
LLM endpoint is the bottleneck, not the CPU).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from regact.config.schema import Execution, Lifecycle, RunConfig
from regact.obs.errors import ErrorCategory, RegactError

Unit = Callable[[str], Awaitable[object]]


class Scheduler:
    """Schedule isolated per-task units, capped by a concurrency limit."""

    def __init__(self, config: RunConfig) -> None:
        self._config = config

    def _workers(self) -> int:
        if self._config.execution is Execution.SEQUENTIAL:
            return 1
        return max(1, self._config.parallel_workers)

    def _validate(self, task_names: list[str]) -> None:
        """Reject impossible parallelism under SINGLE_INSTANCE.

        A game can be made only once, so two workers can never share one game.
        Parallel work must fan out over DISTINCT games; a duplicated game (or a
        single game) in SINGLE_INSTANCE cannot run concurrently.
        """
        if self._workers() == 1:
            return
        single_instance = self._config.problem.lifecycle is Lifecycle.SINGLE_INSTANCE
        if single_instance and len(set(task_names)) != len(task_names):
            raise RegactError(
                ErrorCategory.EVAL_HARNESS,
                "single_instance parallel runs require distinct games (a game is made once)",
            )

    async def run(self, unit: Unit, task_names: list[str]) -> list[object]:
        """Run ``unit(task_name)`` for each task, sequential or concurrency-bounded."""
        self._validate(task_names)
        workers = self._workers()
        if workers == 1:
            return [await unit(name) for name in task_names]

        semaphore = asyncio.Semaphore(workers)

        async def _bounded(name: str) -> object:
            async with semaphore:
                try:
                    return await unit(name)
                except Exception as exc:
                    return f"task_error: {type(exc).__name__}: {exc}"

        return list(await asyncio.gather(*(_bounded(name) for name in task_names)))
