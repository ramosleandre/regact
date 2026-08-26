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
from typing import Any

from regact.config.schema import Lifecycle, RunConfig
from regact.obs.errors import ErrorCategory, RegactError

# A work item is opaque to the scheduler (a task name, or a (task, attempt) pair); ``task_of``
# recovers the game name it targets, for the single-instance parallelism check.
Unit = Callable[[Any], Awaitable[object]]


class Scheduler:
    """Schedule isolated per-task units, capped by a concurrency limit."""

    def __init__(self, config: RunConfig) -> None:
        self._config = config

    def _workers(self) -> int:
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

    async def run(
        self, unit: Unit, items: list[Any], *, task_of: Callable[[Any], str] = lambda i: i
    ) -> list[object]:
        """Run ``unit(item)`` for each item, sequential or concurrency-bounded."""
        self._validate([task_of(i) for i in items])
        workers = self._workers()
        if workers == 1:
            return [await unit(item) for item in items]

        semaphore = asyncio.Semaphore(workers)

        async def _bounded(item: Any) -> object:
            async with semaphore:
                try:
                    return await unit(item)
                except Exception as exc:
                    return f"task_error: {type(exc).__name__}: {exc}"

        return list(await asyncio.gather(*(_bounded(item) for item in items)))
