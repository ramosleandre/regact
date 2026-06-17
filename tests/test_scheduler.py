"""Unit tests: the Scheduler (sequential/parallel + single-instance guard)."""

import asyncio

import pytest

from regact.config.schema import (
    AgentConfig,
    AgentName,
    Execution,
    Lifecycle,
    ProblemConfig,
    RunConfig,
)
from regact.obs.errors import RegactError
from regact.orchestration.scheduler import Scheduler


def _config(execution: Execution, workers: int, lifecycle: Lifecycle) -> RunConfig:
    return RunConfig(
        agent=AgentConfig(name=AgentName.SCRIPTED),
        problem=ProblemConfig(name="p", lifecycle=lifecycle),
        execution=execution,
        parallel_workers=workers,
    )


async def test_sequential_runs_in_order() -> None:
    order: list[str] = []

    async def unit(name: str) -> str:
        order.append(name)
        return name.upper()

    cfg = _config(Execution.SEQUENTIAL, 1, Lifecycle.MULTI_INSTANCE)
    out = await Scheduler(cfg).run(unit, ["a", "b", "c"])
    assert order == ["a", "b", "c"]
    assert out == ["A", "B", "C"]


async def test_parallel_runs_all_tasks() -> None:
    async def unit(name: str) -> str:
        await asyncio.sleep(0)
        return name

    cfg = _config(Execution.PARALLEL, 3, Lifecycle.MULTI_INSTANCE)
    out = await Scheduler(cfg).run(unit, ["a", "b", "c", "d"])
    assert sorted(out) == ["a", "b", "c", "d"]


def test_validate_rejects_duplicate_game_in_single_instance_parallel() -> None:
    cfg = _config(Execution.PARALLEL, 2, Lifecycle.SINGLE_INSTANCE)
    with pytest.raises(RegactError):
        Scheduler(cfg)._validate(["g", "g"])


def test_validate_allows_distinct_games_in_single_instance_parallel() -> None:
    cfg = _config(Execution.PARALLEL, 2, Lifecycle.SINGLE_INSTANCE)
    Scheduler(cfg)._validate(["g1", "g2"])  # distinct => fine


def test_validate_allows_duplicates_when_sequential() -> None:
    cfg = _config(Execution.SEQUENTIAL, 1, Lifecycle.SINGLE_INSTANCE)
    Scheduler(cfg)._validate(["g", "g"])  # workers==1 => no concurrency conflict
