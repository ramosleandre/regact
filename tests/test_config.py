"""Tests for the run config schema."""

from regact.config.schema import (
    AgentConfig,
    AgentName,
    Execution,
    Lifecycle,
    ObsMode,
    ProblemConfig,
    RunConfig,
)
from regact.security.runtime import SandboxRuntime


def test_run_config_defaults() -> None:
    cfg = RunConfig(
        agent=AgentConfig(name=AgentName.SCRIPTED),
        problem=ProblemConfig(name="arc_agi"),
    )
    assert cfg.features == ["controller"]
    assert cfg.execution is Execution.SEQUENTIAL
    assert cfg.parallel_workers == 1
    assert cfg.problem.lifecycle is Lifecycle.MULTI_INSTANCE
    assert cfg.problem.obs_mode is ObsMode.RAW
    assert cfg.limits.keep_alive > 0
    assert cfg.limits.max_moves > 0
    assert cfg.security.runtime is SandboxRuntime.AUTO


def test_enum_string_values() -> None:
    assert AgentName.CLAUDE.value == "claude"
    assert Lifecycle.SINGLE_INSTANCE.value == "single_instance"


def test_mutable_defaults_are_not_shared() -> None:
    a = RunConfig(agent=AgentConfig(name=AgentName.ALAN), problem=ProblemConfig(name="x"))
    b = RunConfig(agent=AgentConfig(name=AgentName.ALAN), problem=ProblemConfig(name="y"))
    a.features.append("world_model")
    assert b.features == ["controller"]
