"""Tests for the run config schema."""

from regact.config.schema import (
    AgentConfig,
    AgentName,
    ControllerConfig,
    Lifecycle,
    ObsMode,
    ProblemConfig,
    RunConfig,
)


def test_run_config_defaults() -> None:
    cfg = RunConfig(
        agent=AgentConfig(name=AgentName.SCRIPTED),
        problem=ProblemConfig(name="arc_agi"),
    )
    assert cfg.features == {}  # no optional features by default
    assert cfg.controller == ControllerConfig()  # the always-on controller, default knobs
    assert cfg.controller.shadow_replay is False  # programmatic default (Hydra profile sets True)
    assert cfg.parallel_workers == 1
    assert cfg.problem.tasks == []
    assert cfg.problem.lifecycle is Lifecycle.MULTI_INSTANCE
    assert cfg.problem.obs_mode is ObsMode.RAW
    assert cfg.limits.max_turns > 0
    assert cfg.sandbox is False


def test_enum_string_values() -> None:
    assert AgentName.CLAUDE.value == "claude"
    assert Lifecycle.SINGLE_INSTANCE.value == "single_instance"


def test_mutable_defaults_are_not_shared() -> None:
    a = RunConfig(agent=AgentConfig(name=AgentName.ALAN), problem=ProblemConfig(name="x"))
    b = RunConfig(agent=AgentConfig(name=AgentName.ALAN), problem=ProblemConfig(name="y"))
    a.features["world_model"] = {}
    a.problem.tasks.append("g1")
    assert b.features == {}
    assert b.problem.tasks == []
