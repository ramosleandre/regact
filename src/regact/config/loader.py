"""Build a typed :class:`RunConfig` from a plain mapping.

Both front-ends funnel through here: ``run_kaggle`` loads a YAML profile to a
dict, ``run_exp`` lets Hydra compose a dict — then this maps it to the typed
config explicitly. Doing the enum conversion by hand (rather than a structured
config) keeps it simple and avoids ``StrEnum`` round-trip surprises.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from regact.config.schema import (
    AgentConfig,
    AgentName,
    Execution,
    InfoMode,
    Lifecycle,
    LimitsConfig,
    ObsMode,
    ProblemConfig,
    RunConfig,
    SecurityConfig,
)


def run_config_from_mapping(data: Mapping[str, Any]) -> RunConfig:
    """Map a plain ``{agent, problem, limits, ...}`` mapping to a ``RunConfig``."""
    agent = dict(data.get("agent") or {})
    problem = dict(data.get("problem") or {})
    return RunConfig(
        agent=AgentConfig(
            name=AgentName(agent["name"]),
            model=agent.get("model"),
            base_url=agent.get("base_url"),
            api_key=agent.get("api_key"),
            args=dict(agent.get("args") or {}),
        ),
        problem=ProblemConfig(
            name=str(problem["name"]),
            lifecycle=Lifecycle(problem.get("lifecycle", Lifecycle.MULTI_INSTANCE)),
            obs_mode=ObsMode(problem.get("obs_mode", ObsMode.RAW)),
            info_mode=InfoMode(problem.get("info_mode", InfoMode.INFORMATIVE)),
            seed=problem.get("seed"),
            kwargs=dict(problem.get("kwargs") or {}),
        ),
        task_names=list(data.get("task_names") or []),
        features=list(data.get("features") or ["controller"]),
        tools=dict(data.get("tools") or {}),
        execution=Execution(data.get("execution", Execution.SEQUENTIAL)),
        parallel_workers=int(data.get("parallel_workers", 1)),
        limits=LimitsConfig(**dict(data.get("limits") or {})),
        security=SecurityConfig(**dict(data.get("security") or {})),
        experiment_name=data.get("experiment_name"),
        output_root=str(data.get("output_root", "experiments")),
    )
