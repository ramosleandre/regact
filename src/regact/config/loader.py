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
from regact.security.runtime import SandboxRuntime


def _limits_from(raw: Mapping[str, Any]) -> LimitsConfig:
    """Build ``LimitsConfig`` coercing numeric fields to int.

    Values may arrive as strings — env-var interpolation (``${oc.env:VAR,default}``)
    yields a string when the variable is set. Coerce so the loop's comparisons never
    hit ``float >= str``. ``None``/empty stays ``None`` for the optional fields.
    """

    def _int_or_none(value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    fields: dict[str, Any] = dict(raw)
    for name in ("keep_alive", "max_moves", "n_episodes"):
        if name in fields and fields[name] is not None:
            fields[name] = int(fields[name])
    for name in ("walltime_s", "env_step_budget"):
        if name in fields:
            fields[name] = _int_or_none(fields[name])
    return LimitsConfig(**fields)


def run_config_from_mapping(data: Mapping[str, Any]) -> RunConfig:
    """Map a plain ``{agent, problem, limits, ...}`` mapping to a ``RunConfig``."""
    agent = dict(data.get("agent") or {})
    problem = dict(data.get("problem") or {})
    sec = dict(data.get("security") or {})
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
        execution=Execution(data.get("execution", Execution.SEQUENTIAL)),
        parallel_workers=int(data.get("parallel_workers", 1)),
        limits=_limits_from(data.get("limits") or {}),
        security=SecurityConfig(
            sandbox=SandboxRuntime(sec.get("sandbox", SandboxRuntime.AUTO)),
            deny_egress=bool(sec.get("deny_egress", False)),
            require_sandbox=bool(sec.get("require_sandbox", False)),
            runtime_opts=dict(sec.get("runtime_opts") or {}),
        ),
        record_video=bool(data.get("record_video", True)),
        shadow_replay=bool(data.get("shadow_replay", False)),
        experiment_name=data.get("experiment_name"),
        output_root=str(data.get("output_root", "experiments")),
    )
