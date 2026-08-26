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
    ControllerConfig,
    InfoMode,
    Lifecycle,
    LimitsConfig,
    ObsMode,
    ProblemConfig,
    RunConfig,
)


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
    for name in ("max_turns", "max_consecutive_no_tool_turns"):
        if fields.get(name) is not None:
            fields[name] = int(fields[name])
    for name in ("max_seconds_per_task", "max_actions_per_env"):
        if name in fields:
            fields[name] = _int_or_none(fields[name])
    return LimitsConfig(**fields)


def _sandbox_bool(value: Any) -> bool:
    """``sandbox`` is a bool; reject the legacy backend-name strings loudly
    (``bool("none")`` is True, so silent coercion would invert the intent)."""
    if isinstance(value, bool) or value is None:
        return bool(value)
    raise ValueError(
        f"sandbox must be true/false (got {value!r}); to force a backend use "
        "sandbox_opts.backend=<seatbelt|bwrap>"
    )


def _features_from(raw: Any) -> dict[str, dict[str, Any]]:
    """Normalize the OPTIONAL ``features`` to ``{name: params}``; a plain name list means
    no params. Absent = no extra features (the controller is always-on core, not here)."""
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(name): dict(params or {}) for name, params in raw.items()}
    return {str(name): {} for name in raw}


def _controller_from(raw: Any) -> ControllerConfig:
    """Build ``ControllerConfig`` from the ``controller`` mapping (defaults if absent).

    ``n_episodes``/``max_moves`` may arrive as strings via env interpolation; coerce them.
    The booleans come through as real YAML/CLI bools, so they are passed through untouched.
    """
    fields: dict[str, Any] = dict(raw or {})
    for name in ("n_episodes", "max_moves"):
        if fields.get(name) is not None:
            fields[name] = int(fields[name])
    return ControllerConfig(**fields)


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
            tasks=list(problem.get("tasks") or []),
            lifecycle=Lifecycle(problem.get("lifecycle", Lifecycle.MULTI_INSTANCE)),
            obs_mode=ObsMode(problem.get("obs_mode", ObsMode.RAW)),
            info_mode=InfoMode(problem.get("info_mode", InfoMode.INFORMATIVE)),
            seed=problem.get("seed"),
            kwargs=dict(problem.get("kwargs") or {}),
        ),
        controller=_controller_from(data.get("controller")),
        features=_features_from(data.get("features")),
        parallel_workers=int(data.get("parallel_workers", 1)),
        n_attempts_per_task=int(data.get("n_attempts_per_task", 1)),
        first_obs_in_prompt=bool(data.get("first_obs_in_prompt", False)),
        limits=_limits_from(data.get("limits") or {}),
        sandbox=_sandbox_bool(data.get("sandbox", False)),
        sandbox_opts=dict(data.get("sandbox_opts") or {}),
        experiment_name=data.get("experiment_name"),
        output_root=str(data.get("output_root", "experiments")),
    )
