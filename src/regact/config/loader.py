"""Build a typed :class:`RunConfig` from a plain mapping.

Both front-ends funnel through here: ``run_kaggle`` loads a YAML profile to a
dict, ``run_exp`` lets Hydra compose a dict — then this maps it to the typed
config explicitly. Doing the enum conversion by hand (rather than a structured
config) keeps it simple and avoids ``StrEnum`` round-trip surprises.
"""

from __future__ import annotations

import os
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
    ProtocolConfig,
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
    if fields.pop("max_actions_per_env", None) is not None:
        raise ValueError(
            "limits.max_actions_per_env was removed: use limits.max_actions_per_episode, "
            "which renews on every reset (the old cap renewed only on a new make_env)"
        )
    for name in (
        "max_turns_per_task",
        "max_consecutive_no_tool_turns",
        "max_tool_calls",
        "max_seconds_per_task",
        "max_actions_per_episode",
        "max_actions_per_task",
        "experiment_deadline_unix",
    ):
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

    ``n_episodes``/``max_moves``/``n_videos`` may arrive as strings via env interpolation;
    coerce them. The booleans come through as real YAML/CLI bools, passed through untouched.
    """
    fields: dict[str, Any] = dict(raw or {})
    for name in ("n_episodes", "max_moves", "n_videos"):
        if fields.get(name) is not None:
            fields[name] = int(fields[name])
    return ControllerConfig(**fields)


def _protocol_from(raw: Any) -> ProtocolConfig:
    """Select the old workflow when absent; accept a name or a protocol mapping."""
    if raw is None:
        return ProtocolConfig()
    if isinstance(raw, str):
        return ProtocolConfig(name=raw)
    if not isinstance(raw, Mapping):
        raise ValueError("protocol must be a name or a mapping containing name")
    fields = dict(raw)
    name = fields.pop("name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("protocol.name must be a non-empty string")
    return ProtocolConfig(name=name, options=fields)


# Launch facts the LAUNCHER cannot pass: sbatch only returns a job id after submission, so the
# running process is the first thing that knows it.
_ENV_LAUNCH_FIELDS = {
    "slurm_job_id": "SLURM_JOB_ID",
    "slurm_nodelist": "SLURM_JOB_NODELIST",
}


def launch_facts_from_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The ``launch`` fields only the running process can see, omitting any that is unset.

    Omitted rather than recorded as ``None`` so a laptop run's record stays empty and a missing
    field always means "not under Slurm", never "under Slurm and unreported".
    """
    source = os.environ if env is None else env
    return {key: source[var] for key, var in _ENV_LAUNCH_FIELDS.items() if source.get(var)}


def run_config_from_mapping(data: Mapping[str, Any]) -> RunConfig:
    """Map a plain ``{agent, problem, limits, ...}`` mapping to a ``RunConfig``."""
    agent = dict(data.get("agent") or {})
    problem = dict(data.get("problem") or {})
    config = RunConfig(
        agent=AgentConfig(
            name=AgentName(agent["name"]),
            model=agent.get("model"),
            base_url=agent.get("base_url"),
            api_key=agent.get("api_key"),
            args=dict(agent.get("args") or {}),
            vision=bool(agent.get("vision", False)),
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
        protocol=_protocol_from(data.get("protocol")),
        controller=_controller_from(data.get("controller")),
        features=_features_from(data.get("features")),
        parallel_workers=int(data.get("parallel_workers", 1)),
        n_attempts_per_task=int(data.get("n_attempts_per_task", 1)),
        first_obs_in_prompt=bool(data.get("first_obs_in_prompt", False)),
        flagging_warning_cap=int(data.get("flagging_warning_cap", 3)),
        dry_run=bool(data.get("dry_run", False)),
        resume=data.get("resume"),
        resume_any_version=bool(data.get("resume_any_version", False)),
        limits=_limits_from(data.get("limits") or {}),
        sandbox=_sandbox_bool(data.get("sandbox", False)),
        sandbox_opts=dict(data.get("sandbox_opts") or {}),
        experiment_name=data.get("experiment_name"),
        output_root=str(data.get("output_root", "experiments")),
        launch=dict(data.get("launch") or {}),
    )
    remote = config.agent.model == "remote" and config.agent.args.get("backend") == "scripted"
    if remote and config.sandbox and config.sandbox_opts.get("network_isolation", True):
        raise ValueError(
            "agent=alan_remote needs sandbox_opts.network_isolation=false: its HTTP endpoint "
            "listens inside the sandbox's network namespace, which the host cannot reach"
        )
    return config
