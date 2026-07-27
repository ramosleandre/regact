"""The typed run config both entry points build.

Hydra (run_exp) and argparse+YAML (run_kaggle) are two front-ends that produce
the same ``RunConfig``; everything downstream consumes only this object. Every
closed choice is a ``str``-valued ``Enum``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from regact.security.runtime import SandboxRuntime

# Config fields whose values are secrets and must never be written to run artifacts.
_SECRET_FIELDS = frozenset({"api_key"})
_REDACTED = "***redacted***"


class AgentName(StrEnum):
    ALAN = "alan"
    CLAUDE = "claude"
    CODEX = "codex"
    SCRIPTED = "scripted"  # deterministic backend for tests


class Lifecycle(StrEnum):
    MULTI_INSTANCE = "multi_instance"  # fresh env per episode
    SINGLE_INSTANCE = "single_instance"  # one env per game, level-reset only


class ObsMode(StrEnum):
    RAW = "raw"
    # ascii / structured / vlm_caption land with their renderers (Later).


class InfoMode(StrEnum):
    """How much the prompt tells the agent about the task."""

    INFORMATIVE = "informative"  # full description (obs, actions, goal)
    MINIMAL = "minimal"  # the agent discovers the rules by interaction


class Execution(StrEnum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


@dataclass
class AgentConfig:
    name: AgentName
    model: str | None = None
    base_url: str | None = None  # None => use the CLI's own auth (e.g. Claude subscription)
    api_key: str | None = None
    args: dict[str, Any] = field(default_factory=dict)  # backend-specific CLI params


@dataclass
class ProblemConfig:
    name: str  # the problem family, e.g. "arc_agi" (iterates its games)
    lifecycle: Lifecycle = Lifecycle.MULTI_INSTANCE
    obs_mode: ObsMode = ObsMode.RAW
    info_mode: InfoMode = InfoMode.INFORMATIVE
    seed: int | None = None  # ignored by deterministic envs (ARC)
    kwargs: dict[str, Any] = field(default_factory=dict)  # problem-specific ctor args


@dataclass
class LimitsConfig:
    keep_alive: int = 150  # max idle agent turns before the loop gives up
    max_moves: int = 2500  # max env.step per controller rollout (eval)
    n_episodes: int = 1  # eval episodes per submission (MULTI_INSTANCE: more = better stats)
    walltime_s: int | None = None  # wall-clock budget for the whole task (per game)
    env_step_budget: int | None = None  # hard cap on env steps per env (anti-runaway; None = off)


@dataclass
class SecurityConfig:
    sandbox: SandboxRuntime = SandboxRuntime.AUTO  # which OS sandbox wraps the agent subprocess
    deny_egress: bool = False  # Block external internet except the loaded agent's declared host
    runtime_opts: dict[str, Any] = field(default_factory=dict)  # backend extras, e.g. image=.sif


@dataclass
class RunConfig:
    """The full description of one experiment (one or many tasks)."""

    agent: AgentConfig
    problem: ProblemConfig
    task_names: list[str] = field(default_factory=list)  # empty = all games of the problem
    features: list[str] = field(default_factory=lambda: ["controller"])
    execution: Execution = Execution.SEQUENTIAL
    parallel_workers: int = 1
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    record_video: bool = True
    shadow_replay: bool = False
    experiment_name: str | None = None
    output_root: str = "experiments"


def redacted_config_dict(config: RunConfig) -> dict[str, Any]:
    """Serialize a :class:`RunConfig` with secret fields masked, for run artifacts.

    ``dataclasses.asdict`` would write ``agent.api_key`` verbatim into ``config.json``;
    a configured key would then sit in the experiment dir in plaintext. This masks any
    field named in ``_SECRET_FIELDS`` (recursively) while leaving the rest untouched.
    """

    def _mask(value: Any, key: str | None = None) -> Any:
        if key in _SECRET_FIELDS and value is not None:
            return _REDACTED
        if isinstance(value, dict):
            return {k: _mask(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [_mask(v) for v in value]
        return value

    return {k: _mask(v, k) for k, v in dataclasses.asdict(config).items()}
