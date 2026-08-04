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
    tasks: list[str] = field(default_factory=list)  # empty = every task exposed by the problem
    lifecycle: Lifecycle = Lifecycle.MULTI_INSTANCE
    obs_mode: ObsMode = ObsMode.RAW
    info_mode: InfoMode = InfoMode.INFORMATIVE
    seed: int | None = None  # ignored by deterministic envs (ARC)
    kwargs: dict[str, Any] = field(default_factory=dict)  # problem-specific ctor args


@dataclass
class LimitsConfig:
    """Per-task run limits; each name states its scope."""

    max_turns: int = 150  # agent turns per task before the loop gives up
    max_seconds_per_task: int | None = None  # wall-clock per task, from session start
    max_actions_per_env: int | None = None  # env.step cap per env instance (from its make)


@dataclass
class RunConfig:
    """The full description of one experiment (one or many tasks)."""

    agent: AgentConfig
    problem: ProblemConfig
    # Tasks live under ``problem.tasks``; features own their own knobs (``features.<name>``),
    # including the controller's eval knobs (n_episodes, max_moves, record_video, shadow_replay).
    features: dict[str, dict[str, Any]] = field(default_factory=lambda: {"controller": {}})
    parallel_workers: int = 1  # 1 = sequential
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    sandbox: bool = False  # confine agent+eval subprocesses, deny egress; fail if no backend
    sandbox_opts: dict[str, Any] = field(default_factory=dict)  # expert: backend=..., image=...
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
