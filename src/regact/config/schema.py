"""The typed run config both entry points build.

Hydra (run_exp) and argparse+YAML (run_kaggle) are two front-ends that produce
the same ``RunConfig``; everything downstream consumes only this object. Every
closed choice is a ``str``-valued ``Enum``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from regact.security.runtime import SandboxRuntime


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
    walltime_s: int | None = None  # wall-clock budget for the whole task (per game)
    token_budget: int | None = None


@dataclass
class SecurityConfig:
    sandbox: SandboxRuntime = SandboxRuntime.AUTO  # which OS sandbox wraps the agent subprocess
    deny_egress: bool = False  # also block external internet (only safe when the LLM is local)
    runtime_opts: dict[str, Any] = field(default_factory=dict)  # backend extras, e.g. image=.sif


@dataclass
class RunConfig:
    """The full description of one experiment (one or many tasks)."""

    agent: AgentConfig
    problem: ProblemConfig
    task_names: list[str] = field(default_factory=list)  # empty = all games of the problem
    features: list[str] = field(default_factory=lambda: ["controller"])
    tools: dict[str, bool] = field(default_factory=dict)  # cross-cutting tool toggles
    execution: Execution = Execution.SEQUENTIAL
    parallel_workers: int = 1
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    experiment_name: str | None = None
    output_root: str = "experiments"
