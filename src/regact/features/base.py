"""The feature/extension model.

A ``Feature`` is a self-contained capability the agent uses or builds. It bundles
four parts: workdir templates, a prompt fragment, tools, and hooks. A run selects
a set of features; the bootstrap, prompt builder, tool surface, and teardown
assemble themselves from that set. New features add a file here and register a
name; the core is untouched. Features are independent — there is no dependency
graph; ``controller`` is always present as the base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from regact.config.schema import Lifecycle
from regact.envclient.client import EnvClient
from regact.obs.result import EvalResult
from regact.session.state import ExperimentState
from regact.tools.base import Tool
from regact.workspace.templates import TemplateFile

__all__ = [
    "EvalResult",
    "Feature",
    "FeatureContext",
    "Hook",
    "HookPhase",
    "RunDeps",
    "TemplateFile",
    "build_features",
    "register_feature",
]


@dataclass
class FeatureContext:
    """Static inputs a feature needs to render itself (templates + prompt)."""

    problem_name: str
    task_name: str
    workdir: str


@dataclass
class RunDeps:
    """Run-scoped dependencies the orchestrator owns and hands to a feature.

    Distinct from :class:`FeatureContext`: these exist only once a run is live, so
    they feed the runtime side of a feature (``tools`` and ``hooks``), never the
    static rendering side. The feature reads them; it never stores them.

    Carries the **agnostic** ``EnvClient`` (not a controller executor) — a
    controller feature builds its own ``ControllerExecutor`` from it, so this base
    type stays free of controller-specific imports.
    """

    experiment: ExperimentState
    env_client: EnvClient
    lifecycle: Lifecycle
    solution_path: str
    submissions_dir: str
    n_episodes: int = 1
    max_moves: int = 400
    compute_episode_metrics: Callable[..., dict[str, Any]] | None = None
    aggregate_episode_metrics: Callable[..., dict[str, Any]] | None = None


class HookPhase(StrEnum):
    """When the loop fires a hook. Only points the loop actually observes (env
    steps live behind HTTP, so they are not loop phases)."""

    TEARDOWN = "teardown"  # session end, on every non-aborted exit path
    # POST_SUBMIT lands with the anti-cheat scan (Block 10).


class Hook(ABC):
    """Framework-run work the loop fires at a phase (re-score, verify, scan…).

    A hook captures whatever it needs via ``Feature.hooks(deps)``, so ``run`` takes
    no arguments. It returns an :class:`EvalResult` to record, or ``None``.
    """

    phase: HookPhase

    @abstractmethod
    async def run(self) -> EvalResult | None: ...


class Feature(ABC):
    """A composable capability.

    Static side (``templates``, ``prompt_fragment``) takes a
    :class:`FeatureContext`; runtime side (``tools``, ``hooks``) takes
    :class:`RunDeps` supplied by the orchestrator. The feature stays stateless.
    """

    name: str

    @abstractmethod
    def templates(self, ctx: FeatureContext) -> list[TemplateFile]:
        """Files to scaffold into the workdir."""
        ...

    @abstractmethod
    def prompt_fragment(self, ctx: FeatureContext) -> str | None:
        """Markdown appended to the agent's first message (or ``None``)."""
        ...

    @abstractmethod
    def tools(self, deps: RunDeps) -> list[Tool]:
        """Tools exposed to the agent, wired with the run's dependencies."""
        ...

    @abstractmethod
    def hooks(self, deps: RunDeps) -> list[Hook]:
        """Framework-run hooks fired by the loop at their phase (finalize, verify…)."""
        ...


# Concrete features register here (Block 6 adds ``controller``); the config names
# features by string, so it never imports a feature class and there is no cycle.
_REGISTRY: dict[str, Callable[[], Feature]] = {}


def register_feature(name: str, factory: Callable[[], Feature]) -> None:
    """Bind a feature name to a zero-arg factory."""
    _REGISTRY[name] = factory


def build_features(names: list[str]) -> list[Feature]:
    """Resolve feature names to instances via the registry."""
    _load_builtins()
    try:
        return [_REGISTRY[name]() for name in names]
    except KeyError as exc:
        raise ValueError(
            f"unknown feature {exc.args[0]!r}; registered: {sorted(_REGISTRY)}"
        ) from exc


def _load_builtins() -> None:
    """Import the built-in feature modules so they self-register on first use."""
    from regact.features import controller  # noqa: F401
