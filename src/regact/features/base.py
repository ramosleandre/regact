"""The feature/extension model.

A ``Feature`` is a self-contained capability the agent uses or builds. It bundles
up to four parts: workdir templates, a prompt fragment, tools, and an eval/verify
hook. A run selects a set of features; the bootstrap, prompt builder, tool
surface, and eval assemble themselves from that set. New features add a file here
and register a name; the core is untouched. Features are independent — there is
no dependency graph; ``controller`` is always present as the base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from regact.config.schema import Lifecycle
from regact.obs.result import EvalResult
from regact.session.state import ExperimentState
from regact.tools.base import Tool

if TYPE_CHECKING:
    from regact.eval.executor import EvalExecutor


@dataclass
class TemplateFile:
    """A file dropped into the agent's workdir at a relative path."""

    relpath: str  # e.g. "code_library/world_model.py"
    content: str  # rendered template body (signatures the agent fills)


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
    they feed the runtime side of a feature (``tools`` and ``eval_hooks``), never
    the static rendering side. The feature reads them; it never stores them.
    """

    experiment: ExperimentState
    executor: EvalExecutor
    lifecycle: Lifecycle
    solution_path: str
    submissions_dir: str
    n_episodes: int = 1
    max_moves: int = 400


class EvalHook(ABC):
    """A verification run by the framework against the agent's deliverable."""

    @abstractmethod
    def verify(self, *, workdir: str, session: object) -> EvalResult: ...


class Feature(ABC):
    """A composable capability.

    Static side (``templates``, ``prompt_fragment``) takes a
    :class:`FeatureContext`; runtime side (``tools``, ``eval_hooks``) takes
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
    def eval_hooks(self, deps: RunDeps) -> list[EvalHook]:
        """Framework-run verification hooks (e.g. shadow-replay anti-cheat)."""
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
