"""The game/problem interface.

A problem owns everything game-specific: how to build the native env, the task
list, the obs renderers it supports, metrics, milestones, and its prompt
fragment. The native env it returns must be gym-like (``reset()`` -> obs or
``(obs, info)``; ``step(action)`` -> a 4- or 5-tuple), since the server-side
``WrappedEnv`` drives it. New games register via :func:`build_problem`; each
backend is imported lazily so this module needs no game library installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from regact.config.schema import ObsMode
from regact.env.renderer import ObsRenderer


class BaseProblem(ABC):
    """Base class for all problems (games)."""

    name: str

    @abstractmethod
    def make_env(self, task_name: str) -> Any:
        """Construct the native env for a task (gym.Env, ARC arcade env, ...)."""
        ...

    @abstractmethod
    def get_task_names(self) -> list[str]:
        """The games/levels this problem iterates."""
        ...

    @abstractmethod
    def obs_renderer(self, task_name: str, *, mode: ObsMode) -> ObsRenderer:
        """The server-side renderer for the requested obs mode."""
        ...

    def milestone_detector(self, task_name: str) -> Callable[[Any], list[str]] | None:
        """Optional per-step milestone detector for the wrapper."""
        return None

    @abstractmethod
    def compute_episode_metrics(self, env: Any) -> dict[str, Any]:
        """Per-episode metrics (success/steps/reward | levels_completed, ...)."""
        ...

    @abstractmethod
    def aggregate_episode_metrics(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        """Roll per-episode metrics into a run aggregate."""
        ...

    @abstractmethod
    def prompt_fragment(self, task_name: str) -> str:
        """The game-family description shown to the agent."""
        ...

    @abstractmethod
    def config_kwargs(self) -> dict[str, Any]:
        """Kwargs needed to rebuild this problem on the trusted eval side."""
        ...


# Problem names live in the config (a free string); classes live here, imported
# lazily on first build so a base install needs no gymnasium / arc library.
_REGISTRY: dict[str, Callable[[dict[str, Any]], BaseProblem]] = {}


def register_problem(name: str, factory: Callable[[dict[str, Any]], BaseProblem]) -> None:
    """Bind a problem name to a factory taking its config kwargs."""
    _REGISTRY[name] = factory


def build_problem(name: str, kwargs: dict[str, Any]) -> BaseProblem:
    """Instantiate a registered problem by name."""
    _load_builtins()
    try:
        return _REGISTRY[name](kwargs)
    except KeyError:
        raise ValueError(f"unknown problem {name!r}; registered: {sorted(_REGISTRY)}") from None


def _load_builtins() -> None:
    """Import built-in problem modules so they self-register (no game lib needed to import)."""
    from regact.problems import arc_agi, minigrid  # noqa: F401
