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
from typing import TYPE_CHECKING, Any

from regact.config.schema import InfoMode, ObsMode
from regact.env.renderer import ObsRenderer

if TYPE_CHECKING:
    from regact.envclient.obs import Obs
    from regact.workspace.templates import TemplateFile


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

    def helper_templates(self, task_name: str) -> list[TemplateFile]:
        """Game-specific helper files dropped into the agent's workdir.

        Distinct from a feature's templates: these are problem-specific (e.g. ARC's
        action-id constants + ``complex_action`` builder). They must be import-free —
        the agent never imports the game library. Default: none.
        """
        return []

    def secret_modules(self) -> tuple[str, ...]:
        """Importable packages that ARE the game (its engine / data library).

        These are pip-installed in the same venv as the framework, so the OS sandbox —
        which allows the whole interpreter prefix for Python to run — would otherwise
        expose their source. The agent reaches the env only over HTTP and never needs
        them, so the sandbox hides their on-disk source (closing the read-the-engine
        cheat, e.g. ``sed .../arcengine/base_game.py``). Default: none.
        """
        return ()

    def render_frame(self, obs: Obs) -> Any | None:
        """Colorize one observation into an RGB frame for video, or ``None``.

        Used only when the ControllerExecutor records a submitted controller's episodes
        (never for the agent's own exploration scripts). Default: no video.
        """
        return None

    def render_obs_text(self, obs: Obs) -> str | None:
        """A compact, human-readable text rendering of one observation, or ``None``.

        Game-specific (the game knows its frame). Distinct from ``render_frame`` (RGB,
        for video).
        """
        return None

    @abstractmethod
    def compute_episode_metrics(self, final_obs: Obs, *, steps: int) -> dict[str, Any]:
        """Per-episode metrics from generic episode data only.

        Takes the universal :class:`Obs` (carrying ``reward``/``is_done``/``info``)
        and the step count — never a controller-specific rollout type — so the
        problem layer stays agnostic of how the episode was produced.
        """
        ...

    @abstractmethod
    def aggregate_episode_metrics(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        """Roll per-episode metrics into a run aggregate."""
        ...

    @abstractmethod
    def build_prompt(self, task_name: str, *, info_mode: InfoMode) -> str:
        """The game prompt for the first message, built per task and info level."""
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
