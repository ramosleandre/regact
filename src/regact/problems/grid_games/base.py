"""Level-based grid games: the shared problem layer for games played on a small colour grid.

A game supplies only its engine (:class:`GridGame`) and its tasks (:class:`GridTask`: an
ordered pack of levels plus a reference action count per level). This layer turns them into a
regact problem with the conventions of ARC-AGI-3, so prompts, metrics, the viewer and the
protocols treat every grid game alike:

- ``obs.frame`` is a one-element list holding the current grid (each game says what a cell is);
- ``obs.info`` has ``state`` (``NOT_FINISHED`` / ``WIN``), ``levels_completed`` and
  ``win_levels``;
- action 0 is RESET, an ordinary game action that restarts the current level (no framework
  reset command); the other ids are the game's own;
- completing the last level wins and ends the episode;
- RHAE and LRHAE-Uncapped compare actions per level with the task's reference (for example
  optimal solutions), as ARC compares them with human baselines.

The prompt never names the game: the agent must discover its rules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from regact.config.schema import InfoMode, ObsMode
from regact.env.renderer import ObsRenderer, jsonify
from regact.envclient.errors import InvalidActionError
from regact.envclient.obs import Obs
from regact.obs.errors import ErrorCategory, RegactError
from regact.problems.base import BaseProblem, observation_prompt
from regact.problems.palette import ARC_PALETTE
from regact.workspace.templates import TemplateFile

_PROMPT = Path(__file__).parents[1] / "prompts" / "grid_game.md"
_RENDER_SCALE = 24  # px per cell in the video; levels are small (about 10x10)
_HEX = "0123456789abcdef"
RESET = 0


class GridGame(ABC):
    """One task's engine: plays the levels of a pack, one at a time."""

    @abstractmethod
    def load_level(self, index: int) -> None:
        """Start level ``index`` from its initial layout."""

    @abstractmethod
    def apply(self, action: int) -> None:
        """Play one non-RESET action on the current level."""

    @abstractmethod
    def solved(self) -> bool:
        """Whether the current level is complete."""

    @abstractmethod
    def grid(self) -> list[list[Any]]:
        """The current level as rows of cells."""

    def close(self) -> None:  # noqa: B027 - most engines hold nothing to release
        """Release the engine's resources, if it holds any."""


@dataclass(frozen=True)
class GridTask:
    """A pack of levels played in order; ``reference_actions`` scores efficiency per level."""

    key: str
    levels: tuple[Any, ...]
    reference_actions: tuple[int, ...]


class GridGameEnv:
    """The gym-like environment the server wraps: a pack of levels, RESET as action 0."""

    def __init__(self, game: GridGame, *, n_levels: int, actions: tuple[int, ...]) -> None:
        self._game = game
        self._n_levels = n_levels
        self._actions = actions
        self._level = 0

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self._level = 0
        self._game.load_level(0)
        return self._game.grid(), self._info()

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        action_id = action.get("action") if isinstance(action, dict) else action
        if type(action_id) is not int or (action_id != RESET and action_id not in self._actions):
            raise InvalidActionError(
                f"expected an action id in {[RESET, *self._actions]}; got {action!r}"
            )
        if self._level >= self._n_levels:
            raise InvalidActionError("the game is already won")
        if action_id == RESET:
            self._game.load_level(self._level)
            return self._game.grid(), 0.0, False, False, self._info()
        self._game.apply(action_id)
        if not self._game.solved():
            return self._game.grid(), 0.0, False, False, self._info()
        self._level += 1
        if self._level < self._n_levels:
            self._game.load_level(self._level)
            return self._game.grid(), 0.0, False, False, self._info()
        return self._game.grid(), 1.0, True, False, self._info()

    def render(self) -> Any:
        return self._game.grid()

    def close(self) -> None:
        self._game.close()

    def _info(self) -> dict[str, Any]:
        won = self._level >= self._n_levels
        return {
            "available_actions": [] if won else [RESET, *self._actions],
            "state": "WIN" if won else "NOT_FINISHED",
            "levels_completed": self._level,
            "win_levels": self._n_levels,
        }


class GridRenderer(ObsRenderer):
    """The native grid as a one-element frame list, so ``obs.frame[-1]`` is the current grid."""

    def render(self, native_obs: object, info: dict[str, Any] | None) -> Obs:
        info = dict(info or {})
        return Obs(
            frame=[jsonify(native_obs)],
            available_actions=list(info.get("available_actions", [])),
            info=info,
        )


def _milestone_detector(env: Any) -> list[str]:
    last = (getattr(env, "last_obs", None) and env.last_obs.info) or {}
    prev = (getattr(env, "prev_obs", None) and env.prev_obs.info) or {}
    out: list[str] = []
    done = last.get("levels_completed", 0)
    if done > prev.get("levels_completed", 0):
        out.append(f"level completed ({done}/{last.get('win_levels', '?')})")
    if last.get("state") == "WIN":
        out.append("game won")
    return out


def _current_grid(frame: Any) -> list[list[int]] | None:
    if isinstance(frame, list) and frame and isinstance(frame[-1], list) and frame[-1]:
        grid = frame[-1]
        if isinstance(grid[0], list):
            return grid
    return None


def _optimal_names(scores: dict[str, float]) -> dict[str, float]:
    """ARC's RHAE and LRHAE-Uncapped against an optimal player rather than a human: ROAE and
    LROAE (Relative Optimal Action Efficiency). No agent beats the optimum, so no cap applies."""
    names = {"rhae": "roae", "lrhae_uncapped": "lroae"}
    return {names.get(key, key): value for key, value in scores.items()}


class GridGameProblem(BaseProblem):
    """A grid game as a regact problem; subclasses set ``name`` and supply tasks and engine."""

    main_metrics = ("roae", "lroae", "relative_env_actions")
    progress_label = "Levels completed"
    actions: tuple[int, ...] = (1, 2, 3, 4)
    frame_note = "each cell is an integer, one per colour (0-15)"  # what a cell is, in the prompt
    # Describes the game's own actions in the prompt, without naming their effects.
    action_note = (
        "Actions 1-4 are directional (typically up, down, left, right); discover their "
        "effects by interaction."
    )

    def __init__(self, *, max_levels: int | None = None) -> None:
        if max_levels is not None and int(max_levels) < 1:
            raise ValueError("max_levels must be a positive integer or null")
        self._max_levels = int(max_levels) if max_levels is not None else None

    @abstractmethod
    def tasks(self) -> dict[str, GridTask]:
        """Every task of the game, by key."""

    @abstractmethod
    def make_game(self, task: GridTask) -> GridGame:
        """A fresh engine for ``task``."""

    def _task(self, task_name: str) -> GridTask:
        try:
            return self.tasks()[task_name]
        except KeyError:
            raise RegactError(
                ErrorCategory.EVAL_HARNESS,
                f"unknown {self.name} task {task_name!r}; known: {sorted(self.tasks())}",
            ) from None

    def _n_levels(self, task: GridTask) -> int:
        n = len(task.levels)
        return min(n, self._max_levels) if self._max_levels is not None else n

    def _reference(self, task_name: str) -> tuple[int, ...]:
        task = self._task(task_name)
        return task.reference_actions[: self._n_levels(task)]

    def task_actions(self, task_name: str) -> tuple[int, ...]:
        """The game's own action ids for ``task_name`` (RESET excluded)."""
        return self.actions

    def make_env(self, task_name: str) -> Any:
        task = self._task(task_name)
        return GridGameEnv(
            self.make_game(task),
            n_levels=self._n_levels(task),
            actions=self.task_actions(task_name),
        )

    def get_task_names(self) -> list[str]:
        return list(self.tasks())

    def reset_commands(self) -> dict[str, str]:
        return {}  # RESET is a game action

    def counts_as_action(self, action: Any) -> bool:
        return action != RESET  # as ARC's scorecard: RESET is never scored

    def obs_renderer(self, task_name: str, *, mode: ObsMode) -> ObsRenderer:
        if mode not in (ObsMode.RAW, ObsMode.RAW_LAST_FRAME_ONLY):
            raise RegactError(
                ErrorCategory.ENV_RUNTIME, f"{self.name}: obs_mode {mode!r} not supported"
            )
        return GridRenderer()

    def progress_value(self, info: dict[str, Any]) -> float | None:
        return int(info.get("levels_completed", 0) or 0)

    def progress_reference(self, task_name: str) -> list[tuple[float, float]]:
        points, total = [], 0
        for level, actions in enumerate(self._reference(task_name), start=1):
            total += actions
            points.append((float(total), float(level)))
        return points

    def milestone_kind(self, milestone: str) -> str:
        if milestone == "game won" or milestone.startswith("level completed ("):
            return "progress"
        return "event"

    def milestone_detector(self, task_name: str) -> Any:
        return _milestone_detector

    def helper_templates(
        self,
        task_name: str,
        *,
        info_mode: InfoMode = InfoMode.INFORMATIVE,
        direct_interaction: bool = True,
    ) -> list[TemplateFile]:
        names = "\n".join(f"ACTION{a} = {a}" for a in self.task_actions(task_name))
        content = (
            '"""Action ids of this game; valid ids are also in `obs.available_actions`."""\n\n'
            f"RESET = {RESET}  # restarts the current level\n{names}\n"
        )
        return [TemplateFile("framework/game_helper.py", content)]

    def render_frame(self, obs: Obs) -> Any | None:
        grid = _current_grid(obs.frame)
        if grid is None:
            return None
        import numpy as np

        cells = np.clip(np.asarray(grid, dtype=np.int64), 0, len(ARC_PALETTE) - 1)
        rgb = np.asarray(ARC_PALETTE, dtype=np.uint8)[cells]
        return np.repeat(np.repeat(rgb, _RENDER_SCALE, axis=0), _RENDER_SCALE, axis=1)

    def render_obs_text(self, obs: Obs) -> str | None:
        grid = _current_grid(obs.frame)
        if grid is None:
            return None
        info = obs.info or {}
        header = (
            f"state={info.get('state', '?')}  "
            f"levels_completed={info.get('levels_completed', 0)}/{info.get('win_levels', '?')}  "
            f"available_actions={obs.available_actions}"
        )
        rows = [
            "".join(_HEX[c] if type(c) is int and 0 <= c < 16 else "?" for c in r) for r in grid
        ]
        return f"{header}\n\n" + "\n".join(rows)

    def enumerate_actions(self, obs: Obs) -> Iterable[Any]:
        return [a for a in obs.available_actions if a != RESET]

    def exploration_score(self, aggregate: dict[str, Any]) -> float:
        return float(aggregate.get("mean_levels_completed", 0) or 0)

    def compute_episode_metrics(self, final_obs: Obs, *, steps: int) -> dict[str, Any]:
        info = final_obs.info or {}
        completed = info.get("levels_completed", 0)
        total = info.get("win_levels", 0)
        return {
            "success": info.get("state") == "WIN",
            "steps": steps,
            "levels_completed": completed,
            "win_levels": total,
            "level_completion_rate": (completed / total) if total else 0.0,
        }

    def failure_metrics(self, *, steps: int) -> dict[str, Any]:
        return {
            "success": False,
            "steps": steps,
            "levels_completed": 0,
            "win_levels": 0,
            "level_completion_rate": 0.0,
        }

    def is_perfect(self, aggregate: dict[str, Any]) -> bool:
        return float(aggregate.get("win_rate", 0.0)) >= 1.0

    def aggregate_episode_metrics(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(episodes)
        if not n:
            return {
                "n_episodes": 0,
                "mean_levels_completion_rate": 0.0,
                "win_rate": 0.0,
                "mean_levels_completed": 0.0,
            }
        return {
            "n_episodes": n,
            "mean_levels_completion_rate": sum(
                e.get("level_completion_rate", 0.0) for e in episodes
            )
            / n,
            "win_rate": sum(bool(e.get("success")) for e in episodes) / n,
            "mean_levels_completed": sum(e.get("levels_completed", 0) for e in episodes) / n,
        }

    def derived_submission_metrics(
        self, task_name: str, raw_episodes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        from regact.problems.arc_agi.scoring import rhae_from_results

        rhae = rhae_from_results(
            {"episodes": raw_episodes}, baseline_actions=list(self._reference(task_name))
        )
        return _optimal_names(rhae.scores()) if rhae is not None else {}

    def derived_trace_metrics(
        self, task_name: str, trace: list[tuple[int, dict[str, Any]]]
    ) -> dict[str, Any]:
        from regact.problems.arc_agi.scoring import actions_per_level_from_trace, rhae_score

        reference = self._reference(task_name)
        levels = [(actions, int(info.get("levels_completed", 0) or 0)) for actions, info in trace]
        per_level = actions_per_level_from_trace(levels)
        return _optimal_names(
            rhae_score(
                baseline_actions=reference,
                actions_per_level=per_level,
                levels_completed=len(per_level),
                total_levels=len(reference),
            ).scores()
        )

    def build_prompt(
        self,
        task_name: str,
        *,
        info_mode: InfoMode,
        obs_mode: ObsMode = ObsMode.RAW,
        direct_interaction: bool = True,
    ) -> str:
        if info_mode is InfoMode.MINIMAL:
            source = (
                "from your own scripts with `make_env()`"
                if direct_interaction
                else "in recorded experience from `framework.data_api`"
            )
            return observation_prompt(
                "# Game\n\nDiscover the rules by interaction. Inspect `obs.frame` and "
                f"`obs.available_actions` {source}; the framework tells you nothing more "
                "about this task.",
                direct_interaction=direct_interaction,
            )
        task = self._task(task_name)
        body = (
            _PROMPT.read_text(encoding="utf-8")
            .replace(
                "{interaction_note}",
                "`make_env()` gives you an isolated client with a gym-like interface "
                "(importable with `from framework.make_env import make_env`). Each `obs` is:"
                if direct_interaction
                else "Recorded experience is accessible through `framework.data_api`. Each "
                "observation is:",
            )
            .replace("{action_note}", self.action_note)
            .replace("{frame_note}", self.frame_note)
            .replace("{levels_to_win}", f"Levels to win: {self._n_levels(task)}")
            .rstrip()
        )
        return observation_prompt(body, direct_interaction=direct_interaction)

    def config_kwargs(self) -> dict[str, Any]:
        return {"max_levels": self._max_levels}
