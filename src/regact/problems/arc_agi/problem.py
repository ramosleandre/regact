"""ARC-AGI-3 problem.

Wraps the ``arc-agi`` arcade in OFFLINE mode: games are loaded from a local
``environnement/`` directory (committed, no API key), server-side. The agent only
ever sees the normalized :class:`Obs` over HTTP and sends JSON int/dict actions;
the gym shim here maps them to native ``GameAction`` — so the agent never imports
the game library. ``arc_agi``/``arcengine`` are imported lazily inside ``make_env``
(and the shim), so this module imports cleanly without the ``arc`` extra.

Lifecycle is enforced elsewhere: ARC runs ``single_instance`` (one make per game,
RESET is a level reset), guarded by ``SingleInstancePolicy`` — not here.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from regact.config.schema import InfoMode, ObsMode
from regact.env.renderer import ObsRenderer, jsonify
from regact.envclient.obs import Obs
from regact.obs.errors import ErrorCategory, RegactError
from regact.problems.arc_agi.tasks import ArcAgiTask, discover_tasks
from regact.problems.base import BaseProblem, register_problem
from regact.workspace.templates import TemplateFile

logger = logging.getLogger(__name__)

_PROMPT = Path(__file__).parents[1] / "prompts" / "arc_agi.md"
_DEFAULT_DIR = os.environ.get("ARC_ENVIRONMENTS_DIR", "environnement")

# Cell value (0-15) -> RGB for the video render — the official ARC-AGI-3 palette
# (mirrors ``arc_agi.rendering.COLOR_MAP``).
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 255, 255),  # 0  white
    (204, 204, 204),  # 1  off-white
    (153, 153, 153),  # 2  neutral light
    (102, 102, 102),  # 3  neutral
    (51, 51, 51),  # 4  off black
    (0, 0, 0),  # 5  black
    (229, 58, 163),  # 6  magenta
    (255, 123, 204),  # 7  magenta light
    (249, 60, 49),  # 8  red
    (30, 147, 255),  # 9  blue
    (136, 216, 241),  # 10 blue light
    (255, 220, 0),  # 11 yellow
    (255, 133, 27),  # 12 orange
    (146, 18, 49),  # 13 maroon
    (79, 204, 48),  # 14 green
    (163, 86, 214),  # 15 purple
)
_RENDER_SCALE = 8


# --------------------------------------------------------------------------- #
# Server-side gym shim: JSON int/dict actions -> native GameAction
# --------------------------------------------------------------------------- #
class _ArcGymShim:
    """Adapt an arc ``EnvironmentWrapper`` to the gym-like interface WrappedEnv drives.

    ``step`` accepts what the agent sends as JSON: an int action id, or a dict
    ``{"action": id, "data": {"x", "y"}}`` for the click action (ACTION6). Both are
    mapped to a native ``GameAction`` here, so the conversion stays server-side.
    """

    def __init__(self, arc_env: Any) -> None:
        from arcengine import GameAction, GameState

        self._env = arc_env
        self._GameAction = GameAction
        self._GameState = GameState
        self._last: Any = getattr(arc_env, "_last_response", None)

    def current(self) -> tuple[Any, dict[str, Any]] | None:
        """The observation without taking an action (the frame from creation/last step)."""
        return None if self._last is None else (self._last, self._info(self._last))

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        obs = self._env.reset()
        self._last = obs
        return obs, self._info(obs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        game_action, data = self._decode(action)
        obs = self._env.step(game_action, data=data)
        if obs is not None:
            self._last = obs
        reward, terminated = self._outcome(obs)
        return obs, reward, terminated, False, self._info(obs)

    def render(self) -> Any:
        """The most recent frame grid (for video); None if unavailable."""
        frame = getattr(self._last, "frame", None)
        if not frame:
            return None
        return frame[-1] if isinstance(frame, (list, tuple)) else frame

    def close(self) -> None:
        """No-op — the Arcade owns the env lifecycle."""

    def _decode(self, action: Any) -> tuple[Any, dict[str, Any] | None]:
        if isinstance(action, dict):
            return self._GameAction.from_id(int(action["action"])), action.get("data")
        return self._GameAction.from_id(int(action)), None

    def _outcome(self, obs: Any) -> tuple[float, bool]:
        if obs is None:
            return 0.0, False
        if obs.state == self._GameState.WIN:
            return 1.0, True
        if obs.state == self._GameState.GAME_OVER:
            return 0.0, True
        return 0.0, False

    def _info(self, obs: Any) -> dict[str, Any]:
        if obs is None:
            return {}
        return {
            "available_actions": list(getattr(obs, "available_actions", [])),
            "state": obs.state.name if getattr(obs, "state", None) is not None else None,
            "levels_completed": getattr(obs, "levels_completed", 0),
            "win_levels": getattr(obs, "win_levels", 0),
        }


# --------------------------------------------------------------------------- #
# Renderer: native FrameDataRaw -> JSON-safe Obs (no arc import needed)
# --------------------------------------------------------------------------- #
class ArcRenderer(ObsRenderer):
    """Turn the native frame (list of 64x64 int grids) into a JSON-safe ``Obs``.

    Import-free: it reads ``native_obs.frame`` and the scalar metadata the shim
    already placed in ``info``, so it needs no arc library to construct.
    """

    def render(self, native_obs: object, info: dict[str, Any] | None) -> Obs:
        info = dict(info or {})
        frame = getattr(native_obs, "frame", native_obs)
        return Obs(
            frame=jsonify(frame),
            available_actions=list(info.get("available_actions", [])),
            info=info,
        )


def _milestone_detector(env: Any) -> list[str]:
    """Emit a milestone on level completion / WIN / GAME_OVER (reads the wrapper's obs)."""
    last_info = (getattr(env, "last_obs", None) and env.last_obs.info) or {}
    prev_info = (getattr(env, "prev_obs", None) and env.prev_obs.info) or {}
    out: list[str] = []
    cur = last_info.get("levels_completed", 0)
    if cur > prev_info.get("levels_completed", 0):
        out.append(f"level completed ({cur}/{last_info.get('win_levels', '?')})")
    state = last_info.get("state")
    if state == "WIN":
        out.append("game won")
    elif state == "GAME_OVER":
        out.append("game over")
    return out


# Per-action description blocks, assembled from the ids a game actually exposes (see
# _actions_for_ids); ACTION5/7 are special and not always present.
_KEYBOARD_ACTIONS = (
    "Directional actions are integer ids (see `obs.available_actions`): typically "
    "ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right."
)
_CLICK_ACTIONS = (
    "The click action ACTION6 takes x,y coordinates (0-63), passed as "
    '`{"action": 6, "data": {"x": 32, "y": 32}}` — or use `complex_action(x, y)` '
    "from `code_library/arc_agi_helper.py`."
)
_ACTION5 = (
    "ACTION5 is a special action whose effect is game-specific (often an interact/"
    "space-like action) — discover what it does by interaction."
)
_ACTION7 = (
    "ACTION7 is a special action whose effect is game-specific (sometimes it undoes "
    "the last move) — discover what it does by interaction."
)
_DIRECTIONAL_IDS = frozenset({1, 2, 3, 4})


def _actions_for_ids(available: Iterable[int]) -> str:
    """Action text for exactly the ids a game exposes — each block included only when its
    action is present in ``available`` (the live ``obs.available_actions``)."""
    ids = set(available)
    blocks: list[str] = []
    if ids & _DIRECTIONAL_IDS:
        blocks.append(_KEYBOARD_ACTIONS)
    if 5 in ids:
        blocks.append(_ACTION5)
    if 6 in ids:
        blocks.append(_CLICK_ACTIONS)
    if 7 in ids:
        blocks.append(_ACTION7)
    return "\n\n".join(blocks)


_HELPER = '''\
"""ARC-AGI helpers (import-free): action ids + the click-action builder.

Use these with the env client in your scripts and controller, e.g.::

    from framework.make_env import make_env
    from code_library.arc_agi_helper import ACTION4, complex_action
    env = make_env()
    obs = env.step(ACTION4)              # a directional action
    obs = env.step(complex_action(32, 32))  # a click at (x=32, y=32)

Valid ids for the current game are in ``obs.available_actions``. ACTION6 is click.
"""

RESET = 0
ACTION1 = 1
ACTION2 = 2
ACTION3 = 3
ACTION4 = 4
ACTION5 = 5
ACTION6 = 6  # click — needs coordinates
ACTION7 = 7


def complex_action(x: int, y: int) -> dict:
    """Payload for the click action: ``env.step(complex_action(x, y))``."""
    return {"action": ACTION6, "data": {"x": x, "y": y}}
'''


_HEX = "0123456789abcdef"


def _current_grid(frame: Any) -> list[list[int]] | None:
    """The current 64x64 grid from an ``Obs.frame``: the last grid if a stack is given."""
    if not isinstance(frame, list) or not frame:
        return None
    if isinstance(frame[0], list) and frame[0] and isinstance(frame[0][0], list):
        frame = frame[-1]  # a stack of grids -> the current (last) one
    if isinstance(frame, list) and frame and isinstance(frame[0], list):
        return frame
    return None


class ArcAgiProblem(BaseProblem):
    """ARC-AGI-3 as a regact problem (offline, local game data)."""

    name = "arc_agi"

    def __init__(
        self,
        *,
        environments_dir: str = _DEFAULT_DIR,
        operation_mode: str = "offline",
    ) -> None:
        self._dir = environments_dir
        self._operation_mode = operation_mode
        self._arcade: Any = None
        self._tasks: dict[str, ArcAgiTask] = discover_tasks(environments_dir)
        logger.info(
            "ArcAgiProblem: %d games discovered in %r (mode=%s)",
            len(self._tasks),
            environments_dir,
            operation_mode,
        )

    def _arcade_instance(self) -> Any:
        if self._arcade is None:
            import arc_agi

            self._arcade = arc_agi.Arcade(
                operation_mode=arc_agi.OperationMode(self._operation_mode),
                environments_dir=self._dir,
            )
        return self._arcade

    def _task(self, task_name: str) -> ArcAgiTask:
        try:
            return self._tasks[task_name]
        except KeyError:
            raise RegactError(
                ErrorCategory.EVAL_HARNESS,
                f"unknown ARC game {task_name!r}; known: {sorted(self._tasks)}",
            ) from None

    def make_env(self, task_name: str) -> Any:
        task = self._task(task_name)
        arc_env = self._arcade_instance().make(task.game_id)
        if arc_env is None:
            raise RegactError(
                ErrorCategory.ENV_RUNTIME,
                f"Arcade.make({task.game_id!r}) returned None (check environments_dir)",
            )
        return _ArcGymShim(arc_env)

    def get_task_names(self) -> list[str]:
        return list(self._tasks)

    def obs_renderer(self, task_name: str, *, mode: ObsMode) -> ObsRenderer:
        if mode is not ObsMode.RAW:
            raise RegactError(
                ErrorCategory.ENV_RUNTIME, f"arc_agi: obs_mode {mode!r} not supported yet"
            )
        return ArcRenderer()

    def milestone_detector(self, task_name: str) -> Any:
        return _milestone_detector

    def helper_templates(self, task_name: str) -> list[TemplateFile]:
        return [TemplateFile("code_library/arc_agi_helper.py", _HELPER)]

    def secret_modules(self) -> tuple[str, ...]:
        return ("arcengine", "arc_agi")

    def render_frame(self, obs: Obs) -> Any | None:
        """Colorize the current 64x64 grid into an upscaled RGB frame for video."""
        grid = _current_grid(obs.frame)
        if grid is None:
            return None
        import numpy as np

        cells = np.clip(np.asarray(grid, dtype=np.int64), 0, len(_PALETTE) - 1)
        rgb = np.asarray(_PALETTE, dtype=np.uint8)[cells]
        s = _RENDER_SCALE
        return np.repeat(np.repeat(rgb, s, axis=0), s, axis=1)

    def render_obs_text(self, obs: Obs) -> str | None:
        """A compact text view of the current frame: a header (state, levels, available
        actions) over the 64x64 grid rendered one hex char per cell (0-f)."""
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
            "".join(_HEX[c] if isinstance(c, int) and 0 <= c < 16 else "?" for c in row)
            for row in grid
        ]
        body = f"{header}\n\n" + "\n".join(rows)
        actions = _actions_for_ids(obs.available_actions)
        return f"{body}\n\nActions available now:\n\n{actions}" if actions else body

    def compute_episode_metrics(self, final_obs: Obs, *, steps: int) -> dict[str, Any]:
        info = final_obs.info or {}
        return {
            "success": info.get("state") == "WIN",
            "steps": steps,
            "levels_completed": info.get("levels_completed", 0),
            "win_levels": info.get("win_levels", 0),
        }

    def aggregate_episode_metrics(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        if not episodes:
            return {"n_episodes": 0, "success_rate": 0.0, "mean_levels_completed": 0.0}
        n = len(episodes)
        return {
            "n_episodes": n,
            "success_rate": sum(bool(e.get("success")) for e in episodes) / n,
            "mean_levels_completed": sum(e.get("levels_completed", 0) for e in episodes) / n,
        }

    def build_prompt(self, task_name: str, *, info_mode: InfoMode) -> str:
        task = self._task(task_name)
        if info_mode is InfoMode.MINIMAL:
            return (
                f"# Game: ARC-AGI-3 ({task.title})\n\n"
                "Discover the rules by interaction. Inspect `obs.frame` and "
                "`obs.available_actions` from your own scripts with `make_env()`; "
                "the framework tells you nothing more about this task."
            )
        parts = [_PROMPT.read_text(encoding="utf-8").replace("{task}", task.title)]
        parts.append(f"**Game id**: `{task.game_id}`")
        if task.win_levels is not None:
            parts.append(f"**Levels to win**: {task.win_levels}")
        if task.baseline_actions:
            total = sum(task.baseline_actions)
            parts.append(f"**Human baseline**: {total} actions total.")
        # The actions are described live in the first observation (render_obs_text), from
        # the real available_actions, so they are not duplicated statically here.
        return "\n\n".join(parts)

    def config_kwargs(self) -> dict[str, Any]:
        return {"environments_dir": self._dir, "operation_mode": self._operation_mode}


register_problem("arc_agi", lambda kwargs: ArcAgiProblem(**kwargs))
