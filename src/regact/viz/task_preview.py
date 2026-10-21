"""Render a small PNG preview of a task's environment.

Rendered from the env (not pulled from a run's artifacts): a MiniGrid task via its gym
``rgb_array`` render, an ARC task or a grid game via its first frame painted with the ARC-AGI-3
palette, downscaled to a thumbnail. Run OFFLINE by scripts/gen_task_previews.py to pre-populate
static/icons_tasks/; the viz loads those PNGs as static files and never builds an env at serve time.
"""

from __future__ import annotations

import io

import numpy as np

from regact.problems.palette import ARC_PALETTE

_ARC_PALETTE = np.array(ARC_PALETTE, dtype=np.uint8)
_THUMB_MAX = 96  # max side of the rendered thumbnail (px)


def render_task_png(task_name: str) -> bytes:
    """A small PNG preview of ``task_name``'s starting state. Raises if the task can't render."""
    if task_name.startswith("MiniGrid"):
        rgb = _minigrid_rgb(task_name)
    elif (game := _grid_game_of(task_name)) is not None:
        rgb = _grid_game_rgb(game, task_name)
    else:
        rgb = _arc_rgb(task_name)
    return _encode(rgb)


def _grid_game_of(task_name: str):
    from regact.problems.base import build_problem
    from regact.problems.grid_games import GRID_GAMES

    for name in GRID_GAMES:
        problem = build_problem(name, {})
        if task_name in problem.get_task_names():
            return problem
    return None


def _grid_game_rgb(problem, task_name: str) -> np.ndarray:
    from regact.config.schema import ObsMode

    native, info = problem.make_env(task_name).reset()
    return problem.render_frame(
        problem.obs_renderer(task_name, mode=ObsMode.RAW).render(native, info)
    )


def _minigrid_rgb(task_name: str) -> np.ndarray:
    import gymnasium as gym

    from regact.problems.minigrid.problem import _ensure_wfc_patterns

    _ensure_wfc_patterns()  # WFC presets need their vendored pattern PNGs; no-op otherwise
    env = gym.make(task_name, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        return np.asarray(env.render(), dtype=np.uint8)
    finally:
        env.close()


def _arc_rgb(task_name: str) -> np.ndarray:
    from regact.problems.base import build_problem

    env = build_problem("arc_agi", {}).make_env(task_name)
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    frame = np.clip(np.asarray(obs.frame[-1], dtype=np.int64), 0, len(_ARC_PALETTE) - 1)
    return _ARC_PALETTE[frame]


def _encode(rgb: np.ndarray) -> bytes:
    from PIL import Image

    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), "RGB")
    w, h = img.size
    scale = _THUMB_MAX / max(w, h)
    if scale < 1:  # NEAREST keeps grid cells crisp when shrinking
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
