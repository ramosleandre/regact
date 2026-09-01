"""Render a small PNG preview of a task's environment, for the viz x-axis thumbnails.

Freshly rendered from the env (not pulled from a run's artifacts): a MiniGrid task via its
gym ``rgb_array`` render, an ARC task via its first frame painted with the ARC-AGI-3 palette.
Downscaled to a thumbnail. The viz caches the result per task, so this runs once per task.
"""

from __future__ import annotations

import io

import numpy as np

# The official ARC-AGI-3 16-colour palette (mirrors problems/arc_agi/problem.py _PALETTE).
_ARC_PALETTE = np.array(
    [
        (255, 255, 255),
        (204, 204, 204),
        (153, 153, 153),
        (102, 102, 102),
        (51, 51, 51),
        (0, 0, 0),
        (229, 58, 163),
        (255, 123, 204),
        (249, 60, 49),
        (30, 147, 255),
        (136, 216, 241),
        (255, 220, 0),
        (255, 133, 27),
        (146, 18, 49),
        (79, 204, 48),
        (163, 86, 214),
    ],
    dtype=np.uint8,
)
_THUMB_MAX = 96  # max side of the rendered thumbnail (px)


def render_task_png(task_name: str) -> bytes:
    """A small PNG preview of ``task_name``'s starting state. Raises if the task can't render."""
    rgb = _minigrid_rgb(task_name) if task_name.startswith("MiniGrid") else _arc_rgb(task_name)
    return _encode(rgb)


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
