"""MiniGrid problem.

Wraps gymnasium MiniGrid envs. Reset variation depends on the env id (fixed-layout
``Empty-*`` vs procedurally generated ones), so ``seed`` matters here (unlike ARC).
``gymnasium``/``minigrid`` are imported lazily inside :meth:`make_env`, so this
module imports cleanly without the ``minigrid`` extra installed. Prompt text lives
in ``problems/prompts/minigrid.md``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from regact.config.schema import HelperConfig, InfoMode, ObsMode
from regact.env.renderer import ObsRenderer, jsonify
from regact.envclient.obs import Obs
from regact.obs.errors import ErrorCategory, RegactError
from regact.problems.base import BaseProblem, register_problem
from regact.problems.minigrid.tasks import ALL_MINIGRID_TASKS
from regact.workspace.templates import TemplateFile

# MiniGrid's tile renderer draws on a pygame Surface, and ``import minigrid`` pulls pygame in.
# On a headless HPC node pygame's SDL video init blocks forever waiting for a display, so
# ``render_frame`` (the eval video) hangs on the first frame -> the eval ReadTimeouts and the run
# scores nothing. Force the headless drivers here (module load happens at build_problem, ahead of
# the lazy minigrid/pygame import in make_env/warmup/render_frame). setdefault so an explicit
# override wins, and the regact imports above never pull minigrid, so this still runs first.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

_PROMPTS = Path(__file__).parents[1] / "prompts"
_PROMPT = _PROMPTS / "minigrid.md"  # intro + mechanics + completion, with a {obs_section} hole
_PROMPT_FULL = _PROMPTS / "minigrid_full.md"  # fully-observable grid fragment
_PROMPT_PARTIAL = _PROMPTS / "minigrid_partial.md"  # egocentric partial-view fragment
_TILE_SIZE = 24  # px per cell in the video render
_WFC_PATTERNS = Path(__file__).parent / "wfc_patterns"  # vendored WFC sample-pattern PNGs

# Upstream env docstrings mix task flavour (mission/rewards) with a NATIVE obs/action
# description that conflicts with our regact-serialized contract; keep only the flavour.
_DOCSTRING_KEEP = ("Description", "Mission Space", "Rewards", "Termination")

# MiniGrid encoding constants, dropped into the workdir (informative modes) so the agent
# imports them instead of guessing. Hardcoded, NOT re-exported from ``minigrid`` - that
# package is a secret module, hidden from the agent sandbox. Values verified against it.
_MINIGRID_HELPER = '''\
"""MiniGrid encoding constants - import these instead of guessing the values.

Each ``obs.frame["image"]`` cell is a triple ``[object, color, state]``. Door ``state`` is
meaningful (open/closed/locked); other objects use 0.
"""

OBJECT_TO_IDX = {
    "unseen": 0, "empty": 1, "wall": 2, "floor": 3, "door": 4,
    "key": 5, "ball": 6, "box": 7, "goal": 8, "lava": 9, "agent": 10,
}
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
STATE_TO_IDX = {"open": 0, "closed": 1, "locked": 2}

IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}
IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

# Agent facing (obs.frame["direction"], and the agent cell's state channel in full obs):
DIR_RIGHT, DIR_DOWN, DIR_LEFT, DIR_UP = 0, 1, 2, 3
'''


class MiniGridRenderer(ObsRenderer):
    """Pass the MiniGrid obs through, made JSON-safe; actions come from info."""

    def render(self, native_obs: object, info: dict[str, Any] | None) -> Obs:
        info = info or {}
        return Obs(
            frame=jsonify(native_obs),
            available_actions=list(info.get("available_actions", [])),
            info={k: jsonify(v) for k, v in info.items()},
        )


class _ActionInfoShim:
    """Wrap a gym env so each obs carries ``available_actions`` in its info dict.

    A plain delegating shim (not a ``gymnasium.Wrapper``) so gymnasium stays a
    lazy import: the server's ``WrappedEnv`` only needs reset/step/render/close.
    """

    def __init__(self, env: Any) -> None:
        self._env = env

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed)
        return obs, self._augment(info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs, reward, terminated, truncated, self._augment(info)

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    def _augment(self, info: dict[str, Any] | None) -> dict[str, Any]:
        info = dict(info or {})
        info["available_actions"] = list(range(int(self._env.action_space.n)))
        return info


class MiniGridProblem(BaseProblem):
    """The complete MiniGrid task family; experiment config selects a subset."""

    name = "minigrid"

    def __init__(self, *, fully_obs: bool = False) -> None:
        self._fully_obs = fully_obs

    def make_env(self, task_name: str) -> Any:
        import gymnasium
        import minigrid  # noqa: F401  (importing registers the MiniGrid env ids)

        if task_name not in ALL_MINIGRID_TASKS:
            raise ValueError(f"unknown MiniGrid task {task_name!r}")
        if "WFC" in task_name:
            _ensure_wfc_patterns()  # repoint the preset at our vendored pattern before make()
        env = gymnasium.make(task_name)
        if self._fully_obs:
            from minigrid.wrappers import FullyObsWrapper

            env = FullyObsWrapper(env)
        return _ActionInfoShim(env)

    def get_task_names(self) -> list[str]:
        return list(ALL_MINIGRID_TASKS)

    def obs_renderer(self, task_name: str, *, mode: ObsMode) -> ObsRenderer:
        if mode is not ObsMode.RAW:
            raise RegactError(
                ErrorCategory.ENV_RUNTIME, f"minigrid: obs_mode {mode!r} not supported yet"
            )
        return MiniGridRenderer()

    def render_frame(self, obs: Obs) -> Any | None:
        """Re-render the symbolic obs into an RGB frame via MiniGrid's own tile renderer."""
        image = obs.frame.get("image") if isinstance(obs.frame, dict) else None
        if image is None:
            return None
        import numpy as np
        from minigrid.core.constants import OBJECT_TO_IDX
        from minigrid.core.grid import Grid

        arr = np.asarray(image, dtype=np.uint8)
        grid, _ = Grid.decode(arr)
        agent = np.argwhere(arr[:, :, 0] == OBJECT_TO_IDX["agent"])
        if len(agent):  # fully-obs: the agent is encoded in the grid + state channel
            pos = (int(agent[0][0]), int(agent[0][1]))
            direction = int(arr[pos[0], pos[1], 2])
        else:  # partial egocentric view: agent at bottom-center, facing up
            pos = (arr.shape[0] // 2, arr.shape[1] - 1)
            direction = 3
        return grid.render(_TILE_SIZE, pos, direction)

    def compute_episode_metrics(self, final_obs: Obs, *, steps: int) -> dict[str, Any]:
        """Generic inputs only: terminated-with-reward = success (truncation is not)."""
        reward = final_obs.reward or 0.0
        return {
            "success": bool(final_obs.is_done and reward > 0),
            "steps": steps,
            "reward": reward,
        }

    def aggregate_episode_metrics(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        if not episodes:
            return {"n_episodes": 0, "success_rate": 0.0, "mean_steps": 0.0, "mean_reward": 0.0}
        n = len(episodes)
        return {
            "n_episodes": n,
            "success_rate": sum(bool(e.get("success")) for e in episodes) / n,
            "mean_steps": sum(e.get("steps", 0) for e in episodes) / n,
            "mean_reward": sum(e.get("reward", 0.0) for e in episodes) / n,
        }

    def build_prompt(
        self, task_name: str, *, info_mode: InfoMode, obs_mode: ObsMode = ObsMode.RAW
    ) -> str:
        # obs_mode is accepted for interface parity; MiniGrid renders one obs mode (RAW).
        if info_mode is InfoMode.MINIMAL:
            return (
                f"# Game: MiniGrid ({task_name})\n\n"
                "Discover the rules by interaction. Inspect `obs.frame` and "
                "`obs.available_actions` from your own scripts with `make_env()`; "
                "the framework tells you nothing more about this task."
            )
        # The observation section is mode-specific: full-map vs egocentric are genuinely
        # different and must never both appear (agents mistook one for the other).
        obs_fragment = (_PROMPT_FULL if self._fully_obs else _PROMPT_PARTIAL).read_text(
            encoding="utf-8"
        )
        prompt = (
            _PROMPT.read_text(encoding="utf-8")
            .replace("{task}", task_name)
            .replace("{obs_section}", obs_fragment.strip())
        )
        if info_mode is InfoMode.INFORMATIVE_DOCSTRING and (doc := _upstream_docstring(task_name)):
            prompt += (
                "\n\n## Upstream task documentation\n\n"
                "The env library's own notes for this task (its generic obs/action wording may "
                "differ from the regact contract above, which is authoritative):\n\n"
                f"{doc}"
            )
        return prompt

    def helper_templates(
        self,
        task_name: str,
        *,
        info_mode: InfoMode = InfoMode.INFORMATIVE,
        helper: HelperConfig | None = None,
    ) -> list[TemplateFile]:
        """Ship the encoding constants in informative modes; minimal mode hands out nothing
        (the agent must discover the encodings by interaction). ``helper`` is unused here."""
        if info_mode is InfoMode.MINIMAL:
            return []
        return [TemplateFile("code_library/minigrid_helper.py", _MINIGRID_HELPER)]

    def warmup(self) -> None:
        # Preimport the heavy libs so the first make_env (server-side) is instant - the import
        # is the ReadTimeout risk on a shared HPC node (gym/minigrid over Lustre).
        import gymnasium  # noqa: F401
        import minigrid  # noqa: F401

        _ensure_wfc_patterns()  # cheap + idempotent; no-op when WFC deps/patterns are absent

    def secret_modules(self) -> tuple[str, ...]:
        """Hide the ``minigrid`` engine from the agent + eval sandboxes.

        MiniGrid is pip-installed in the framework venv, so without this the sandbox (which
        binds the interpreter prefix so Python can run) would expose it - and a nasty agent
        could ``import minigrid; gymnasium.make(<task>)`` to RECONSTRUCT the exact env in-process,
        bypassing the HTTP-only boundary entirely (read layouts, solve offline, replay). Denying
        read on the package dir makes ``import minigrid`` fail, which also breaks
        ``gymnasium.make("MiniGrid-*")`` (the id only registers once ``minigrid`` is imported).
        The env is reached only over HTTP, so nothing legitimate in the sandbox needs it:
        ``render_frame`` runs on the trusted parent side, and the eval child never imports it.
        """
        return ("minigrid",)

    def config_kwargs(self) -> dict[str, Any]:
        return {"fully_obs": self._fully_obs}


def _ensure_wfc_patterns() -> None:
    """Repoint the WFC presets at regact's vendored sample-pattern images.

    minigrid ships the WFC preset configs but NOT their pattern PNGs (``envs/wfc/patterns/*.png``),
    so a WFC reset would ``FileNotFoundError``. We vendor those images and set each preset's
    (mutable) ``pattern_path`` to the vendored copy. Best-effort + idempotent: touches only presets
    whose pattern we vendored, and never raises if the WFC config is unavailable.
    """
    try:
        from minigrid.envs.wfc.config import WFC_PRESETS
    except Exception:
        return
    for cfg in WFC_PRESETS.values():
        vendored = _WFC_PATTERNS / cfg.pattern_path.name
        if vendored.exists():
            cfg.pattern_path = vendored


def _upstream_docstring(task_name: str) -> str | None:
    """The env library's own task docstring (mission / rewards / termination), or ``None``.

    Resolved via the gymnasium registry entry point WITHOUT constructing the env, and imported
    lazily: this runs in the trusted orchestrator (which can import ``minigrid``), never in the
    agent sandbox where it is hidden. The generic obs/action sections are dropped - they
    describe the native gym interface and conflict with our regact-serialized contract.
    """
    try:
        import importlib
        import inspect

        import gymnasium
        import minigrid  # noqa: F401  (registers the MiniGrid ids)

        entry = gymnasium.spec(task_name).entry_point
        if not isinstance(entry, str) or ":" not in entry:
            return None
        module_name, class_name = entry.split(":", 1)
        klass = getattr(importlib.import_module(module_name), class_name)
        return _keep_doc_sections(inspect.cleandoc(klass.__doc__ or "")) or None
    except Exception:
        return None


def _keep_doc_sections(doc: str) -> str:
    """Keep only the ``## <header>`` sections whose header is in ``_DOCSTRING_KEEP``."""
    kept: list[str] = []
    keeping = False
    for line in doc.splitlines():
        header = line.strip()
        if header.startswith("## "):
            keeping = any(header[3:].strip().startswith(k) for k in _DOCSTRING_KEEP)
        if keeping:
            kept.append(line)
    return "\n".join(kept).strip()


register_problem("minigrid", lambda kwargs: MiniGridProblem(**kwargs))
