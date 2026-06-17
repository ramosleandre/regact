"""MiniGrid problem.

Wraps gymnasium MiniGrid envs. Stochastic, so ``seed`` matters here (unlike ARC).
``gymnasium``/``minigrid`` are imported lazily inside :meth:`make_env`, so this
module imports cleanly without the ``minigrid`` extra installed. Prompt text lives
in ``problems/prompts/minigrid.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from regact.config.schema import InfoMode, ObsMode
from regact.env.renderer import ObsRenderer, jsonify
from regact.envclient.obs import Obs
from regact.obs.errors import ErrorCategory, RegactError
from regact.problems.base import BaseProblem, register_problem

_PROMPT = Path(__file__).parent / "prompts" / "minigrid.md"
_DEFAULT_ENV_ID = "MiniGrid-Empty-5x5-v0"


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
    """A MiniGrid task family (one configured env id)."""

    name = "minigrid"

    def __init__(self, *, env_id: str = _DEFAULT_ENV_ID, fully_obs: bool = False) -> None:
        self._env_id = env_id
        self._fully_obs = fully_obs

    def make_env(self, task_name: str) -> Any:
        import gymnasium
        import minigrid  # noqa: F401  (importing registers the MiniGrid env ids)

        env = gymnasium.make(self._env_id)
        if self._fully_obs:
            from minigrid.wrappers import FullyObsWrapper

            env = FullyObsWrapper(env)
        return _ActionInfoShim(env)

    def get_task_names(self) -> list[str]:
        return [self._env_id]

    def obs_renderer(self, task_name: str, *, mode: ObsMode) -> ObsRenderer:
        if mode is not ObsMode.RAW:
            raise RegactError(
                ErrorCategory.ENV_RUNTIME, f"minigrid: obs_mode {mode!r} not supported yet"
            )
        return MiniGridRenderer()

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

    def build_prompt(self, task_name: str, *, info_mode: InfoMode) -> str:
        if info_mode is InfoMode.MINIMAL:
            return (
                f"# Game: MiniGrid ({task_name})\n\n"
                "Discover the rules by interaction. Inspect `obs.frame` and "
                "`obs.available_actions` from your own scripts with `make_env()`; "
                "the framework tells you nothing more about this task."
            )
        return _PROMPT.read_text(encoding="utf-8").replace("{task}", task_name)

    def config_kwargs(self) -> dict[str, Any]:
        return {"env_id": self._env_id, "fully_obs": self._fully_obs}


register_problem("minigrid", lambda kwargs: MiniGridProblem(**kwargs))
