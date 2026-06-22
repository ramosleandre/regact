"""Uniform wrapper around a native env (server-side).

Adds an action counter, uniform ``reset``/``step`` returning a rendered ``Obs``,
derived ``is_done``, and milestone draining. Ported and hardened from GameAgents
``envs/wrapper.py`` (no silent excepts; unexpected step arity raises).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from regact.env.renderer import ObsRenderer
from regact.envclient.obs import Action, Obs
from regact.obs.errors import ErrorCategory, RegactError


class WrappedEnv:
    """Thin, uniform wrapper over a native gym/arcade env."""

    def __init__(
        self,
        native_env: Any,
        *,
        task_name: str,
        renderer: ObsRenderer,
        milestone_detector: Callable[[WrappedEnv], list[str]] | None = None,
        action_adapter: Callable[[Action], Any] | None = None,
        record_frames: bool = False,
    ) -> None:
        self._native = native_env
        self.task_name = task_name
        self._renderer = renderer
        self._milestone_detector = milestone_detector
        self._action_adapter = action_adapter
        self._record_frames = record_frames
        self.action_count: int = 0
        self.last_obs: Obs | None = None
        self.prev_obs: Obs | None = None
        self.last_reward: float | None = None
        self.last_terminated: bool = False
        self.last_truncated: bool = False
        self.last_info: dict[str, Any] | None = None
        self.frame_trace: list[Any] = []
        self._pending_milestones: list[str] = []
        self._capture_initial()

    def _capture_initial(self) -> None:
        """Capture the env's initial observation if it exposes one before any action.

        Some envs (e.g. ARC) have a playable frame at creation; gym envs do not, and
        ``last_obs`` stays ``None`` until ``reset``.
        """
        peek = getattr(self._native, "current", None)
        result = peek() if callable(peek) else None
        if result is None:
            return
        native_obs, info = result
        self.last_info = info
        self.last_obs = self.prev_obs = self._render(native_obs, info, reward=None, done=False)

    def reset(self, *, seed: int | None = None) -> Obs:
        result = self._native.reset(seed=seed) if seed is not None else self._native.reset()
        if isinstance(result, tuple) and len(result) == 2:
            native_obs, info = result
        else:
            native_obs, info = result, None
        self.last_reward = None
        self.last_terminated = False
        self.last_truncated = False
        self.last_info = info
        self._pending_milestones = []
        self.frame_trace = []
        obs = self._render(native_obs, info, reward=None, done=False)
        self.prev_obs = obs
        self.last_obs = obs
        if self._record_frames:
            self._capture_frame()
        return obs

    def step(self, action: Action) -> Obs:
        adapted = self._action_adapter(action) if self._action_adapter else action
        result = self._native.step(adapted)
        if len(result) == 5:
            native_obs, reward, terminated, truncated, info = result
        elif len(result) == 4:
            native_obs, reward, terminated, info = result
            truncated = False
        else:
            raise RegactError(
                ErrorCategory.ENV_RUNTIME,
                f"unexpected step() arity from {type(self._native).__name__}: {len(result)}",
            )
        self.prev_obs = self.last_obs
        self.last_reward = reward
        self.last_terminated = bool(terminated)
        self.last_truncated = bool(truncated)
        self.last_info = info
        self.action_count += 1
        obs = self._render(native_obs, info, reward=reward, done=self.is_done)
        self.last_obs = obs
        if self._milestone_detector is not None:
            self._pending_milestones.extend(self._milestone_detector(self))
        obs.info["milestones"] = self.drain_milestones()
        if self._record_frames:
            self._capture_frame()
        return obs

    @property
    def is_done(self) -> bool:
        return self.last_terminated or self.last_truncated

    def drain_milestones(self) -> list[str]:
        out = self._pending_milestones[:]
        self._pending_milestones = []
        return out

    def close(self) -> None:
        close = getattr(self._native, "close", None)
        if callable(close):
            close()

    def _render(
        self,
        native_obs: object,
        info: dict[str, Any] | None,
        *,
        reward: float | None,
        done: bool,
    ) -> Obs:
        obs = self._renderer.render(native_obs, info)
        obs.reward = reward
        obs.is_done = done
        return obs

    def _capture_frame(self) -> None:
        render_fn = getattr(self._native, "render", None)
        if callable(render_fn):
            self.frame_trace.append(render_fn())
