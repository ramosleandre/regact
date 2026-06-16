"""Owns the env handle for one game/task; applies the lifecycle policy.

Tools and the eval executor reach the env only through a session — never a module
global — so parallel games stay isolated. ``make_native`` is a factory for the
native env (Block 8 passes ``lambda: problem.make_env(task)``), keeping the
session decoupled from ``problems/``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from regact.env.lifecycle import EnvLifecyclePolicy
from regact.env.renderer import ObsRenderer
from regact.env.wrapped_env import WrappedEnv
from regact.envclient.obs import Action, Obs


class EnvSession:
    """Lifecycle-managed access to one game's env."""

    def __init__(
        self,
        *,
        make_native: Callable[[], Any],
        key: str,
        renderer: ObsRenderer,
        lifecycle: EnvLifecyclePolicy,
        milestone_detector: Callable[[WrappedEnv], list[str]] | None = None,
        action_adapter: Callable[[Action], Any] | None = None,
    ) -> None:
        self._make_native = make_native
        self.key = key
        self._renderer = renderer
        self._lifecycle = lifecycle
        self._milestone_detector = milestone_detector
        self._action_adapter = action_adapter
        self._live: WrappedEnv | None = None

    def _build(self) -> WrappedEnv:
        return WrappedEnv(
            self._make_native(),
            task_name=self.key,
            renderer=self._renderer,
            milestone_detector=self._milestone_detector,
            action_adapter=self._action_adapter,
        )

    def make(self) -> WrappedEnv:
        """Acquire the env via the policy (fresh for multi, cached for single)."""
        self._live = self._lifecycle.acquire(self._build, key=self.key)
        return self._live

    def reset(self, *, seed: int | None = None) -> Obs:
        """Reset the live env (full re-seed for multi, level-reset for single)."""
        env = self._live if self._live is not None else self.make()
        return env.reset(seed=seed)

    def assert_can_make(self) -> None:
        """Raise if making this game again would violate the one-make rule."""
        self._lifecycle.assert_can_make(self.key)

    @property
    def live(self) -> WrappedEnv | None:
        """The shared handle (``None`` before the first make)."""
        return self._live

    def close(self) -> None:
        if self._live is not None:
            self._live.close()
            self._live = None
