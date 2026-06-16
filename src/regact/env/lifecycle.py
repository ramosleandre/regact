"""Env lifecycle policies: how an ``EnvSession`` creates the underlying env.

The one-make guard lives here (owned by the policy that holds the handle), not on
the problem — so parallel games never share mutable guard state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from regact.env.wrapped_env import WrappedEnv
from regact.obs.errors import ErrorCategory, RegactError


class EnvLifecyclePolicy(ABC):
    """Strategy for env creation."""

    @abstractmethod
    def acquire(self, build: Callable[[], WrappedEnv], *, key: str) -> WrappedEnv:
        """Build a new env, or return the existing handle (this is what launches it)."""
        ...

    @abstractmethod
    def assert_can_make(self, key: str) -> None:
        """Enforce the creation invariant (the one-make guard). Raise on violation."""
        ...


class MultiInstancePolicy(EnvLifecyclePolicy):
    """Research: a fresh ``WrappedEnv`` on every acquire."""

    def acquire(self, build: Callable[[], WrappedEnv], *, key: str) -> WrappedEnv:
        return build()

    def assert_can_make(self, key: str) -> None:
        return None


class SingleInstancePolicy(EnvLifecyclePolicy):
    """Competition: one env per game; subsequent acquires return the same handle."""

    def __init__(self) -> None:
        self._handles: dict[str, WrappedEnv] = {}

    def acquire(self, build: Callable[[], WrappedEnv], *, key: str) -> WrappedEnv:
        if key not in self._handles:
            self._handles[key] = build()
        return self._handles[key]

    def assert_can_make(self, key: str) -> None:
        if key in self._handles:
            raise RegactError(
                ErrorCategory.ENV_RUNTIME, f"game {key!r} already made (one-make rule)"
            )
