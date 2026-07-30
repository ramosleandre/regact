"""The delegating base for server-side env wrappers.

Features contribute wrapper factories (``Feature.env_wrapper``) that
``EnvSession._build`` applies around each ``WrappedEnv`` it creates, in
``features:`` list order. A wrapper sees the rendered, JSON-safe ``Obs`` —
exactly what the agent receives — and must preserve the ``WrappedEnv`` surface
the server and session read (``reset``/``step``/``close``, ``action_count``,
``last_obs``, …): this base delegates everything; subclasses override only what
they observe. A wrapper contains its own faults — it must never turn a working
``step`` into a broken one.
"""

from __future__ import annotations

from typing import Any, cast

from regact.envclient.obs import Action, Obs


class EnvWrapper:
    """Transparent delegate around a ``WrappedEnv`` (or another wrapper)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def reset(self, *, seed: int | None = None) -> Obs:
        return cast(Obs, self._inner.reset(seed=seed))

    def step(self, action: Action) -> Obs:
        return cast(Obs, self._inner.step(action))

    def close(self) -> None:
        self._inner.close()
