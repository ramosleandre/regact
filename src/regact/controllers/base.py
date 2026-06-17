"""The unit the agent writes.

A controller is a pure policy: observation in, action out. Instantiation is the
per-episode reset; there is no ``reset()``. It never receives the env — the
orchestrator drives ``act`` and applies the returned action. This file is also
copied into the agent's workdir as ``code_library/base_controller.py``.
"""

from __future__ import annotations

from typing import Any


class BaseController:
    """Base class for all controllers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Set up for a fresh episode. Keep any cross-step state on ``self``."""

    def act(self, obs: Any) -> Any:
        """Return an action that the env accepts for the given observation."""
        raise NotImplementedError
