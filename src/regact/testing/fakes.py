"""Deterministic test doubles (no LLM, no real game).

``FakeNativeEnv`` is a tiny gym-like env used to exercise the env layer in CI
and in the ``debug/`` smoke scripts.
"""

from __future__ import annotations

from typing import Any


class FakeNativeEnv:
    """A 1-D corridor: action ``1`` moves right; reaching ``goal`` terminates (reward 1.0)."""

    def __init__(self, *, goal: int = 3) -> None:
        self.goal = goal
        self.pos = 0

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.pos = 0
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if action == 1:
            self.pos = min(self.pos + 1, self.goal)
        terminated = self.pos >= self.goal
        reward = 1.0 if terminated else 0.0
        return self._obs(), reward, terminated, False, self._info()

    def render(self) -> list[int]:
        grid = [0] * (self.goal + 1)
        grid[self.pos] = 1
        return grid

    def _obs(self) -> dict[str, Any]:
        return {"pos": self.pos, "grid": self.render()}

    def _info(self) -> dict[str, Any]:
        return {"available_actions": [0, 1]}
