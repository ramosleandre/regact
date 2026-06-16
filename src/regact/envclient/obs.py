"""The observation DTO that crosses the wire.

The server serializes an ``Obs`` to JSON; the client deserializes it back. It is
a plain data container: no methods touch the game, no game logic, no source. The
agent learns each game's schema by exploring; the container is generic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# An action is whatever the target env's step accepts; kept opaque on the wire.
# Examples: int (MiniGrid), {"action": "ACTION6", "data": {"x": 3, "y": 4}} (ARC).
Action = Any

# A frame is the rendered grid/pixels the renderer chose to expose (or None).
Frame = Any


@dataclass
class Obs:
    """A single observation as seen by the agent/controller."""

    frame: Frame
    reward: float | None = None
    is_done: bool = False
    available_actions: list[Action] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Obs:
        """Rebuild an ``Obs`` from a server payload (dumb deserialization)."""
        return cls(
            frame=payload.get("frame"),
            reward=payload.get("reward"),
            is_done=bool(payload.get("is_done", False)),
            available_actions=list(payload.get("available_actions") or []),
            info=dict(payload.get("info") or {}),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "reward": self.reward,
            "is_done": self.is_done,
            "available_actions": self.available_actions,
            "info": self.info,
        }
