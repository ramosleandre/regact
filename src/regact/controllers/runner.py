"""Drive one controller rollout against an env.

``run_controller`` is the low-level loop: it calls ``controller.act(obs)`` and
applies the action via the env client, step after step, until the env is done or
``max_steps`` is reached. It returns a :class:`ControllerSummary`. It is deliberately
agnostic about lifecycle and aggregation — that lives in the ``ControllerExecutor``,
which calls this once per episode. The env is always reached through the HTTP
client, never imported.
"""

from __future__ import annotations

from typing import Any, Protocol

from regact.controllers.summary import (
    ControllerRun,
    ControllerSummary,
    MilestoneEvent,
)
from regact.envclient.client import EnvClient


class Controller(Protocol):
    """Anything with a pure ``act(obs) -> action`` (a foreign object: Protocol)."""

    def act(self, obs: Any) -> Any: ...


def run_controller(
    env: EnvClient,
    controller: Controller,
    *,
    name: str = "controller",
    max_steps: int = 400,
    collect_frames: bool = False,
) -> ControllerSummary:
    """Roll ``controller`` out on ``env`` (already reset) until done or ``max_steps``.

    With ``collect_frames`` it records each step's ``obs`` (JSON) for later video render.
    """
    history = ControllerRun(name=name)
    obs = env.current()
    frames = [obs.to_json()] if collect_frames else []
    actions: list[Any] = []
    steps = 0

    def done(kind: str, reason: str) -> ControllerSummary:
        return ControllerSummary(kind, reason, steps, history, obs, frames=frames, actions=actions)

    while True:
        if obs.is_done:
            return done("env_done", "environment signalled done")
        if steps >= max_steps:
            return done("max_steps", f"reached max_steps={max_steps}")

        action = controller.act(obs)
        actions.append(action)
        obs = env.step(action)
        steps += 1
        if collect_frames:
            frames.append(obs.to_json())

        for text in obs.info.get("milestones", []):
            history.events.append(MilestoneEvent(step=steps, description=str(text)))
