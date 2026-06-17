"""Drive one controller rollout against an env.

``run_controller`` is the low-level loop: it calls ``controller.act(obs)`` and
applies the action via the env client, step after step, until the env is done or
``max_steps`` is reached. It returns a :class:`RunSummary`. It is deliberately
agnostic about lifecycle and aggregation — that lives in the ``EvalExecutor``,
which calls this once per episode. The env is always reached through the HTTP
client, never imported.
"""

from __future__ import annotations

from typing import Any, Protocol

from regact.envclient.client import EnvClient
from regact.features.controllers.run_summary import (
    ControllerRun,
    MilestoneEvent,
    RunSummary,
)


class Controller(Protocol):
    """Anything with a pure ``act(obs) -> action`` (a foreign object: Protocol)."""

    def act(self, obs: Any) -> Any: ...


def run_controller(
    env: EnvClient,
    controller: Controller,
    *,
    name: str = "controller",
    max_steps: int = 400,
) -> RunSummary:
    """Roll ``controller`` out on ``env`` (already reset) until done or ``max_steps``."""
    history = ControllerRun(name=name)
    obs = env.current()
    steps = 0

    while True:
        if obs.is_done:
            return RunSummary("env_done", "environment signalled done", steps, history, obs)
        if steps >= max_steps:
            return RunSummary("max_steps", f"reached max_steps={max_steps}", steps, history, obs)

        action = controller.act(obs)
        obs = env.step(action)
        steps += 1

        for text in obs.info.get("milestones", []):
            history.events.append(MilestoneEvent(step=steps, description=str(text)))
