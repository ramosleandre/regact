"""Rebuild a task's live environment from its experience store.

The store holds every reset and real action of a task, in order. Replaying them on a fresh
environment brings it back to the recorded position, and comparing each replayed observation with
the recorded one proves it: the game is deterministic, so any difference means the environment or
the store is not the one the task ran with.
"""

from __future__ import annotations

from typing import Any

from regact.env.session import EnvSession
from regact.protocols.cwm.store import ExperienceStore, canonical
from regact.protocols.cwm.validation import differences

_RESETS = {"reset_level": "level", "reset_environment": "environment"}


class RebuildMismatch(Exception):
    """A replayed observation differs from the recorded one: the task cannot be resumed."""

    def __init__(self, where: dict[str, Any], replayed: Any, recorded: Any) -> None:
        self.where, self.replayed, self.recorded = where, replayed, recorded
        self.differences = differences(replayed, recorded, 6)
        super().__init__(
            f"replayed observation differs from the recorded one at episode "
            f"{where['episode_id']}, step {where['step']} (observation {where['observation_id']})"
        )


def rebuild_environment(
    store: ExperienceStore, env: EnvSession, *, seed: int | None, single_instance: bool
) -> dict[str, Any] | None:
    """Replay the store's resets and actions on ``env``, which must not have been made yet.

    Every replayed observation must equal the recorded one, or :class:`RebuildMismatch` names the
    first that does not. Returns the last observation (the live one in single_instance), or None
    for an empty store.

    Episodes are replayed as the coordinator ran them: an explicit reset acts on the live
    instance; any other start (the initial collection, a multi_instance exploration) makes the
    environment through its lifecycle policy and resets it. Replaying everything, not only what
    follows the last environment start, also restores the wrapper's action counters.
    """
    if env.live is not None:
        raise ValueError("rebuild_environment needs an environment session that was never made")
    obs: dict[str, Any] | None = None
    for episode in store.episodes():
        kind = _RESETS.get(episode["purpose"])
        if kind is not None and not single_instance:
            raise ValueError(
                f"episode {episode['episode_id']}: an explicit reset in multi_instance"
            )
        if kind is None:
            live = env.live if single_instance and env.live is not None else env.make()
            obs = live.reset_explicit("environment", seed=seed).to_json()
        else:
            if env.live is None:
                raise ValueError(f"episode {episode['episode_id']}: a reset before any start")
            live = env.live
            obs = live.reset_explicit(kind, seed=seed).to_json()
        where = {
            "episode_id": episode["episode_id"],
            "step": 0,
            "observation_id": episode["initial_obs_id"],
        }
        _same(obs, store.observation(episode["initial_obs_id"]), where)
        for item in store.episode_steps(episode["episode_id"]):
            obs = live.step(item["action"]).to_json()
            where = {
                "episode_id": episode["episode_id"],
                "step": item["step_index"] + 1,
                "observation_id": item["after_obs_id"],
                "transition_id": item["transition_id"],
            }
            _same(obs, store.observation(item["after_obs_id"]), where)
    return obs


def _same(replayed: dict[str, Any], recorded: dict[str, Any], where: dict[str, Any]) -> None:
    if canonical(replayed) != canonical(recorded):
        raise RebuildMismatch(where, replayed, recorded)
