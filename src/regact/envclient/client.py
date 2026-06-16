"""The HTTP env client: mirrors the WrappedEnv surface over HTTP.

This is what ``make_env()`` returns in the agent's workdir, and what the eval
executor drives. It can also be used as a CLI (``python -m regact.envclient``)
for human/script poking. Inspired by arc-3-agents-baseline1 ``client.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import httpx

from regact.envclient.obs import Action, Obs


class EnvClient:
    """A thin HTTP client exposing a gym-like interface."""

    def __init__(self, http: httpx.Client, game_id: str) -> None:
        self._http = http
        self._game_id = game_id
        self.action_count = 0
        self.last_obs: Obs | None = None

    @classmethod
    def connect(cls, base_url: str, game_id: str) -> EnvClient:
        return cls(httpx.Client(base_url=base_url, timeout=30.0), game_id)

    def reset(self, *, seed: int | None = None) -> Obs:
        return self._apply(self._post("reset", {"seed": seed}))

    def step(self, action: Action) -> Obs:
        return self._apply(self._post("step", {"action": action}))

    def current(self) -> Obs:
        resp = self._http.get(f"/env/{self._game_id}/current")
        resp.raise_for_status()
        return self._apply(resp.json())

    def last_step(self) -> int:
        resp = self._http.get(f"/env/{self._game_id}/last-step")
        resp.raise_for_status()
        return int(resp.json()["action_count"])

    def stop(self) -> None:
        self._http.post(f"/env/{self._game_id}/stop").raise_for_status()

    def close(self) -> None:
        self._http.close()

    @property
    def is_done(self) -> bool:
        return self.last_obs.is_done if self.last_obs is not None else False

    @property
    def last_reward(self) -> float | None:
        return self.last_obs.reward if self.last_obs is not None else None

    def _post(self, route: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = self._http.post(f"/env/{self._game_id}/{route}", json=body)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data

    def _apply(self, envelope: dict[str, Any]) -> Obs:
        obs = Obs.from_json(envelope["obs"])
        self.last_obs = obs
        self.action_count = int(envelope["action_count"])
        return obs


def make_env(*, record_frames: bool = False) -> EnvClient:
    """Workdir factory: connect to the env server named by the run's environment.

    Reads ``REGACT_ENV_URL`` and ``REGACT_GAME_ID`` (set by the workspace
    bootstrap). Returns an :class:`EnvClient`, never the native env.
    """
    base_url = os.environ.get("REGACT_ENV_URL", "http://127.0.0.1:8000")
    game_id = os.environ.get("REGACT_GAME_ID", "")
    if not game_id:
        raise RuntimeError("REGACT_GAME_ID is not set (the workspace bootstrap sets it).")
    return EnvClient.connect(base_url, game_id)


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m regact.envclient --game <id> reset|step <action>|current``."""
    parser = argparse.ArgumentParser(prog="regact.envclient")
    parser.add_argument("--url", default=os.environ.get("REGACT_ENV_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--game", default=os.environ.get("REGACT_GAME_ID", ""))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("reset")
    sub.add_parser("current")
    step_parser = sub.add_parser("step")
    step_parser.add_argument("action", type=int)
    args = parser.parse_args(argv)

    client = EnvClient.connect(args.url, args.game)
    if args.cmd == "reset":
        obs = client.reset()
    elif args.cmd == "step":
        obs = client.step(args.action)
    else:
        obs = client.current()
    print(obs.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
