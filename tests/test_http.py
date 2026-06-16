"""Tests for the HTTP env boundary (server <-> client), via FastAPI TestClient."""

import httpx
import pytest
from fastapi.testclient import TestClient

from regact.env.lifecycle import EnvLifecyclePolicy, MultiInstancePolicy, SingleInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.envclient.client import EnvClient
from regact.testing.fakes import FakeNativeEnv


def _server(lifecycle: EnvLifecyclePolicy) -> EnvServer:
    server = EnvServer()
    server.register(
        "g",
        EnvSession(
            make_native=lambda: FakeNativeEnv(goal=3),
            key="g",
            renderer=RawRenderer(),
            lifecycle=lifecycle,
        ),
    )
    return server


def _client(server: EnvServer, game_id: str = "g") -> EnvClient:
    return EnvClient(TestClient(server.app), game_id)


def test_http_roundtrip() -> None:
    client = _client(_server(MultiInstancePolicy()))
    obs = client.reset()
    assert obs.frame == {"pos": 0, "grid": [1, 0, 0, 0]}
    assert obs.available_actions == [0, 1]
    assert client.action_count == 0
    obs = client.step(1)
    assert obs.frame["pos"] == 1
    assert client.action_count == 1
    assert client.is_done is False


def test_http_episode_to_goal() -> None:
    client = _client(_server(MultiInstancePolicy()))
    client.reset()
    for _ in range(3):
        client.step(1)
    assert client.is_done is True
    assert client.last_reward == 1.0
    assert client.last_step() == 3


def test_unknown_game_404() -> None:
    client = _client(_server(MultiInstancePolicy()), game_id="nope")
    with pytest.raises(httpx.HTTPStatusError) as exc:
        client.reset()
    assert exc.value.response.status_code == 404


def test_step_before_reset_409() -> None:
    client = _client(_server(MultiInstancePolicy()))
    with pytest.raises(httpx.HTTPStatusError) as exc:
        client.step(1)
    assert exc.value.response.status_code == 409


def test_single_instance_persists_across_reset() -> None:
    # one-make server-side: same handle => action_count survives a (level) reset
    client = _client(_server(SingleInstancePolicy()))
    client.reset()
    client.step(1)
    client.step(1)
    assert client.last_step() == 2
    client.reset()  # level reset on the SAME handle
    assert client.last_step() == 2


def test_multi_instance_fresh_on_reset() -> None:
    client = _client(_server(MultiInstancePolicy()))
    client.reset()
    client.step(1)
    client.step(1)
    assert client.last_step() == 2
    client.reset()  # fresh env
    assert client.last_step() == 0
