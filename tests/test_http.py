"""Tests for the HTTP env boundary (server <-> client), via FastAPI TestClient."""

import pytest
from fastapi.testclient import TestClient

try:  # starlette's TestClient prefers httpx2 when it is installed
    from httpx2 import HTTPStatusError
except ImportError:
    from httpx import HTTPStatusError

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


def test_env_client_connect_timeout_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env-call budget is generous by default and tunable via REGACT_ENV_CLIENT_TIMEOUT_S,
    so a slow first make_env() handshake under CPU contention does not hit a tight 30s budget."""
    default = EnvClient.connect("http://127.0.0.1:1", "g")
    assert default._http.timeout.read == 120.0  # generous default, not the old 30s
    default._http.close()
    monkeypatch.setenv("REGACT_ENV_CLIENT_TIMEOUT_S", "45")
    tuned = EnvClient.connect("http://127.0.0.1:1", "g")
    assert tuned._http.timeout.read == 45.0
    tuned._http.close()


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
    with pytest.raises(HTTPStatusError) as exc:
        client.reset()
    assert exc.value.response.status_code == 404


def test_step_before_reset_409() -> None:
    client = _client(_server(MultiInstancePolicy()))
    with pytest.raises(HTTPStatusError) as exc:
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


def test_env_fault_returns_422_not_a_terminal_500() -> None:
    """A native env step/reset that raises (a malformed action, a WFC gen failure) comes back as a
    clean 422 - so FastAPI does NOT surface an unhandled ASGI 500 (which uvicorn logs to the
    operator's terminal), and the agent's env client gets a usable error instead of a bare crash."""

    class _RaisingStep(FakeNativeEnv):
        def step(self, action: int) -> object:  # type: ignore[override]
            raise KeyError("x")  # e.g. arcengine on a malformed ACTION6

    server = EnvServer()
    server.register(
        "g",
        EnvSession(
            make_native=lambda: _RaisingStep(goal=3),
            key="g",
            renderer=RawRenderer(),
            lifecycle=MultiInstancePolicy(),
        ),
    )
    app = TestClient(server.app)
    assert app.post("/env/g/reset", json={}).status_code == 200  # reset ok -> env is live
    r = app.post("/env/g/step", json={"action": 6})
    assert r.status_code == 422 and "step failed" in r.json()["detail"]

    class _RaisingReset(FakeNativeEnv):
        def reset(self, *, seed: int | None = None) -> object:  # type: ignore[override]
            raise RuntimeError("Could not generate a valid pattern")  # a WFC gen failure

    s2 = EnvServer()
    s2.register(
        "g",
        EnvSession(
            make_native=lambda: _RaisingReset(goal=3),
            key="g",
            renderer=RawRenderer(),
            lifecycle=MultiInstancePolicy(),
        ),
    )
    r2 = TestClient(s2.app).post("/env/g/reset", json={})
    assert r2.status_code == 422 and "reset failed" in r2.json()["detail"]
