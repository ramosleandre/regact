"""Integration: the env-server transport (in-process + real uvicorn)."""

import pytest

from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.orchestration.env_transport import serve_env
from regact.testing.fakes import FakeNativeEnv

pytestmark = pytest.mark.integration


def _server() -> EnvServer:
    server = EnvServer()
    server.register(
        "g",
        EnvSession(
            make_native=lambda: FakeNativeEnv(goal=3),
            key="g",
            renderer=RawRenderer(),
            lifecycle=MultiInstancePolicy(),
        ),
    )
    return server


async def test_in_process_transport_drives_the_env() -> None:
    async with serve_env(_server(), "g", in_process=True) as conn:
        obs = conn.client.reset()
        assert obs.available_actions == [0, 1]
        conn.client.step(1)
        assert conn.client.action_count == 1


async def test_real_uvicorn_transport_binds_and_drives() -> None:
    async with serve_env(_server(), "g", in_process=False) as conn:
        assert conn.base_url.startswith("http://127.0.0.1:")
        assert not conn.base_url.endswith(":0")  # a real ephemeral port was bound
        conn.client.reset()
        for _ in range(3):
            conn.client.step(1)
        assert conn.client.is_done is True
        assert conn.client.last_reward == 1.0
