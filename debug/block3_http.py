"""Block 3 manual smoke: drive a FakeNativeEnv over HTTP (server <-> client).

Run:  python debug/block3_http.py   (uses FastAPI TestClient = in-process HTTP)
"""

from fastapi.testclient import TestClient

from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.envclient.client import EnvClient
from regact.testing.fakes import FakeNativeEnv


def main() -> None:
    server = EnvServer()
    server.register(
        "corridor",
        EnvSession(
            make_native=lambda: FakeNativeEnv(goal=3),
            key="corridor",
            renderer=RawRenderer(),
            lifecycle=MultiInstancePolicy(),
        ),
    )
    client = EnvClient(TestClient(server.app), "corridor")

    obs = client.reset()
    print("reset over HTTP:", obs.frame, "| actions:", obs.available_actions)
    while not client.is_done:
        obs = client.step(1)
        print(f"step -> grid={obs.frame['grid']} reward={obs.reward} done={obs.is_done}")
    print(f"server reports last-step = {client.last_step()} actions")


if __name__ == "__main__":
    main()
