"""Block 6 (eval) manual smoke: evaluate a controller end-to-end, no LLM.

Writes a trivial forward-stepping controller, drives it over HTTP against
FakeNativeEnv via the ControllerExecutor, and prints the scored EvalResult.

Run:  python debug/block6_eval.py
"""

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from regact.config.schema import Lifecycle
from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.envclient.client import EnvClient
from regact.controllers.executor import ControllerExecutor
from regact.testing.fakes import FakeNativeEnv

_SOLUTION = """\
class Controller:
    def act(self, obs):
        return 1  # always step forward

def get_controller():
    return Controller()
"""


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

    tmp = Path(tempfile.mkdtemp(prefix="regact-eval-"))
    (tmp / "solution.py").write_text(_SOLUTION)
    out = tmp / "results.json"

    result = ControllerExecutor(client).run(
        task_name="corridor",
        solution_path=str(tmp / "solution.py"),
        output_path=str(out),
        lifecycle=Lifecycle.MULTI_INSTANCE,
        n_episodes=3,
        max_moves=10,
    )
    print("EvalResult.aggregate:", result.aggregate)
    print("per-episode:")
    for ep in result.episodes:
        print(f"  episode {ep.episode}: {ep.stop_kind} | metrics={ep.metrics}")
    print("results.json written at:", out)


if __name__ == "__main__":
    main()
