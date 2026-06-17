"""Tests for run_controller + EvalExecutor: drive a controller over HTTP, score it.

No LLM: a trivial controller is written to a temp solution.py and evaluated against
FakeNativeEnv behind the in-process TestClient.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from regact.config.schema import Lifecycle
from regact.controllers.executor import EvalExecutor
from regact.controllers.runner import run_controller
from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.envclient.client import EnvClient
from regact.obs.errors import ErrorCategory
from regact.testing.fakes import FakeNativeEnv

# A controller that always steps forward (action 1) reaches the goal in 3 moves.
_FORWARD_SOLUTION = """\
class Controller:
    def act(self, obs):
        return 1

def get_controller():
    return Controller()
"""

# A controller that raises inside act().
_BROKEN_SOLUTION = """\
class Controller:
    def act(self, obs):
        raise RuntimeError("boom")

def get_controller():
    return Controller()
"""

# A solution file with no get_controller (import-time contract violation).
_NO_FACTORY_SOLUTION = "x = 1\n"


def _client() -> EnvClient:
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
    return EnvClient(TestClient(server.app), "g")


def test_run_controller_reaches_goal() -> None:
    client = _client()
    client.reset()

    class Forward:
        def act(self, obs: object) -> int:
            return 1

    summary = run_controller(client, Forward(), max_steps=10)
    assert summary.stop_kind == "env_done"
    assert summary.total_steps == 3


def test_run_controller_hits_max_steps() -> None:
    client = _client()
    client.reset()

    class Stay:
        def act(self, obs: object) -> int:
            return 0  # never advances

    summary = run_controller(client, Stay(), max_steps=5)
    assert summary.stop_kind == "max_steps"
    assert summary.total_steps == 5


def _write_solution(tmp_path: Path, body: str) -> str:
    path = tmp_path / "solution.py"
    path.write_text(body)
    return str(path)


def test_executor_scores_a_solution(tmp_path: Path) -> None:
    executor = EvalExecutor(_client())
    out = str(tmp_path / "results.json")
    result = executor.run(
        task_name="corridor",
        solution_path=_write_solution(tmp_path, _FORWARD_SOLUTION),
        output_path=out,
        lifecycle=Lifecycle.MULTI_INSTANCE,
        n_episodes=2,
        max_moves=10,
    )
    assert result.aggregate["n_episodes"] == 2
    assert result.aggregate["success_rate"] == 1.0
    assert result.aggregate["mean_steps"] == 3.0
    # results.json written and matches.
    assert json.loads(Path(out).read_text())["aggregate"]["success_rate"] == 1.0


def test_executor_single_instance_runs_one_episode(tmp_path: Path) -> None:
    executor = EvalExecutor(_client())
    result = executor.run(
        task_name="corridor",
        solution_path=_write_solution(tmp_path, _FORWARD_SOLUTION),
        output_path=str(tmp_path / "results.json"),
        lifecycle=Lifecycle.SINGLE_INSTANCE,
        n_episodes=5,  # ignored under single-instance
        max_moves=10,
    )
    assert result.aggregate["n_episodes"] == 1


def test_executor_catches_controller_exception(tmp_path: Path) -> None:
    executor = EvalExecutor(_client())
    result = executor.run(
        task_name="corridor",
        solution_path=_write_solution(tmp_path, _BROKEN_SOLUTION),
        output_path=str(tmp_path / "results.json"),
        lifecycle=Lifecycle.MULTI_INSTANCE,
        n_episodes=1,
        max_moves=10,
    )
    assert result.aggregate["n_errors"] == 1
    assert result.episodes[0].error_category is ErrorCategory.AGENT_SOLUTION


def test_executor_flags_missing_factory(tmp_path: Path) -> None:
    executor = EvalExecutor(_client())
    result = executor.run(
        task_name="corridor",
        solution_path=_write_solution(tmp_path, _NO_FACTORY_SOLUTION),
        output_path=str(tmp_path / "results.json"),
        lifecycle=Lifecycle.MULTI_INSTANCE,
        n_episodes=1,
        max_moves=10,
    )
    assert result.error_category is ErrorCategory.AGENT_SOLUTION
    assert result.episodes == []
