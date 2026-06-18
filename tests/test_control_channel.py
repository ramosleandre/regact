"""Integration: the framework-tool control channel for CLI agents.

A CLI agent (Claude/codex) can't receive native Python tools, so it invokes
SubmitSolution / ExitTask over HTTP. This drives that path end-to-end: a real
uvicorn server (on its own thread) with the ControllerFeature tools bound, hit
exactly like the workdir ``control.py`` would. Uses the real transport so the
sync executor (run via to_thread) can reach the same server without deadlock.
"""

import json
from pathlib import Path

import httpx
import pytest

from regact.config.schema import Lifecycle
from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.features.base import RunDeps
from regact.features.controller import ControllerFeature
from regact.orchestration.env_transport import serve_env
from regact.session.state import ExperimentState
from regact.testing.fakes import FakeNativeEnv

pytestmark = pytest.mark.integration

_FORWARD = """\
class Controller:
    def act(self, obs):
        return 1


def get_controller():
    return Controller()
"""


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


async def test_control_channel_runs_submit_and_exit(tmp_path: Path) -> None:
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "solution.py").write_text(_FORWARD)

    server = _server()
    async with serve_env(server, "g", in_process=False) as conn:
        experiment = ExperimentState(problem_name="p", task_name="g", n_eval_episodes=1, n_videos=0)
        deps = RunDeps(
            experiment=experiment,
            env_client=conn.client,
            lifecycle=Lifecycle.MULTI_INSTANCE,
            solution_path=str(workdir / "solution.py"),
            submissions_dir=str(workdir / "submissions"),
            n_episodes=1,
            max_moves=10,
        )
        tools = ControllerFeature().tools(deps)
        server.bind_control("g", tools, cwd=str(workdir))
        url = f"{conn.base_url}/control/g/tool"

        # The agent's `control.py SubmitSolution` would POST exactly this.
        submit = httpx.post(url, json={"name": "SubmitSolution", "input": {}}, timeout=30.0)
        assert submit.status_code == 200
        assert submit.json()["is_error"] is False
        assert experiment.submission_count == 1
        results = json.loads((workdir / "submissions" / "0" / "results.json").read_text())
        assert results["aggregate"]["success_rate"] == 1.0

        # `control.py ExitTask` flips the exit flag the loop watches.
        exit_resp = httpx.post(url, json={"name": "ExitTask", "input": {}}, timeout=30.0)
        assert exit_resp.status_code == 200
        assert experiment.exit_requested is True


async def test_control_channel_unbound_and_unknown_tool(tmp_path: Path) -> None:
    server = _server()
    async with serve_env(server, "g", in_process=False) as conn:
        url = f"{conn.base_url}/control/g/tool"
        # Not bound yet -> 503.
        assert httpx.post(url, json={"name": "SubmitSolution"}, timeout=30.0).status_code == 503

        server.bind_control("g", [], cwd=str(tmp_path))
        # Bound but the tool name is unknown -> 404.
        assert httpx.post(url, json={"name": "Nope"}, timeout=30.0).status_code == 404
