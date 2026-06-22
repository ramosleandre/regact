"""The sandboxed eval: SandboxedExecutor runs the controller in a real subprocess.

A real uvicorn env server (own thread) + a bootstrapped workdir (its baked
``framework.make_env``), then the executor spawns ``regact.controllers.eval_runner``
and scores its raw outcomes. The wrapper is identity here (no OS sandbox in CI); it
proves the subprocess plumbing — connect over HTTP, run, write raw, score on this side.
"""

from pathlib import Path

from regact.config.schema import Lifecycle
from regact.controllers.executor import SandboxedExecutor
from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.features.controller import ControllerFeature
from regact.orchestration.env_transport import serve_env
from regact.testing.fakes import FakeNativeEnv
from regact.workspace.bootstrap import Workspace

_FORWARD = """\
from code_library.base_controller import BaseController


class Controller(BaseController):
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


async def test_sandboxed_executor_scores_via_subprocess(tmp_path: Path) -> None:
    workdir = str(tmp_path / "wd")
    server = _server()
    async with serve_env(server, "g", in_process=False) as conn:
        Workspace(workdir).bootstrap(
            [ControllerFeature()],
            problem_name="p",
            task_name="g",
            env_base_url=conn.base_url,
            game_id="g",
            lifecycle=Lifecycle.MULTI_INSTANCE,
        )
        (Path(workdir) / "solution.py").write_text(_FORWARD)
        executor = SandboxedExecutor(workdir=workdir, sandbox_wrap=lambda argv: argv)
        result = executor.run(
            task_name="g",
            solution_path=str(Path(workdir) / "solution.py"),
            output_path=str(Path(workdir) / "submissions" / "0" / "results.json"),
            lifecycle=Lifecycle.MULTI_INSTANCE,
            n_episodes=1,
            max_moves=10,
        )
    # The eval ran out-of-process and was scored here.
    assert result.executor == "subprocess"
    assert result.error is None
    assert result.aggregate["success_rate"] == 1.0
    # results.json was persisted for the viewer.
    assert (Path(workdir) / "submissions" / "0" / "results.json").is_file()
