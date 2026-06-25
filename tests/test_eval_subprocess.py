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
    # results.json + a snapshot of the evaluated controller were persisted for the viewer.
    assert (Path(workdir) / "submissions" / "0" / "results.json").is_file()
    assert (Path(workdir) / "submissions" / "0" / "solution.py").is_file()


async def test_sandboxed_executor_records_video(tmp_path: Path) -> None:
    import numpy as np

    def render_frame(obs: object) -> object:  # the problem's job; fixed RGB here
        return np.full((16, 16, 3), 120, dtype=np.uint8)

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
        executor = SandboxedExecutor(
            workdir=workdir, sandbox_wrap=lambda argv: argv, render_frame=render_frame
        )
        executor.run(
            task_name="g",
            solution_path=str(Path(workdir) / "solution.py"),
            output_path=str(Path(workdir) / "submissions" / "0" / "results.json"),
            lifecycle=Lifecycle.MULTI_INSTANCE,
            n_episodes=1,
            max_moves=10,
            record_video=True,
        )
    video = Path(workdir) / "submissions" / "0" / "video_0.mp4"
    assert video.is_file() and video.stat().st_size > 0


async def test_sandboxed_executor_shadow_replay_scores_on_trusted_env(tmp_path: Path) -> None:
    """With shadow_replay, the score comes from re-applying the recorded actions on the trusted
    env (this side), not from the subprocess's self-reported obs."""
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
        executor = SandboxedExecutor(
            workdir=workdir,
            sandbox_wrap=lambda argv: argv,
            env_client=conn.client,
            shadow_replay=True,
        )
        result = executor.run(
            task_name="g",
            solution_path=str(Path(workdir) / "solution.py"),
            output_path=str(Path(workdir) / "submissions" / "0" / "results.json"),
            lifecycle=Lifecycle.MULTI_INSTANCE,
            n_episodes=1,
            max_moves=10,
        )
    assert result.executor == "shadow_replay"  # scored by the trusted replay, not the subprocess
    assert result.aggregate["success_rate"] == 1.0
