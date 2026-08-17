"""Tests for the always-on Controller: templates, prompt, tools wiring, teardown hook.

RunDeps carries the agnostic EnvClient (not a controller executor); the controller
builds its own ControllerExecutor from it. So these wire a real client over a
TestClient + FakeNativeEnv - no LLM, no real game.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from regact.config.schema import ControllerConfig, Lifecycle
from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.envclient.client import EnvClient
from regact.features.base import FeatureContext, HookPhase, RunDeps, build_features
from regact.features.controller import Controller, FinalizeControllerHook
from regact.session.state import ExperimentState
from regact.testing.fakes import FakeNativeEnv
from regact.tools.exit_task import ExitTask
from regact.tools.submit_solution import SubmitSolution
from regact.workspace.bootstrap import Workspace

# A controller that always steps forward reaches the corridor goal in 3 moves.
_FORWARD = """\
class Controller:
    def act(self, obs):
        return 1


def get_controller():
    return Controller()
"""


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


def _ctx() -> FeatureContext:
    return FeatureContext(problem_name="grid", task_name="lvl1", workdir="/tmp/wd")


def _deps(tmp_path: Path) -> RunDeps:
    return RunDeps(
        experiment=ExperimentState(
            problem_name="grid", task_name="g", n_eval_episodes=2, n_videos=0
        ),
        env_client=_client(),
        lifecycle=Lifecycle.MULTI_INSTANCE,
        solution_path=str(tmp_path / "solution.py"),
        submissions_dir=str(tmp_path / "submissions"),
    )


def test_controller_templates_lay_out_files() -> None:
    relpaths = {t.relpath for t in Controller().templates(_ctx())}
    assert relpaths == {
        "code_library/base_controller.py",
        "code_library/example_controller.py",
        "code_library/interactive_script_example.py",
        "solution.py",
    }


def test_interactive_script_template_wires_env_and_controller() -> None:
    templates = {t.relpath: t.content for t in Controller().templates(_ctx())}
    script = templates["code_library/interactive_script_example.py"]
    assert "from framework.make_env import make_env" in script
    assert "from code_library.example_controller import ExampleController" in script
    assert "obs.available_actions" in script  # prints small facts, not the full frame


def test_controller_prompt_fragment_explains_contract() -> None:
    fragment = Controller().prompt_fragment(_ctx())
    assert fragment is not None
    assert "act(self, obs)" in fragment  # the controller contract
    assert "interactive_script_example.py" in fragment  # points at the runnable template
    assert "SubmitSolution" in fragment  # how to submit (ExitTask lives in the control block)
    # The stub code lives in the workdir file, not pasted into the prompt.
    assert "def get_controller" not in fragment


def test_controller_tools_wired_with_run_deps(tmp_path: Path) -> None:
    tools = Controller(n_episodes=2, max_moves=100).tools(_deps(tmp_path))
    assert isinstance(tools[0], SubmitSolution)
    assert isinstance(tools[1], ExitTask)
    assert {t.name for t in tools} == {"SubmitSolution", "ExitTask"}


def test_controller_hook_is_teardown_finalize(tmp_path: Path) -> None:
    hooks = Controller(n_episodes=2, max_moves=100).hooks(_deps(tmp_path))
    assert len(hooks) == 1 and isinstance(hooks[0], FinalizeControllerHook)
    assert hooks[0].phase is HookPhase.TEARDOWN


async def test_finalize_hook_rescores_existing_solution(tmp_path: Path) -> None:
    (tmp_path / "solution.py").write_text(_FORWARD)
    deps = _deps(tmp_path)
    result = await Controller(n_episodes=2, max_moves=100).hooks(deps)[0].run()
    assert result is not None
    # It scored the final solution and wrote the official "final" result.
    final = json.loads((tmp_path / "submissions" / "final" / "results.json").read_text())
    assert final["aggregate"]["success_rate"] == 1.0
    assert deps.experiment.last_submission_results is not None


async def test_finalize_hook_skips_when_no_solution(tmp_path: Path) -> None:
    deps = _deps(tmp_path)  # no solution.py on disk
    result = await Controller(n_episodes=2, max_moves=100).hooks(deps)[0].run()
    assert result is None
    assert not (tmp_path / "submissions" / "final").exists()


async def test_default_solution_scores_without_editing(tmp_path: Path) -> None:
    """A freshly bootstrapped, unedited solution.py runs and scores: the default subclasses
    ExampleController, so a no-op submission no longer raises NotImplementedError."""
    templates = {t.relpath: t.content for t in Controller().templates(_ctx())}
    stub = templates["solution.py"]
    assert "from code_library.example_controller import ExampleController" in stub
    assert "raise NotImplementedError" not in stub

    ws = Workspace(str(tmp_path / "wd"))
    ws.bootstrap(
        [],
        controller=Controller(),
        problem_name="grid",
        task_name="lvl1",
        env_base_url="http://127.0.0.1:9000",
        game_id="grid-lvl1",
        lifecycle=Lifecycle.MULTI_INSTANCE,
    )
    deps = RunDeps(
        experiment=ExperimentState(
            problem_name="grid", task_name="g", n_eval_episodes=1, n_videos=0
        ),
        env_client=_client(),
        lifecycle=Lifecycle.MULTI_INSTANCE,
        solution_path=str(Path(ws.root) / "solution.py"),
        submissions_dir=str(tmp_path / "submissions"),
    )
    result = await Controller(n_episodes=1, max_moves=50).hooks(deps)[0].run()
    assert result is not None
    assert result.error is None  # the untouched stub ran end-to-end


def test_from_config_maps_knobs() -> None:
    controller = Controller.from_config(ControllerConfig(n_episodes=3, shadow_replay=True))
    assert controller._n_episodes == 3  # config knobs reach the controller
    assert controller._shadow_replay is True


def test_controller_is_not_a_registered_feature() -> None:
    """The controller is core, built from config.controller - it must NOT be resolvable as
    a feature (a stray `features=controller` should fail loudly, not silently duplicate it)."""
    with pytest.raises(ValueError, match="unknown feature 'controller'"):
        build_features({"controller": {}})


def test_bootstrap_with_controller_writes_solution(tmp_path: Path) -> None:
    """The agnostic base + the always-on controller together produce a full workdir."""
    ws = Workspace(str(tmp_path / "wd"))
    ws.bootstrap(
        [],
        controller=Controller(),
        problem_name="grid",
        task_name="lvl1",
        env_base_url="http://127.0.0.1:9000",
        game_id="grid-lvl1",
        lifecycle=Lifecycle.MULTI_INSTANCE,
    )
    root = Path(ws.root)
    assert (root / "solution.py").exists()
    assert (root / "code_library" / "base_controller.py").exists()
    assert (root / "code_library" / "example_controller.py").exists()
    assert (root / "framework" / "make_env.py").exists()  # base still there
