"""Tests for ControllerFeature: templates, prompt, tools wiring, teardown hook, registry."""

from pathlib import Path

from regact.config.schema import Lifecycle
from regact.features.base import FeatureContext, HookPhase, RunDeps, build_features
from regact.features.controller import ControllerFeature, FinalizeControllerHook
from regact.obs.result import EvalResult
from regact.session.state import ExperimentState
from regact.tools.exit_task import ExitTask
from regact.tools.submit_solution import SubmitSolution
from regact.workspace.bootstrap import Workspace


class _FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs: object) -> EvalResult:
        self.calls.append(kwargs)
        return EvalResult(task="t", aggregate={"success_rate": 1.0})


def _ctx() -> FeatureContext:
    return FeatureContext(problem_name="grid", task_name="lvl1", workdir="/tmp/wd")


def _deps(tmp_path: Path) -> RunDeps:
    return RunDeps(
        experiment=ExperimentState(
            problem_name="grid", task_name="lvl1", n_eval_episodes=2, n_videos=0
        ),
        executor=_FakeExecutor(),  # type: ignore[arg-type]
        lifecycle=Lifecycle.MULTI_INSTANCE,
        solution_path=str(tmp_path / "solution.py"),
        submissions_dir=str(tmp_path / "submissions"),
        n_episodes=2,
        max_moves=100,
    )


def test_controller_templates_lay_out_three_files() -> None:
    relpaths = {t.relpath for t in ControllerFeature().templates(_ctx())}
    assert relpaths == {
        "code_library/base_controller.py",
        "code_library/example_controller.py",
        "solution.py",
    }


def test_controller_prompt_fragment_explains_contract() -> None:
    fragment = ControllerFeature().prompt_fragment(_ctx())
    assert fragment is not None
    assert "act(obs)" in fragment
    assert "SubmitSolution" in fragment
    assert "ExitTask" in fragment


def test_controller_tools_wired_with_run_deps(tmp_path: Path) -> None:
    tools = ControllerFeature().tools(_deps(tmp_path))
    assert isinstance(tools[0], SubmitSolution)
    assert isinstance(tools[1], ExitTask)
    assert {t.name for t in tools} == {"SubmitSolution", "ExitTask"}


def test_controller_hook_is_teardown_finalize(tmp_path: Path) -> None:
    hooks = ControllerFeature().hooks(_deps(tmp_path))
    assert len(hooks) == 1 and isinstance(hooks[0], FinalizeControllerHook)
    assert hooks[0].phase is HookPhase.TEARDOWN


async def test_finalize_hook_rescores_existing_solution(tmp_path: Path) -> None:
    (tmp_path / "solution.py").write_text("def get_controller():\n    return None\n")
    deps = _deps(tmp_path)
    result = await ControllerFeature().hooks(deps)[0].run()
    assert result is not None
    # It drove the executor and wrote the official "final" result.
    assert deps.executor.calls[0]["output_path"].endswith("submissions/final/results.json")  # type: ignore[union-attr]
    assert deps.experiment.last_submission_results is not None


async def test_finalize_hook_skips_when_no_solution(tmp_path: Path) -> None:
    deps = _deps(tmp_path)  # no solution.py on disk
    result = await ControllerFeature().hooks(deps)[0].run()
    assert result is None
    assert deps.executor.calls == []


def test_build_features_resolves_controller() -> None:
    features = build_features(["controller"])
    assert len(features) == 1 and isinstance(features[0], ControllerFeature)


def test_bootstrap_with_controller_feature_writes_solution(tmp_path: Path) -> None:
    """The agnostic base + ControllerFeature together produce a full workdir."""
    ws = Workspace(str(tmp_path / "wd"))
    ws.bootstrap(
        build_features(["controller"]),
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
