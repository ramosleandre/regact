"""Tests for Block 5: experiment state, submit/exit tools, workspace, prompt builder."""

from pathlib import Path

from regact.config.schema import Lifecycle
from regact.features.base import Feature, FeatureContext, Hook, RunDeps, TemplateFile
from regact.obs.result import EvalResult
from regact.prompt.builder import PromptBuilder
from regact.session.state import ExperimentState
from regact.tools.base import Tool, ToolContext
from regact.tools.exit_task import ExitTask
from regact.tools.submit_solution import SubmitSolution
from regact.workspace.bootstrap import Workspace


def _state() -> ExperimentState:
    return ExperimentState(problem_name="p", task_name="t", n_eval_episodes=1, n_videos=0)


def test_experiment_state_roundtrip(tmp_path: Path) -> None:
    state = _state()
    state.submission_count = 2
    path = str(tmp_path / "experiment_state.json")
    state.save(path)
    loaded = ExperimentState.load(path)
    assert loaded.submission_count == 2
    assert loaded.problem_name == "p"


async def test_exit_task_sets_flag() -> None:
    state = _state()
    tool = ExitTask(state)
    assert state.exit_requested is False
    out = await tool.call({}, ToolContext(cwd="/tmp"))
    assert state.exit_requested is True
    assert out.is_error is False


class _FakeExecutor:
    """Stand-in for the Block 6 ControllerExecutor: records the call, returns a fixed result."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs: object) -> EvalResult:
        self.calls.append(kwargs)
        return EvalResult(task="t", aggregate={"success_rate": 1.0})


async def test_submit_solution_runs_executor_and_records(tmp_path: Path) -> None:
    state = _state()
    executor = _FakeExecutor()
    tool = SubmitSolution(
        state,
        executor,  # type: ignore[arg-type]
        solution_path=str(tmp_path / "solution.py"),
        submissions_dir=str(tmp_path / "submissions"),
        task_name="t",
        lifecycle=Lifecycle.MULTI_INSTANCE,
    )
    out = await tool.call({}, ToolContext(cwd=str(tmp_path)))

    assert len(executor.calls) == 1
    assert state.submission_count == 1
    assert state.last_submission_results == {
        "task": "t",
        "aggregate": {"success_rate": 1.0},
        "episodes": [],
        "error": None,
        "error_category": None,
        "executor": None,
    }
    assert out.data == {"submission": 0, "aggregate": {"success_rate": 1.0}}
    # The executor wrote into submissions/0/results.json
    assert (tmp_path / "submissions" / "0").is_dir()


def test_workspace_bootstrap_multi(tmp_path: Path) -> None:
    ws = Workspace(str(tmp_path / "wd"))
    ws.bootstrap(
        [],
        problem_name="grid",
        task_name="lvl1",
        env_base_url="http://127.0.0.1:9000",
        game_id="grid-lvl1",
        lifecycle=Lifecycle.MULTI_INSTANCE,
    )
    root = Path(ws.root)
    # The agnostic base is only the tree + make_env.py; solution.py is the
    # ControllerFeature's template, not part of the base.
    assert not (root / "solution.py").exists()
    assert (root / "code_library").is_dir()
    assert (root / "knowledge_base").is_dir()
    make_env = (root / "framework" / "make_env.py").read_text()
    assert "http://127.0.0.1:9000" in make_env
    assert "grid-lvl1" in make_env
    assert "_HANDLE" not in make_env  # multi-instance => fresh per call
    assert ws.solution_path().endswith("solution.py")


def test_workspace_bootstrap_single_uses_shared_handle(tmp_path: Path) -> None:
    ws = Workspace(str(tmp_path / "wd"))
    ws.bootstrap(
        [],
        problem_name="arc",
        task_name="ls20",
        env_base_url="http://127.0.0.1:9000",
        game_id="arc-ls20",
        lifecycle=Lifecycle.SINGLE_INSTANCE,
    )
    make_env = (Path(ws.root) / "framework" / "make_env.py").read_text()
    assert "_HANDLE" in make_env  # single-instance => one shared handle


class _StubFeature(Feature):
    name = "stub"

    def templates(self, ctx: FeatureContext) -> list[TemplateFile]:
        return [TemplateFile("code_library/note.py", "# stub feature\n")]

    def prompt_fragment(self, ctx: FeatureContext) -> str | None:
        return "## Stub feature\nUse the stub."

    def tools(self, deps: RunDeps) -> list[Tool]:
        return []

    def hooks(self, deps: RunDeps) -> list[Hook]:
        return []


def test_workspace_writes_feature_templates(tmp_path: Path) -> None:
    ws = Workspace(str(tmp_path / "wd"))
    ws.bootstrap(
        [_StubFeature()],
        problem_name="grid",
        task_name="lvl1",
        env_base_url="http://x",
        game_id="g",
        lifecycle=Lifecycle.MULTI_INSTANCE,
    )
    assert (Path(ws.root) / "code_library" / "note.py").read_text() == "# stub feature\n"


class _StubProblem:
    name = "grid"

    def build_prompt(self, task_name: str, *, info_mode: object) -> str:
        return f"# Game: grid\n\nYou are playing grid task {task_name}."


def test_prompt_builder_system_carries_everything_static() -> None:
    """The system prompt holds role + game + feature + control + lifecycle (all static)."""
    builder = PromptBuilder()
    system = builder.build_system_prompt(
        _StubProblem(),  # type: ignore[arg-type]
        "lvl1",
        [_StubFeature()],
        lifecycle=Lifecycle.MULTI_INSTANCE,
        control_actions="client_cli",
        tool_names=["SubmitSolution", "ExitTask"],
    )
    assert "make_env" in system  # role
    assert "grid" in system and "lvl1" in system  # game section
    assert "Stub feature" in system  # feature fragment layered in
    assert "fresh" in system.lower()  # multi-instance lifecycle block (enum-keyed)
    assert "framework/control.py SubmitSolution" in system  # client_cli control block


def test_prompt_builder_drops_empty_feature_fragments() -> None:
    class _Silent(_StubFeature):
        def prompt_fragment(self, ctx: FeatureContext) -> str | None:
            return None

    system = PromptBuilder().build_system_prompt(
        _StubProblem(),  # type: ignore[arg-type]
        "lvl1",
        [_Silent()],
        lifecycle=Lifecycle.MULTI_INSTANCE,
    )
    assert "Stub feature" not in system
    assert "grid" in system


def test_first_message_is_the_observation_or_a_start() -> None:
    builder = PromptBuilder()
    # generic start is lifecycle-agnostic (no make_env: single-instance uses env.current)
    assert "submit" in builder.build_first_message().lower()
    framed = builder.build_first_message("OBS_GRID_HERE")
    assert "OBS_GRID_HERE" in framed and "first observation" in framed.lower()
