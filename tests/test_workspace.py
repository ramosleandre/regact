"""Tests for Block 5: experiment state, submit/exit tools, workspace, prompt builder."""

import re
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
        "aggregate_unverified": None,
        "episodes": [],
        "error": None,
        "error_category": None,
        "executor": None,
        "features": {},
    }
    assert out.data == {"submission": 0, "aggregate": {"success_rate": 1.0}}
    # The executor wrote into submissions/0/results.json
    assert (tmp_path / "submissions" / "0").is_dir()


async def test_submit_solution_records_feature_metrics(tmp_path: Path) -> None:
    """A feature's own numbers land in the submission under its name, so several
    features can contribute without knowing about each other."""
    import json

    state = ExperimentState(problem_name="p", task_name="t")
    executor = _FakeExecutor()
    tool = SubmitSolution(
        state,
        executor,  # type: ignore[arg-type]
        solution_path=str(tmp_path / "solution.py"),
        submissions_dir=str(tmp_path / "submissions"),
        task_name="t",
        lifecycle=Lifecycle.MULTI_INSTANCE,
        feature_metrics=lambda: {"cwm": {"coherence": 0.9}},
    )
    await tool.call({}, ToolContext(cwd=str(tmp_path)))

    assert state.last_submission_results is not None
    assert state.last_submission_results["features"] == {"cwm": {"coherence": 0.9}}
    written = json.loads((tmp_path / "submissions" / "0" / "results.json").read_text())
    assert written["features"] == {"cwm": {"coherence": 0.9}}


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
    # The agnostic base is only the tree + make_env.py; solution.py is the always-on
    # controller's template (no controller passed here), not part of the base.
    assert not (root / "solution.py").exists()
    assert (root / "code_library").is_dir()
    assert (root / "code_library" / "__init__.py").exists()  # importable package
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
    """The system prompt holds role + game + feature + control (multi adds no lifecycle block)."""
    builder = PromptBuilder()
    system = builder.build_system_prompt(
        _StubProblem(),  # type: ignore[arg-type]
        "lvl1",
        [_StubFeature()],
        lifecycle=Lifecycle.MULTI_INSTANCE,
        tool_protocol="client_cli",
        tool_names=["SubmitSolution", "ExitTask"],
    )
    assert "make_env" in system  # role
    assert "grid" in system and "lvl1" in system  # game section
    assert "Stub feature" in system  # feature fragment layered in
    assert "framework/control.py SubmitSolution" in system  # client_cli control block


def test_prompt_builder_bash_block_merges_shell_examples_into_control_block() -> None:
    """A bash_block agent gets the fenced-block terminal fragment (shell idioms + submit/exit
    folded in); a client_cli agent gets the plain Framework tools block instead."""
    builder = PromptBuilder()

    def build(tool_protocol: str) -> str:
        return builder.build_system_prompt(
            _StubProblem(),  # type: ignore[arg-type]
            "lvl1",
            [_StubFeature()],
            lifecycle=Lifecycle.MULTI_INSTANCE,
            tool_protocol=tool_protocol,  # type: ignore[arg-type]
            tool_names=["SubmitSolution", "ExitTask"],
        )

    bash = build("bash_block")
    assert "Working in the terminal" in bash and "cat > code_library" in bash
    assert "framework/control.py SubmitSolution" in bash  # submit/exit folded in
    assert "## Framework tools" not in bash  # merged away

    # A bash-only dialect: hermes_xml teaches the same one-command-per-turn loop with the
    # Qwen/hermes <tool_call> markup instead of a fenced block; submit/exit fold in the same way.
    hermes = build("hermes_xml")
    assert "<tool_call>" in hermes and "<function=Bash>" in hermes
    assert "framework/control.py SubmitSolution" in hermes
    assert "```bash" not in hermes  # not the fenced-block dialect

    # glm dialect: GLM's native <tool_call>Bash<arg_key>command</arg_key><arg_value>...
    # The example must PARSE under the same regex alancode's GLMFormat uses, or it re-teaches the
    # exact imitation drift this format exists to fix.
    glm = build("glm")
    assert "framework/control.py SubmitSolution" in glm
    glm_pat = re.compile(
        r"<tool_call>(\w+)((?:<arg_key>.*?</arg_key><arg_value>.*?</arg_value>)+)</tool_call>",
        re.DOTALL,
    )
    parsed = [m.group(1) for m in glm_pat.finditer(glm)]
    assert parsed and all(name == "Bash" for name in parsed)  # every taught call parses as Bash

    native = build("client_cli")
    assert "## Framework tools" in native and "cat > code_library" not in native


def test_control_cli_prompt_and_binding_agree_for_every_protocol() -> None:
    """The seam invariant behind the glm 503 bug: a protocol whose PROMPT teaches the workdir
    control CLI must be exactly one task.py BINDS the channel for - both key off uses_control_cli.
    Guards every protocol so a new dialect cannot re-teach a channel that was never bound."""
    from regact.agent.capabilities import TOOL_PROTOCOLS, uses_control_cli
    from regact.prompt.builder import _control_channel_block

    for protocol in TOOL_PROTOCOLS:
        block = _control_channel_block(protocol, ["SubmitSolution", "ExitTask"])  # type: ignore[arg-type]
        teaches_control_cli = "framework/control.py" in block
        assert teaches_control_cli == uses_control_cli(protocol), protocol  # prompt == binding


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
    msg = builder.build_first_message()
    # the generic first message is agnostic: a thin "begin" that defers the how to the
    # system prompt - it names no controller, no solution.py, no features.
    assert "task" in msg.lower()
    assert "controller" not in msg.lower() and "solution.py" not in msg
    assert "feature" not in msg.lower()
    framed = builder.build_first_message("OBS_GRID_HERE")
    assert "OBS_GRID_HERE" in framed and "first observation" in framed.lower()
