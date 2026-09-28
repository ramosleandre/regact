"""Protocol boundary and byte parity with pre-refactor main (see fixture source_commit)."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from regact.agent.capabilities import ToolProtocol
from regact.agent.events import ToolCall
from regact.agent.scripted_agent import ScriptedAgent
from regact.config.loader import run_config_from_mapping
from regact.config.schema import (
    AgentConfig,
    AgentName,
    ControllerConfig,
    Lifecycle,
    LimitsConfig,
    ProblemConfig,
    ProtocolConfig,
    RunConfig,
    redacted_config_dict,
)
from regact.env.renderer import RawRenderer
from regact.envclient.obs import Obs
from regact.features.base import FeatureContext, Hook, HookPhase
from regact.obs.result import EvalResult
from regact.orchestration.task import run_task
from regact.problems.base import BaseProblem
from regact.protocols import registry
from regact.protocols.base import ExperimentProtocol, ProtocolContext, ProtocolSession
from regact.protocols.policy_search import PolicySearchProtocol, PolicySearchSession
from regact.protocols.registry import build_protocol, register_protocol
from regact.session.state import ExperimentState
from regact.testing.fakes import FakeNativeEnv
from regact.tools.base import Tool, ToolContext, ToolOutput
from regact.workspace.bootstrap import Workspace
from regact.workspace.templates import TemplateFile

_BASELINE = json.loads(
    (Path(__file__).parent / "fixtures/policy_search_artifacts.json").read_text()
)


def _config(**kwargs: Any) -> RunConfig:
    return RunConfig(
        agent=AgentConfig(name=AgentName.SCRIPTED), problem=ProblemConfig(name="fake"), **kwargs
    )


@pytest.mark.parametrize(
    "key,expected", [(k, v) for k, v in _BASELINE["prompts"].items() if k.split("|")[2] == "False"]
)
def test_policy_search_prompt_bytes_match_main(key: str, expected: str) -> None:
    dialect, exit_enabled, cwm, lifecycle, verbalize = key.split("|")
    config = _config(
        controller=ControllerConfig(exit_task_enabled=exit_enabled == "True"),
        features={"cwm": {}} if cwm == "True" else {},
    )
    config.problem.lifecycle = Lifecycle(lifecycle)
    problem = SimpleNamespace(
        name="fake", build_prompt=lambda *a, **k: "# Game: fake\nReach the goal."
    )
    prompt = build_protocol(config).build_system_prompt(
        problem,  # type: ignore[arg-type]
        "corridor",
        tool_protocol=dialect,  # type: ignore[arg-type]
        tool_names=["SubmitSolution", *(["ExitTask"] if exit_enabled == "True" else [])],
        verbalize_variant=verbalize,
    )
    # September 28: the shared environment now supplies numeric rewards. This
    # approved one-line contract update is the only exception to the old brief.
    reward_line = "- `obs.reward` - reward from the preceding action"
    assert reward_line + "\n" in prompt
    baseline_prompt = prompt.replace(reward_line + "\n", reward_line + " (may be `None`)\n")
    assert hashlib.sha256(baseline_prompt.encode()).hexdigest() == expected


@pytest.mark.parametrize(
    "key,expected",
    [(k, v) for k, v in _BASELINE["workspace"].items() if k.split("|")[0] == "False"],
)
def test_policy_search_workspace_bytes_match_main(
    key: str, expected: dict[str, str], tmp_path: Path
) -> None:
    cwm, lifecycle = key.split("|")
    config = _config(features={"cwm": {}} if cwm == "True" else {})
    config.problem.lifecycle = Lifecycle(lifecycle)
    protocol = build_protocol(config)
    Workspace(str(tmp_path)).bootstrap(
        [],
        templates=protocol.templates,
        problem_name="fake",
        task_name="corridor",
        env_base_url="http://env:1234",
        game_id="corridor",
        lifecycle=config.problem.lifecycle,
    )
    actual = {
        str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert actual == expected


def test_first_message_bytes_match_main() -> None:
    from regact.prompt.builder import PromptBuilder

    assert [PromptBuilder().build_first_message(o) for o in (None, "first observation")] == (
        _BASELINE["first_messages"]
    )


def test_config_default_explicit_and_saved_round_trip() -> None:
    raw = {"agent": {"name": "scripted"}, "problem": {"name": "fake"}}
    implicit = run_config_from_mapping(raw)
    for selection in ("policy_search", {"name": "policy_search"}):
        explicit = run_config_from_mapping({**raw, "protocol": selection})
        assert explicit == implicit
        assert run_config_from_mapping(redacted_config_dict(explicit)) == implicit
    config = run_config_from_mapping({**raw, "protocol": {"name": "future", "budget": 15}})
    assert config.protocol.options == {"budget": 15}
    assert redacted_config_dict(config)["protocol"] == {"name": "future", "budget": 15}
    assert run_config_from_mapping(redacted_config_dict(config)) == config


def test_hydra_protocol_selection_preserves_controller_overrides() -> None:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import regact

    with initialize_config_dir(
        version_base=None, config_dir=str(Path(regact.__file__).parent / "conf")
    ):
        implicit = compose(config_name="config", overrides=["controller.exit_task_enabled=false"])
        explicit = compose(
            config_name="config",
            overrides=["protocol=policy_search", "controller.exit_task_enabled=false"],
        )
    assert OmegaConf.to_container(implicit, resolve=True) == OmegaConf.to_container(
        explicit, resolve=True
    )
    typed = run_config_from_mapping(OmegaConf.to_container(explicit, resolve=True))
    assert typed.protocol.name == "policy_search"
    assert not typed.controller.exit_task_enabled


def test_registry_rejects_unknown_duplicate_and_invalid_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "_REGISTRY", {})
    with pytest.raises(ValueError, match="unknown experiment protocol"):
        build_protocol(_config(protocol=ProtocolConfig(name="missing")))
    with pytest.raises(ValueError, match="controller"):
        build_protocol(_config(protocol=ProtocolConfig(options={"max_moves": 2})))
    with pytest.raises(ValueError, match="already registered"):
        register_protocol("policy_search", _ToyProtocol)
    register_protocol("toy", _ToyProtocol)
    with pytest.raises(ValueError, match="already registered"):
        register_protocol("toy", _ToyProtocol)


def test_factory_error_is_not_disguised_as_unknown_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(config: RunConfig) -> ExperimentProtocol:
        raise KeyError("implementation bug")

    monkeypatch.setattr(registry, "_REGISTRY", {"broken": broken})
    with pytest.raises(KeyError, match="implementation bug"):
        build_protocol(_config(protocol=ProtocolConfig(name="broken")))


def test_policy_search_completion_precedence_and_reminder_modes() -> None:
    state = ExperimentState(problem_name="p", task_name="t", exit_requested=True)
    state.last_submission_results = {
        "aggregate": {"evaluation_complete": True, "n_errors": 0},
        "episodes": [],
    }
    session = PolicySearchSession(experiment=state, is_perfect=lambda _: True)
    assert session.stop_reason() == "solved"
    state.last_submission_results["error"] = "failed"
    assert session.stop_reason() == "agent_exit"
    state.exit_requested = False
    assert session.stop_reason() is None
    assert "ExitTask" in session.reminder(1)
    session.exit_task_enabled = False
    assert "ExitTask" not in session.reminder(1)
    assert "Submit NOW" in session.reminder(10)
    state.submission_count = 1
    assert "Submit NOW" not in session.reminder(10)


class _Problem(BaseProblem):
    name = "fake"

    def make_env(self, task_name: str) -> Any:
        return FakeNativeEnv()

    def get_task_names(self) -> list[str]:
        return ["corridor"]

    def obs_renderer(self, task_name: str, *, mode: Any) -> RawRenderer:
        return RawRenderer()

    def compute_episode_metrics(self, final_obs: Obs, *, steps: int) -> dict[str, Any]:
        return {"success": final_obs.is_done, "steps": steps}

    def aggregate_episode_metrics(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "success_rate": sum(bool(e.get("success")) for e in episodes) / max(1, len(episodes))
        }

    def config_kwargs(self) -> dict[str, Any]:
        return {}

    def build_prompt(self, task_name: str, *, info_mode: Any, obs_mode: Any = None) -> str:
        return "Game instructions."


class _FinishHook(Hook):
    phase = HookPhase.TEARDOWN

    def __init__(self, ctx: ProtocolContext) -> None:
        self.ctx = ctx

    async def run(self) -> EvalResult | None:
        # The shared runtime must persist its verdict before protocol finalization.
        state = json.loads(Path(self.ctx.output_dir, "logs/experiment_state.json").read_text())
        Path(self.ctx.output_dir, "finished.txt").write_text(state["exit_reason"])
        return None


@dataclass(kw_only=True)
class _ToySession(ProtocolSession):
    phase: int = 0

    def stop_reason(self) -> str | None:
        return "toy_complete" if self.phase == 2 else None

    def reminder(self, reminders: int) -> str:
        return f"Toy phase {self.phase}"


class _Advance(Tool):
    name = "Advance"
    description = "Advance the toy protocol."
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    def __init__(self, session: _ToySession) -> None:
        self.session = session

    async def call(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        self.session.phase += 1
        return ToolOutput(data={"phase": self.session.phase})


class _ToyProtocol(ExperimentProtocol):
    """A deliberately different workflow: no controller, submission or policy-search prompt."""

    name = "toy"

    def __init__(self, config: RunConfig) -> None:
        self.config = config

    def validate(self) -> None:
        pass

    def templates(self, ctx: FeatureContext) -> list[TemplateFile]:
        return [TemplateFile("toy.py", "# Toy artifact\n")]

    def build_system_prompt(
        self,
        problem: BaseProblem,
        task_name: str,
        *,
        tool_protocol: ToolProtocol,
        tool_names: list[str],
        verbalize_variant: str,
    ) -> str:
        return "Toy workflow. Call Advance twice."

    def bind(self, ctx: ProtocolContext) -> ProtocolSession:
        # Poison legacy state: a different protocol must not stop on these fields.
        ctx.experiment.exit_requested = True
        ctx.experiment.last_submission_results = {
            "aggregate": {"evaluation_complete": True, "success_rate": 1},
            "episodes": [],
        }
        session = _ToySession(hooks=[_FinishHook(ctx)])
        session.tools = [_Advance(session)]
        return session


@pytest.mark.parametrize("limit,reason,turns", [(10, "toy_complete", 2), (1, "tool_call_limit", 1)])
async def test_another_protocol_uses_shared_runner_without_controller_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int, reason: str, turns: int
) -> None:
    monkeypatch.setattr(registry, "_REGISTRY", {})
    register_protocol("toy", _ToyProtocol)
    config = _config(protocol=ProtocolConfig(name="toy"), limits=LimitsConfig(max_tool_calls=limit))
    agent = ScriptedAgent([[ToolCall("1", "Advance", {})], [ToolCall("2", "Advance", {})]])
    assert (
        await run_task(config, _Problem(), "corridor", output_dir=str(tmp_path), agent=agent)
        == reason
    )
    state = json.loads((tmp_path / "logs/experiment_state.json").read_text())
    assert state["turn"] == turns
    assert state["tool_calls_total"] == turns
    assert (tmp_path / "finished.txt").read_text() == reason
    assert (tmp_path / "workdir/toy.py").exists()
    assert not (tmp_path / "workdir/solution.py").exists()
    assert not (tmp_path / "workdir/submissions").exists()
    transcript = (tmp_path / "logs/transcript.jsonl").read_text()
    assert "Toy workflow" in transcript
    assert "SubmitSolution" not in transcript
    if turns == 2:
        assert "Toy phase 1" in transcript


async def test_unknown_protocol_fails_before_task_artifacts(tmp_path: Path) -> None:
    config = _config(protocol=ProtocolConfig(name="missing"))
    with pytest.raises(ValueError, match="unknown experiment protocol"):
        await run_task(config, _Problem(), "corridor", output_dir=str(tmp_path / "run"))
    assert not (tmp_path / "run").exists()


def test_feature_templates_see_preceding_controller_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = build_protocol(_config())
    assert isinstance(protocol, PolicySearchProtocol)
    feature = SimpleNamespace(
        templates=lambda ctx: [TemplateFile(relpath="extra.py", content="# extra")]
    )
    protocol.features.append(feature)
    original = feature.templates

    def dependent_templates(ctx: FeatureContext) -> list[TemplateFile]:
        assert Path(ctx.workdir, "solution.py").exists()
        assert Path(ctx.workdir, "code_library/base_controller.py").exists()
        return original(ctx)

    monkeypatch.setattr(feature, "templates", dependent_templates)
    Workspace(str(tmp_path)).bootstrap(
        [],
        templates=protocol.templates,
        problem_name="fake",
        task_name="corridor",
        env_base_url="http://env:1234",
        game_id="corridor",
        lifecycle=Lifecycle.MULTI_INSTANCE,
    )
    assert (tmp_path / "extra.py").exists()
