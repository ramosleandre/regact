"""Agent-visible prompt/files agree across dialects, game modes and optional helpers."""

import re
from pathlib import Path

import pytest

from regact.config.schema import (
    AgentConfig,
    AgentName,
    InfoMode,
    ProblemConfig,
    ProtocolConfig,
    RunConfig,
)
from regact.problems.arc_agi.problem import ArcAgiProblem
from regact.protocols.cwm.commands import enabled_commands
from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.prompting import workspace_docs
from regact.protocols.cwm.protocol import CwmProtocol
from regact.security.detection import flag_tool_call
from regact.security.policy import default_policy
from regact.workspace.bootstrap import Workspace


@pytest.mark.parametrize("dialect", ["client_cli", "bash_block", "hermes_xml", "glm", "native"])
@pytest.mark.parametrize("helpers", [True, False])
@pytest.mark.parametrize("planner", [True, False])
def test_prompt_only_teaches_available_cwm_interfaces(tmp_path, dialect, helpers, planner):
    cfg = RunConfig(
        AgentConfig(AgentName.SCRIPTED),
        ProblemConfig("arc_agi"),
        protocol=ProtocolConfig(
            "cwm", {"workspace_helpers_enabled": helpers, "planner": {"enabled": planner}}
        ),
    )
    problem = ArcAgiProblem(
        environments_dir=str(Path(__file__).resolve().parents[1] / "environnement")
    )
    protocol = CwmProtocol(cfg)
    helper_files = problem.helper_templates("ls20", direct_interaction=False)
    Workspace(str(tmp_path)).bootstrap(
        [],
        templates=protocol.templates,
        expose_environment=False,
        command_script=protocol.command_script,
        problem_name=problem.name,
        task_name="ls20",
        env_base_url="http://unused",
        game_id="ls20",
        lifecycle=cfg.problem.lifecycle,
        helper_templates=helper_files,
    )
    prompt = protocol.build_system_prompt(
        problem,
        "ls20",
        tool_protocol=dialect,
        tool_names=list(enabled_commands(protocol.options)),
        verbalize_variant="off",
    )
    all_docs = [p for p in tmp_path.rglob("*.md")]
    text = prompt + "\n".join(p.read_text() for p in all_docs)
    for absent in (
        "SubmitSolution",
        "ExitTask",
        "code_library",
        "make_env",
        "solution.py",
        "obs.frame",
        "__SIZE_RATIO__",
        "__SCRIPT_PATH__",
        "exploration.py",
        "framework/control.py",
    ):
        assert absent not in text
    assert 'obs["frame"]' in prompt
    assert (tmp_path / "framework/arc_agi_helper.py").is_file()
    assert (tmp_path / "framework/commands.py").is_file()
    assert (tmp_path / "controller.py").is_file()
    assert not (tmp_path / "framework/control.py").exists()
    assert not (tmp_path / "exploration.py").exists()
    assert not (tmp_path / "code_library").exists()
    assert "model_env.py" not in text
    assert ("cwm_env.py" in text) == helpers
    assert ("simulate.py" in text) == helpers
    assert "framework/simulation.py" not in text
    assert "simulates it again" not in text
    assert "No preliminary simulation or predicted novelty is required" in text
    assert "source files or imported dependencies have changed" in text
    assert "CWM_INTERFACE.md" not in text
    assert not (tmp_path / "CWM_INTERFACE.md").exists()
    assert sorted(p.name for p in (tmp_path / "world_model").iterdir()) == [
        "__init__.py",
        "model_parser.py",
        "model_render.py",
        "model_state.py",
        "model_transition.py",
    ]
    assert ("Working in the terminal" in prompt) == (dialect in ("bash_block", "hermes_xml", "glm"))
    assert ("python framework/commands.py PlanInCWM" in prompt) == (planner and dialect != "native")
    assert (tmp_path / "goal.py").exists() == planner
    assert (tmp_path / "docs/plan_in_CWM.md").exists() == planner
    for p in tmp_path.rglob("*.py"):
        compile(p.read_text(), str(p), "exec")
    if not planner:
        all_text = text + "\n".join(p.read_text() for p in tmp_path.rglob("*.py"))
        for absent in ("PlanInCWM", "goal.py", "plans/", "plan_in_CWM.md"):
            assert absent not in all_text
    assert 'obs["info"]["milestones"]' in text
    assert "The list is not cumulative" in text
    # Every generated file is discoverable in the inventory; optional files cannot leak.
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert p.name in prompt
    for doc in all_docs:
        for target in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
            assert (doc.parent / target).is_file(), (doc, target)
    assert (
        flag_tool_call("Bash", {"command": "cat framework/arc_agi_helper.py"}, default_policy())
        == []
    )


def test_docs_show_run_values_and_have_no_unresolved_markers():
    options = CwmConfig.from_mapping(
        {
            "threshold_max_state_obs_size_ratio": 0.25,
            "execution": {"max_seconds_per_call": None},
            "data_api": {"max_items": 7},
        }
    )
    docs = {f.relpath: f.content for f in workspace_docs(options, vision=True)}
    assert "**0.25**" in docs["docs/CWM_modeling_phase.md"]
    assert "unlimited second limit" in docs["docs/CWM_modeling_phase.md"]
    assert not re.search(r"__[A-Z_]+__", "\n".join(docs.values()))


def test_game_prompt_minimal_mode_uses_dictionary_fields():
    problem = ArcAgiProblem(
        environments_dir=str(Path(__file__).resolve().parents[1] / "environnement")
    )
    prompt = problem.build_prompt("ls20", info_mode=InfoMode.MINIMAL, direct_interaction=False)
    assert 'obs["frame"]' in prompt and "make_env" not in prompt


@pytest.mark.integration
def test_moved_action_helper_works_in_isolated_submission(rig):  # noqa: F811
    from test_cwm_protocol import accept

    c, server = rig
    helper = ArcAgiProblem(
        environments_dir=str(Path(__file__).resolve().parents[1] / "environnement")
    ).helper_templates("ls20", direct_interaction=False)[0]
    (c.workdir / helper.relpath).write_text(helper.content)
    accept(c)
    (
        c.workdir / "controller.py"
    ).write_text('''"""Test new states with the provided action constant."""
from framework.arc_agi_helper import ACTION1
from framework.action_list_controller import ExplorationControllerFromListActions

def get_controller():
    return ExplorationControllerFromListActions([ACTION1] * 4)
''')
    result = c.tool("RunController", {})
    assert not result.get("error"), result
    assert result["real_actions"] == 4


from test_cwm_protocol import rig  # noqa: E402, F401


@pytest.mark.integration
def test_data_api_module_docstring_example_runs(rig, monkeypatch, capsys):  # noqa: F811
    import sys
    import textwrap
    from types import ModuleType

    import numpy as np

    c, _ = rig
    c.collect_initial()
    monkeypatch.setattr(
        c.problem, "render_frame", lambda obs: np.asarray([obs.frame], dtype=np.uint8)
    )
    module = ModuleType("framework.data_api")
    code = (c.workdir / "framework/data_api.py").read_text()
    exec(compile(code, "framework/data_api.py", "exec"), module.__dict__)
    module._query = lambda op, **args: c.data({"op": op, **args})
    package = ModuleType("framework")
    package.data_api = module
    monkeypatch.setitem(sys.modules, "framework", package)
    monkeypatch.setitem(sys.modules, "framework.data_api", module)
    monkeypatch.chdir(c.workdir)
    example = module.__doc__.split("Example from a workspace script:\n", 1)[1].split("\n\n", 1)[0]
    exec(textwrap.dedent(example), {})
    assert (c.workdir / "observation.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "Image of observation" in capsys.readouterr().out
