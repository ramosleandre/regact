"""The two managed protocols share the brief and extend only declared sections."""

import re
from pathlib import Path

import pytest

from regact.config.schema import (
    AgentConfig,
    AgentName,
    Lifecycle,
    ProblemConfig,
    ProtocolConfig,
    RunConfig,
)
from regact.features.base import FeatureContext
from regact.problems.arc_agi.problem import ArcAgiProblem
from regact.problems.minigrid.problem import MiniGridProblem
from regact.protocols.cwm.commands import enabled_commands
from regact.protocols.managed.prompting import reset_commands
from regact.protocols.registry import build_protocol


def sections(prompt):
    parts = re.split(r"^# (.+)\n", prompt, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


@pytest.mark.parametrize("game", ["arc", "minigrid"])
@pytest.mark.parametrize("lifecycle", list(Lifecycle))
@pytest.mark.parametrize("dialect", ["client_cli", "hermes_xml", "native"])
@pytest.mark.parametrize("image_count", [0, 3])
def test_common_sections_and_runtime_guidance(game, lifecycle, dialect, image_count):
    if game == "arc":
        problem = ArcAgiProblem(
            environments_dir=str(Path(__file__).resolve().parents[1] / "environnement")
        )
        task = "ls20"
    else:
        problem = MiniGridProblem(fully_obs=True)
        task = "MiniGrid-Empty-5x5-v0"
    prompts = {}
    for name in ("vanilla", "cwm"):
        config = RunConfig(
            AgentConfig(AgentName.CLAUDE, vision=True),
            ProblemConfig(problem.name, lifecycle=lifecycle),
            protocol=ProtocolConfig(
                name,
                {
                    "n_tmp_images_saved_per_exploration": image_count,
                    "execution": {
                        "max_seconds_per_call": None,
                        "max_seconds_per_RunController": 13,
                    },
                },
            ),
        )
        protocol = build_protocol(config)
        commands = (
            enabled_commands(protocol.options) if name == "cwm" else {"RunController": ""}
        )
        prompts[name] = protocol.build_system_prompt(
            problem,
            task,
            tool_protocol=dialect,
            tool_names=[*commands, *reset_commands(config, problem)],
            verbalize_variant="off",
        )
        prompt = prompts[name]
        assert not re.search(r"__[A-Z_]+__", prompt)
        assert "phase-based workflow" not in prompt
        assert "seconds of active execution" not in prompt
        assert "Each isolated callback" not in prompt
        assert "An unlimited setting" not in prompt
        if image_count:
            assert "up to 3 observation images" in prompt
            assert "tmp/images/obs_id_<ID>.png" in prompt
            assert "including a refused one" in prompt
        else:
            assert "Automatic image previews are disabled" in prompt
            assert "tmp/images/" not in prompt
        assert ("continues from the current live observation" in prompt) == (
            lifecycle is Lifecycle.SINGLE_INSTANCE
        )
        assert "Every call creates a fresh controller; its private memory is not carried over." in prompt
        assert "An environment reporting is_done=True stops this controller call." in prompt
        if lifecycle is Lifecycle.MULTI_INSTANCE:
            assert "ResetLevel" not in prompt and "ResetEnvironment" not in prompt
        elif game == "arc":
            assert "You can reset the current ARC level, preserving completed levels with the ResetLevel command." in prompt
            assert "You can reset the whole environment from level 1 with the ResetEnvironment command." in prompt
        else:
            assert "ResetLevel" not in prompt
            assert "You can reset the whole environment from its initial state with the ResetEnvironment command." in prompt
    vanilla, cwm = (sections(prompts[name]) for name in ("vanilla", "cwm"))
    assert vanilla.keys() == cwm.keys()
    assert list(vanilla)[0] == "Role"
    headings = list(vanilla)
    game_heading = next(name for name in headings if name.startswith("Game:"))
    assert headings.index("Working directory") < headings.index(game_heading)
    assert headings.index(game_heading) < headings.index("Workflow")
    differences = {name for name in vanilla if vanilla[name] != cwm[name]}
    assert differences == {
        "Working directory",
        "Workflow",
        "Framework tools" if dialect == "native" else "Framework commands",
    }
    assert "act(obs)" in vanilla["Workflow"]
    for absent in ("CWM", "world_model/", "simulate.py", "UpdateCodeWorldModel", "PlanInCWM"):
        assert absent not in prompts["vanilla"]
    files = list(
        build_protocol(
            RunConfig(
                AgentConfig(AgentName.SCRIPTED),
                ProblemConfig(problem.name),
                protocol=ProtocolConfig("vanilla"),
            )
        ).templates(FeatureContext(problem_name=problem.name, task_name=task, workdir=""))
    )
    commands_file = next(f.content for f in files if f.relpath == "framework/commands.py")
    assert "Run a framework command" in commands_file and "CWM" not in commands_file


@pytest.mark.parametrize("name", ["vanilla", "cwm"])
def test_text_only_agent_is_not_told_to_open_images(name):
    """An Alan agent has no image-reading tool; pointing it at one wasted pilot calls."""
    problem = ArcAgiProblem(
        environments_dir=str(Path(__file__).resolve().parents[1] / "environnement")
    )
    config = RunConfig(
        AgentConfig(AgentName.ALAN), ProblemConfig(problem.name), protocol=ProtocolConfig(name)
    )
    protocol = build_protocol(config)
    commands = enabled_commands(protocol.options) if name == "cwm" else {"RunController": ""}
    prompt = protocol.build_system_prompt(
        problem,
        "ls20",
        tool_protocol="hermes_xml",
        tool_names=[*commands, *problem.reset_commands()],
        verbalize_variant="off",
    )
    assert "image-reading tool" not in prompt
    assert "cannot display images" in prompt


@pytest.mark.parametrize("name", ["vanilla", "cwm"])
def test_text_only_agent_is_never_offered_images(name, tmp_path):
    config = RunConfig(
        AgentConfig(AgentName.ALAN),
        ProblemConfig("minigrid"),
        protocol=ProtocolConfig(name),
    )
    protocol = build_protocol(config)
    problem = MiniGridProblem(fully_obs=True)
    prompt = protocol.build_system_prompt(
        problem,
        "MiniGrid-Empty-5x5-v0",
        tool_protocol="client_cli",
        tool_names=["RunController"],
        verbalize_variant="off",
    )
    assert "Your tools cannot display images" in prompt
    files = protocol.templates(
        FeatureContext(problem_name="minigrid", task_name="MiniGrid-Empty-5x5-v0", workdir="")
    )
    for f in files:
        assert "save_image" not in f.content and "image-reading tool" not in f.content, f.relpath
