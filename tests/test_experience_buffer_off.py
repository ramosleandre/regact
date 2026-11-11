"""protocol.exp_buffer=false (C1): each command prints a summary line and the current
observation; the controller's error and invalid action are bounded in every configuration."""

import json

import pytest

from regact.config.schema import AgentConfig, AgentName, ProblemConfig, ProtocolConfig, RunConfig
from regact.protocols.managed.observation_text import observation_text
from regact.protocols.managed.session import ManagedSession, ManagedTool
from regact.protocols.registry import build_protocol
from regact.tools.base import ToolContext
from test_cwm_protocol import exploration
from test_managed_protocol import make_rig  # noqa: F401  (fixture)

OBS = {"reward": 0.0, "is_done": False, "available_actions": [1, 2], "info": {"milestones": []}}


def _split(text, n_rows):
    lines = text.split("\n")
    return lines[:n_rows], "\n".join(lines[n_rows:])


def test_a_single_hex_grid_is_printed_above_its_placeholder():
    obs = {"frame": [[0, 10, 15], [3, 4, 5]], **OBS}
    rows, rest = _split(observation_text(obs, "hex"), 2)
    assert rows == ["0af", "345"]
    assert '  "frame": <the grid above>,' in rest.split("\n")
    assert json.loads(rest.replace("<the grid above>", json.dumps(obs["frame"]))) == obs


@pytest.mark.parametrize(
    "n, placeholder",
    [(1, "[<the grid above>]"), (3, "[<2 grids not shown>, <the grid above>]")],
)
def test_a_list_of_grids_shows_the_last_grid_and_its_place(n, placeholder):
    obs = {"frame": [[[i, 12]] for i in range(n)], **OBS}
    rows, rest = _split(observation_text(obs, "hex"), 1)
    assert rows == [f"{n - 1}c"]
    assert f'  "frame": {placeholder},' in rest.split("\n")


def test_an_empty_frame_list_prints_as_json():
    obs = {"frame": [], **OBS}
    assert json.loads(observation_text(obs, "hex")) == obs


@pytest.mark.parametrize(
    "frame, cells",
    [
        ([[0, 16], [1, 2]], "hex"),  # a value above 15 falls back to JSON
        ([[0, 1], [2]], "hex"),  # ragged rows are not a grid
        ([[5, 28], [1, 2]], "scalar"),
        ([[[0, 7], [0]], [[1], [0, 1, 6]]], "list"),
        ({"image": [[[1, 0, 0], [2, 5, 0]]], "direction": 0, "mission": "get the key"}, "list"),
    ],
)
def test_other_frames_print_as_json_one_row_per_line(frame, cells):
    obs = {"frame": frame, **OBS}
    text = observation_text(obs, cells)
    assert json.loads(text) == obs
    assert all(len(line) < 200 for line in text.split("\n"))


def test_grid_rows_are_one_line_each():
    text = observation_text({"frame": [[[0, 7], [0]], [[1], [0, 1, 6]]], **OBS}, "list")
    assert '    [[0,7],[0]],' in text.split("\n")
    assert '    [[1],[0,1,6]]' in text.split("\n")


def _config(protocol, **options):
    return RunConfig(
        AgentConfig(AgentName.SCRIPTED),
        ProblemConfig("fake"),
        protocol=ProtocolConfig(protocol, options),
    )


def test_cwm_needs_the_experience_buffer():
    with pytest.raises(ValueError, match="exp_buffer=true"):
        build_protocol(_config("cwm", exp_buffer=False))


@pytest.mark.parametrize("protocol", ["vanilla", "cwm"])
def test_acting_without_programs_is_not_available_yet(protocol):
    with pytest.raises(ValueError, match="policy_programs"):
        build_protocol(_config(protocol, policy_programs=False)).validate()


def test_buffer_previews_need_the_buffer():
    config = _config("vanilla", exp_buffer=False, n_tmp_images_saved_per_exploration=2)
    config.agent.vision = True
    with pytest.raises(ValueError, match="exp_buffer=false"):
        build_protocol(config).validate()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_without_the_buffer_a_call_prints_a_summary_and_the_observation(make_rig):
    c = make_rig(exp_buffer=False, target=1)
    assert not (c.workdir / "framework" / "data_api.py").exists()
    with pytest.raises(ValueError, match="no experience buffer"):
        c.data({"op": "summary"})
    session = ManagedSession(coordinator=c)
    assert json.loads(session.first_observation_text()) == c.initial
    exploration(c, (1, 1))
    out = await ManagedTool("RunController", c).call({}, ToolContext(cwd=str(c.workdir)))
    first, _, rest = out.data.partition("\n")
    assert first == "RunController 1: the action list finished after 2 actions"
    assert json.loads(rest) == c.store.observation(c.current_id)
    assert json.loads(rest)["frame"] == [2] * 30
    reset = await ManagedTool("ResetEnvironment", c).call({}, ToolContext(cwd=str(c.workdir)))
    first, _, rest = reset.data.partition("\n")
    assert first == "ResetEnvironment: the environment restarted from its initial state"
    assert json.loads(rest)["frame"] == [0] * 30
    assert "The current observation" in session.resume_notice()


@pytest.mark.integration
@pytest.mark.parametrize("exp_buffer", [True, False])
def test_a_controller_error_and_its_invalid_action_are_bounded(make_rig, exp_buffer):
    c = make_rig(exp_buffer=exp_buffer, target=1)
    (c.workdir / "controller.py").write_text(
        '"""Smuggle data out."""\n'
        "class C:\n"
        "    def is_done(self, obs): return False\n"
        "    def act(self, obs): return list(range(1000))\n"
        "def get_controller(): return C()\n"
    )
    result = c.tool("RunController", {})
    assert result["stop_reason"] == "controller_error"
    assert len(result["error"]) <= 200
    assert len(result["error_context"]["action"]) <= 200
    (c.workdir / "controller.py").write_text(
        '"""Smuggle data out."""\n'
        "class C:\n"
        "    def is_done(self, obs): return False\n"
        "    def act(self, obs): raise ValueError('x' * 5000)\n"
        "def get_controller(): return C()\n"
    )
    result = c.tool("RunController", {})
    assert result["stop_reason"] == "controller_error"
    assert len(result["error"]) <= 200


@pytest.mark.integration
def test_is_done_is_optional(make_rig):
    c = make_rig(target=1)
    (c.workdir / "controller.py").write_text(
        '"""Walk until the game ends."""\n'
        "class C:\n"
        "    def act(self, obs): return 1\n"
        "def get_controller(): return C()\n"
    )
    result = c.tool("RunController", {})
    assert result["stop_reason"] == "environment_done"
    assert result["real_actions"] == 6


@pytest.mark.parametrize("vision", [False, True])
def test_the_prompt_without_the_buffer_describes_the_printed_result(tmp_path, vision):
    from pathlib import Path
    from types import SimpleNamespace

    from regact.orchestration.task import _bootstrap_workdir
    from regact.problems.arc_agi.problem import ArcAgiProblem

    problem = ArcAgiProblem(
        environments_dir=str(Path(__file__).resolve().parents[1] / "environnement")
    )
    config = _config("vanilla", exp_buffer=False)
    config.agent.vision = vision
    protocol = build_protocol(config)
    _bootstrap_workdir(
        config,
        problem,
        "ls20",
        workdir=str(tmp_path),
        conn=SimpleNamespace(base_url="http://unused"),
        protocol=protocol,
    )
    assert not (tmp_path / "framework" / "data_api.py").exists()
    assert not (tmp_path / "framework" / "arc_agi_helper.py").exists()
    prompt = protocol.build_system_prompt(
        problem, "ls20", tool_protocol="client_cli", tool_names=["RunController"], verbalize_variant="off"
    )
    assert "data_api" not in prompt and "the dataset" not in prompt
    assert "one hex digit per cell" in prompt and "a placeholder marks the grid's place" in prompt
    assert ("tmp/images/final.png" in prompt) == vision
    assert "image" not in prompt.lower() or vision
