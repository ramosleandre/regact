"""Agent-facing evidence stays honest across screening, truncation and helper calls."""

import base64
import json
import runpy
import subprocess
import sys

import pytest

from regact.config.loader import _helper_from
from regact.config.schema import AgentName
from regact.protocols.cwm.feedback import present
from regact.protocols.cwm.session import CwmTool
from regact.protocols.cwm.validation import differences
from regact.tools.base import ToolContext
from test_cwm_protocol import accept, exploration
from test_cwm_protocol import rig as rig


def test_structural_differences_count_omissions_without_losing_shape_changes():
    expected = {"frame": [[1, 2], [3]], "info": {"a": 1}}
    observed = {"frame": [[9], [4, 5], [6]], "info": {"b": 2}}
    full = differences(expected, observed, 100)
    assert "differences_omitted" not in full
    paths = [d["path"] for d in full["differences"]]
    assert "frame.length" in paths and "frame[0].length" in paths
    assert "frame[1].length" in paths and "frame[0][0]" in paths
    short = differences(expected, observed, 2)
    assert short["differences"] == full["differences"][:2]
    assert short["differences_omitted"] == len(full["differences"]) - 2
    assert differences(0, 0.0, 1)["differences"][0]["path"] == "type"
    assert differences({"frame": [1]}, {"frame": 1}, 1)["differences"][0]["path"] == "frame.type"


@pytest.mark.integration
async def test_two_stages_only_report_real_metrics_after_real_execution(rig):
    c, _ = rig
    accept(c)
    exploration(c, (1, 1))
    tool = CwmTool("SubmitExplorationController", c)
    context = ToolContext(cwd=str(c.workdir))
    rejected = await tool.call({}, context)
    assert rejected.data.startswith("Running in Code World Model... Failure. Stopping here.\n{")
    assert "Running in actual env" not in rejected.data
    fields = json.loads(rejected.data.split("\n", 1)[1])
    assert fields["status"] == "Refused" and "metrics" not in fields
    assert "real_actions" not in fields
    exploration(c, (1, 1, 1, 1))
    completed = await tool.call({}, context)
    assert completed.data.startswith("Running in Code World Model... Success. Found 2 new observations.")
    assert "\n\nRunning in actual env... Done.\n" in completed.data
    fields = json.loads(completed.data[completed.data.index("{"):])
    assert fields["real_actions"] == 4 and "metrics" in fields
    assert "simulation_actions" not in fields and "predicted_novel_observations" not in fields


@pytest.mark.integration
async def test_planner_reports_saved_length_and_independent_warnings(rig):
    c, _ = rig
    accept(c)
    (c.workdir / "goal.py").write_text('"""Reach four."""\ndef achieved(s): return s.n==4\n')
    raw = c.tool("PlanInCWM", {})
    public = present("PlanInCWM", raw, c.options, c.config.limits)
    saved = runpy.run_path(str(c.workdir / raw["path"]))["ACTIONS"]
    assert f"{len(saved)}-length" in public["message"]
    assert public["elapsed_seconds"] == round(raw["elapsed_seconds"], 5)
    assert "Warning:" not in public["message"]
    for achieved, optimal in ((False, True), (True, False), (False, False)):
        variant = present("PlanInCWM", {**raw, "achieved": achieved, "optimality_proven": optimal}, c.options, c.config.limits)
        assert ("partial plan" in variant["message"]) == (not achieved)
        assert ("before optimality" in variant["message"]) == (not optimal)


@pytest.mark.integration
def test_state_size_extremes_only_appear_when_ratio_is_not_respected(rig):
    c, _ = rig
    accept(c)
    result = c.tool("UpdateCodeWorldModel", {})
    public = present("UpdateCodeWorldModel", result, c.options, c.config.limits)
    size = public["state_size"]
    assert size["ratio"] == size["state_bytes"] / size["observation_bytes"]
    assert "smallest_state" not in size and "largest_state" not in size
    # Keep full diagnostics internally; unrelated refusals do not expose the extremes.
    assert "smallest_state" in result and "largest_state" in result
    refused = present("UpdateCodeWorldModel", {
        **result, "accepted": False, "failures": {"reconstruction_mismatch": 1}
    }, c.options, c.config.limits)
    assert "smallest_state" not in refused["state_size"]
    assert "largest_state" not in refused["state_size"]
    # The threshold is strict: equality must also return useful size witnesses.
    c.options.threshold_max_state_obs_size_ratio = size["ratio"]
    rejected = c.tool("UpdateCodeWorldModel", {})
    public = present("UpdateCodeWorldModel", rejected, c.options, c.config.limits)
    assert public["status"] == "Refused" and public["failures"]["compression_ratio"]
    size = public["state_size"]
    assert size["smallest_state"]["state_bytes"] <= size["largest_state"]["state_bytes"]
    for key in ("smallest_state", "largest_state"):
        item = size[key]
        assert c.store.observation(item["observation_id"])
        assert item["ratio"] == item["state_bytes"] / item["observation_bytes"]


@pytest.mark.integration
def test_image_helper_prints_source_after_success_and_returns_none(rig, capsys):
    c, _ = rig
    module = runpy.run_path(str(c.workdir / "framework/data_api.py"))
    namespace = module["save_image"].__globals__
    namespace["_query"] = lambda *a, **k: {"png_base64": base64.b64encode(b"image").decode()}
    path = c.workdir / "image.png"
    cases = (
        ({"observation_id": 1}, "observation 1"),
        ({"transition_id": 2}, "transition 2 (after)"),
        ({"transition_id": 2, "which": "before"}, "transition 2 (before)"),
        ({"diagnostic_id": 3}, "diagnostic 3 (observed)"),
        ({"diagnostic_id": 3, "which": "predicted"}, "diagnostic 3 (predicted)"),
    )
    for args, label in cases:
        assert module["save_image"](path, **args) is None
        assert capsys.readouterr().out == f"Image of {label} saved at {path}\n"
        assert path.read_bytes() == b"image"
    with pytest.raises(FileNotFoundError):
        module["save_image"](c.workdir / "missing/image.png", observation_id=1)
    assert capsys.readouterr().out == ""
    assert "list_observation_ids" in module and "list_transition_ids" in module
    assert "list_observations" not in module and "list_transitions" not in module
    starter = runpy.run_path(str(c.workdir / "exploration.py"))
    assert starter["get_controller"]().is_done(object()) is False
    help_result = subprocess.run([sys.executable, str(c.workdir / "framework/control.py"), "--help"], text=True, capture_output=True, check=True)
    for name in ("UpdateCodeWorldModel", "PlanInCWM", "SubmitExplorationController"):
        assert f"{name}: " in help_result.stdout
    assert "request" not in help_result.stdout.lower()


def test_cwm_png_default_and_explicit_override():
    for agent in (AgentName.CODEX, AgentName.CLAUDE):
        assert _helper_from(None, agent).to_png
        assert not _helper_from(None, agent, protocol="cwm").to_png
        assert _helper_from({"to_png": True}, agent, protocol="cwm").to_png
