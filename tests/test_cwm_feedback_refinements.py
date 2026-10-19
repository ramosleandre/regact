"""Agent-facing evidence stays honest across real exploration, truncation and helper calls."""

import base64
import json
import runpy
import subprocess
import sys

import pytest

from regact.config.schema import AgentName
from regact.protocols.cwm.feedback import present
from regact.protocols.cwm.session import CwmTool
from regact.tools.base import ToolContext
from test_cwm_protocol import accept, exploration
from test_cwm_protocol import rig as rig


@pytest.mark.integration
async def test_feedback_reports_real_execution_even_without_novelty(rig):
    c, _ = rig
    accept(c)
    exploration(c, (1, 1))
    tool = CwmTool("RunController", c)
    context = ToolContext(cwd=str(c.workdir))
    completed = await tool.call({}, context)
    fields = json.loads(completed.data)
    assert fields["status"] == "Completed"
    assert fields["real_actions"] == 2 and fields["actual_novel_observations"] == 0
    assert "metrics" in fields
    assert "simulation_actions" not in fields and "predicted_novel_observations" not in fields
    source = c.workdir / "world_model/model_transition.py"
    source.write_text(source.read_text() + "\n# changed\n")
    refused = await tool.call({}, context)
    assert refused.is_error
    fields = json.loads(refused.data)
    assert set(fields) == {"status", "message"} and fields["status"] == "Refused"
    assert "No real action was taken" in fields["message"]


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
def test_state_size_is_shown_only_when_size_caused_the_refusal(rig):
    c, _ = rig
    accept(c)
    result = c.tool("UpdateCodeWorldModel", {})
    assert "state_size" not in present("UpdateCodeWorldModel", result, c.options, c.config.limits)
    # Keep full diagnostics internally; other refusals do not show the size either.
    assert "smallest_state" in result and "largest_state" in result
    refused = present("UpdateCodeWorldModel", {
        **result, "accepted": False, "failures": {"reconstruction_mismatch": 1}
    }, c.options, c.config.limits)
    assert "state_size" not in refused
    ratio = result["state_obs_size_ratio"]
    # The threshold is strict: equality must also return useful size witnesses.
    c.options.threshold_max_state_obs_size_ratio = ratio
    rejected = c.tool("UpdateCodeWorldModel", {})
    public = present("UpdateCodeWorldModel", rejected, c.options, c.config.limits)
    assert public["status"] == "Refused" and public["failures"]["compression_ratio"]
    size = public["state_size"]
    assert size["smallest_state"]["state_bytes"] <= size["largest_state"]["state_bytes"]
    for key in ("smallest_state", "largest_state"):
        item = size[key]
        assert c.store.observation(item["observation_id"])
        assert item["ratio"] == round(item["state_bytes"] / item["observation_bytes"], 3)


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
    starter = runpy.run_path(str(c.workdir / "controller.py"))
    assert starter["get_controller"]().is_done(object()) is False
    help_result = subprocess.run([sys.executable, str(c.workdir / "framework/commands.py"), "--help"], text=True, capture_output=True, check=True)
    for name in ("UpdateCodeWorldModel", "PlanInCWM", "RunController"):
        assert f"{name}: " in help_result.stdout
    assert "request" not in help_result.stdout.lower()
