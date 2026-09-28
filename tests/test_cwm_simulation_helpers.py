"""Agent workflow contracts: inspectable IDs/images and local CWM simulation."""

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from regact.features.base import FeatureContext
from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.ids import expand_id_ranges, format_id_ranges
from regact.protocols.cwm.images import clear_images, save_preview, select_preview_ids
from regact.protocols.cwm.session import CwmTool
from regact.protocols.cwm.templates import templates
from regact.tools.base import ToolContext
from regact.viz.reader import ToolCallView, TurnItem, TurnView, _tag_tool_calls
from test_cwm_protocol import accept, exploration
from test_cwm_protocol import rig as rig


def test_inclusive_id_ranges_are_compact_bounded_and_loadable():
    assert format_id_ranges([8, 1, 3, 2, 3, 4, 11, 10]) == "[1:4, 8, 10:11]"
    assert expand_id_ranges("[1:4, 8, 10:11]", max_items=7) == [1, 2, 3, 4, 8, 10, 11]
    assert format_id_ranges([]) == "[]"
    assert expand_id_ranges("[]", max_items=3) == []
    for bad in ("[0]", "[3:1]", "[1,,2]", "[1:1000000000]", "[True]", "1:4"):
        with pytest.raises(ValueError):
            expand_id_ranges(bad, max_items=100)
    assert select_preview_ids([1, 2, 3, 2, 4, 5, 6], 4) == [1, 2, 5, 6]
    assert select_preview_ids([1, 2, 3, 4, 5, 6], 3) == [1, 2, 6]
    assert select_preview_ids([1, 2, 3], 1) == [1]
    assert select_preview_ids([1, 2, 3], 0) == []


def test_new_settings_and_optional_workspace_helpers():
    cfg = CwmConfig.from_mapping({})
    assert cfg.n_tmp_images_saved_per_exploration == 8 and cfg.workspace_helpers_enabled
    assert (
        CwmConfig.from_mapping(
            {"n_tmp_images_saved_per_exploration": 0}
        ).n_tmp_images_saved_per_exploration
        == 0
    )
    for options in (
        {"n_tmp_images_saved_per_exploration": -1},
        {"n_tmp_images_saved_per_exploration": 2.5},
        {"n_tmp_images_saved_per_exploration": True},
        {"workspace_helpers_enabled": 1},
    ):
        with pytest.raises(ValueError):
            CwmConfig.from_mapping(options)
    context = FeatureContext("fake", "counter", "/tmp/cwm-test")
    files = {x.relpath: x.content for x in templates(context, cfg)}
    for name in ("framework/simulation.py",):
        compile(files[name], name, "exec")
    assert "def make_cwm_env" in files["framework/simulation.py"]
    disabled = {
        x.relpath: x.content
        for x in templates(context, CwmConfig.from_mapping({"workspace_helpers_enabled": False}))
    }
    assert "world_model/model_env.py" not in disabled and "framework/simulation.py" not in disabled
    assert "Local simulation helpers" not in disabled["docs/CWM_modeling_phase.md"]


def test_preview_cleanup_cannot_follow_agent_symlinks(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "keep.txt"
    protected.write_text("keep")
    (work / "tmp").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        clear_images(work)
    (work / "tmp").unlink()
    clear_images(work)
    (work / "tmp/images/link").symlink_to(outside, target_is_directory=True)
    nested = work / "tmp/images/nested"
    nested.mkdir()
    (nested / "junk").write_text("temporary")
    clear_images(work)
    assert list((work / "tmp/images").iterdir()) == [] and protected.read_text() == "keep"
    (work / "tmp/images/obs_id_1.png").symlink_to(protected)
    with pytest.raises(OSError):
        save_preview(work, 1, b"wrong")
    assert protected.read_text() == "keep"


async def submit(c):
    value = await CwmTool("SubmitExplorationController", c).call(
        {}, ToolContext(cwd=str(c.workdir))
    )
    return json.loads(value.data[value.data.index("{") :]), value.data


@pytest.mark.integration
async def test_real_exploration_ids_preview_cap_and_next_submission_clears(rig, monkeypatch):
    c, _ = rig
    accept(c)
    c.options.n_tmp_images_saved_per_exploration = 4
    monkeypatch.setattr(
        "regact.protocols.cwm.viewer.png", lambda problem, obs: bytes([obs["frame"][0]])
    )
    exploration(c, (1,) * 5)
    result, text = await submit(c)
    assert result["observation_ids"] == "[1:6]" and result["transition_ids"] == "[1:5]"
    assert result["observation_images"] == "tmp/images/obs_id_<n>.png for <n> in [1:2, 5:6]"
    images = c.workdir / "tmp/images"
    assert sorted(p.name for p in images.iterdir()) == [
        "obs_id_1.png",
        "obs_id_2.png",
        "obs_id_5.png",
        "obs_id_6.png",
    ]
    assert len(c.data({"op": "observations", "ids": result["observation_ids"]})) == 6
    assert len(c.data({"op": "transitions", "ids": result["transition_ids"]})) == 5
    assert "simulation" not in text.lower() or "Code World Model" in text
    again, _ = await submit(c)
    assert again["status"] == "Refused" and "new observations" in again["message"]
    assert not list(images.iterdir()) and "observation_images" not in again


@pytest.mark.integration
async def test_partial_real_error_retains_ids_and_identifies_invalid_action(rig, monkeypatch):
    c, _ = rig
    accept(c)
    c.options.n_tmp_images_saved_per_exploration = 0
    exploration(c, (1, 1, 1, 99))
    result, _ = await submit(c)
    assert result["status"] == "Incomplete" and result["real_actions"] == 3
    assert result["observation_ids"] == "[1:4]" and result["transition_ids"] == "[1:3]"
    assert result["error"]["action"] == 99
    assert (
        "Invalid action" in result["error"]["message"]
        and "only action 1" in result["error"]["message"]
    )
    assert "observation_images" not in result and list((c.workdir / "tmp/images").iterdir()) == []


@pytest.mark.integration
def test_local_env_reset_modes_and_controller_runner(rig):
    c, _ = rig
    accept(c)
    exploration(c)
    script = """
import json
from framework import data_api
from framework.simulation import make_cwm_env, run_controller
from framework.simulation import EnvCWM
calls=[]
data_api.summary=lambda: {"initial_observation_id": 1}
data_api.load_observations=lambda ids: calls.append(ids) or [INITIAL]
env=make_cwm_env()
assert calls == [[1]] and env.state.n == 0
a=env.step(1); assert a.n == 1
object.__setattr__(a, "n", 99); assert env.state.n == 1
assert env.reset().n == 0 and calls == [[1]]
other=env.state; object.__setattr__(other, "n", 2)
assert env.reset(other).n == 2 and env.reset().n == 0
observed=make_cwm_env(obs_mode=True)
assert observed.reset()["frame"] == [0]*30
assert observed.step(1)["frame"] == [1]*30
before=len(calls)
explicit=make_cwm_env(initial_state=env.state)
assert len(calls) == before
try: EnvCWM()
except ValueError as e: assert "initial_state" in str(e)
else: raise AssertionError("implicit data access")
from exploration import get_controller
result=run_controller(get_controller(),env,max_actions=3)
assert result == {"actions":3,"stop_reason":"max_actions"}
print("HELPERS_OK")
""".replace("INITIAL", repr(c.initial))
    done = subprocess.run(
        [sys.executable, "-c", script], cwd=c.workdir, text=True, capture_output=True, timeout=15
    )
    assert done.returncode == 0, done.stderr
    assert "Action 3:" in done.stdout and "HELPERS_OK" in done.stdout


@pytest.mark.integration
def test_pure_model_env_is_usable_inside_isolated_controller(rig):
    c, _ = rig
    accept(c)
    c.options.n_tmp_images_saved_per_exploration = 0
    (
        c.workdir / "exploration.py"
    ).write_text('''"""Inspect one simulated successor before acting."""
from framework.simulation import EnvCWM
class ExplorationController:
 def is_done(self,s): return s.n >= 4
 def act(self,s):
  simulation=EnvCWM(initial_state=s)
  assert simulation.step(1).n == s.n+1
  return 1
def get_controller(): return ExplorationController()
''')
    result = c.tool("SubmitExplorationController", {})
    assert result["real_actions"] == 4 and not result.get("error"), result


@pytest.mark.integration
def test_direct_environment_denial_is_flagged_even_when_caller_catches_it(rig):
    c, server = rig
    c.context = SimpleNamespace(experiment=SimpleNamespace(flagged_tool_calls=0), logger=Mock())
    with TestClient(server.app) as http:
        denied = http.post("/env/counter/step", json={"action": 1})
    assert denied.status_code == 409 and c.context.experiment.flagged_tool_calls == 1
    logged = c.store.db.execute(
        "SELECT payload FROM events WHERE kind='flagged_tool_call'"
    ).fetchone()
    assert json.loads(logged[0])["operation"] == "step"
    call = ToolCallView(id="denied", name="Bash", input={}, result=denied.text, is_error=False)
    _tag_tool_calls([TurnView(items=[TurnItem(kind="tool", tool=call)])], [])
    assert call.tag == "cheat" and call.flags
    c.context = None
