"""Real experiments need a coherent CWM, not predicted novelty or a trial episode."""

import pytest

from test_cwm_protocol import collect, exploration, model
from test_cwm_protocol import rig as rig

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("mutation", ["edit", "delete", "syntax", "symlink", "dependency", "shadow"])
def test_stale_model_and_dependency_changes_do_not_start_an_episode(rig, mutation):
    c, _ = rig
    collect(c)
    model(c.workdir)
    dependency = c.workdir / "cwm_rules.py"
    dependency.write_text("STEP = 1\n")
    source = c.workdir / "world_model/model_transition.py"
    source.write_text("import cwm_rules\nimport fractions\n" + source.read_text())
    assert c.tool("UpdateCodeWorldModel", {})["accepted"]
    if mutation == "edit":
        source.write_text(source.read_text() + "# changed\n")
    elif mutation == "delete":
        source.unlink()
    elif mutation == "syntax":
        source.write_text("def step(\n")
    elif mutation == "symlink":
        target = c.workdir / "alternate.py"
        target.write_text(source.read_text())
        source.unlink()
        source.symlink_to(target)
    elif mutation == "dependency":
        dependency.write_text("STEP = 2\n")
    else:
        (c.workdir / "fractions.py").write_text("# shadows the original library\n")
    exploration(c)
    before = c.store.summary()
    result = c.tool("RunController", {})
    assert result["error_type"] == "cwm_source_changed", result
    assert c.store.summary() == before
    assert c.terminal is None and c.phase == "Active Exploration"


def test_controller_and_debug_script_edits_do_not_require_revalidation(rig):
    c, _ = rig
    collect(c)
    model(c.workdir)
    assert c.tool("UpdateCodeWorldModel", {})["accepted"]
    exploration(c, (1,))
    (c.workdir / "simulate.py").write_text("raise RuntimeError('local debug only')\n")
    (c.workdir / "goal.py").write_text("# unrelated edit\n")
    result = c.tool("RunController", {})
    assert result["real_actions"] == 1 and not result.get("error"), result


def test_prediction_exception_stops_only_its_action_after_prior_real_steps(rig):
    c, _ = rig
    collect(c)
    model(c.workdir)
    (c.workdir / "world_model/model_transition.py").write_text(
        "from world_model.model_state import State\ndef step(s,a):\n"
        " if s.n == 3: raise ValueError('unimplemented transition')\n"
        " return State(s.n + 1)\n"
    )
    assert c.tool("UpdateCodeWorldModel", {})["accepted"]
    exploration(c, (1,) * 5)
    before = c.store.summary()["n_total_transitions"]
    result = c.tool("RunController", {})
    assert result["real_actions"] == 3, result
    assert c.store.summary()["n_total_transitions"] == before + 3
    assert result["stop_reason"] == "model_error"
    assert result["error_context"]["callback"] == "step"
    assert result["observation_ids"] == [1, 2, 3, 4]
    assert c.phase == "CWM Modeling"


def test_predicted_familiar_outcome_can_reveal_a_contradiction(rig, monkeypatch):
    c, _ = rig
    collect(c)
    model(c.workdir)
    (c.workdir / "world_model/model_transition.py").write_text(
        "from world_model.model_state import State\n"
        "def step(s,a): return State(s.n + (1 if a == 1 else 0))\n"
    )
    assert c.tool("UpdateCodeWorldModel", {})["accepted"]
    native_type = type(c.env.live._native)
    original = native_type.step
    monkeypatch.setattr(native_type, "step", lambda self, action: original(self, 1))
    exploration(c, (2,))
    result = c.tool("RunController", {})
    assert result["real_actions"] == 1
    assert result["actual_novel_observations"] == 0
    assert result["stop_reason"] == "prediction_mismatch"
    diagnostic = c.store.get_diagnostic(result["diagnostic_id"])
    assert diagnostic["predicted"]["frame"][0] == 0
    assert diagnostic["observed"]["frame"][0] == 1
    assert c.phase == "CWM Modeling"
