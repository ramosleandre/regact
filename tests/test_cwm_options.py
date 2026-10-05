"""Optional planner and explicit pagination contracts at the actual tool/data boundary."""

import inspect
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

from regact.features.base import FeatureContext
from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.feedback import present
from regact.protocols.cwm.protocol import CwmProtocol
from regact.protocols.cwm.session import CwmSession
from regact.protocols.cwm.templates import templates
from test_cwm_protocol import accept, collect, exploration, model
from test_cwm_protocol import rig as rig


def test_planner_defaults_and_validation():
    assert CwmConfig.from_mapping({}).planner.enabled is False
    for bad in (None, 0, 1, "false"):
        with pytest.raises(ValueError, match="boolean"):
            CwmConfig.from_mapping({"planner": {"enabled": bad}})
    # An unused planner depth must not constrain ordinary explorations.
    CwmConfig.from_mapping({"max_actions_per_RunController": 1})
    with pytest.raises(ValueError, match="depth"):
        CwmConfig.from_mapping({"max_actions_per_RunController": 1, "planner": {"enabled": True}})


@pytest.mark.integration
@pytest.mark.parametrize("rig", [False, True], indirect=True)
def test_planner_registration_refusal_and_shared_notices(rig):
    c, server = rig
    enabled = c.options.planner.enabled
    protocol = CwmProtocol(c.config)
    protocol.coordinator = c
    session = protocol.bind(None)
    names = [tool.name for tool in session.tools]
    assert ("PlanInCWM" in names) == enabled
    server.bind_control("counter", session.tools, cwd=str(c.workdir))
    if not enabled:
        before = c.store.summary()
        with TestClient(server.app) as http:
            response = http.post("/control/counter/tool", json={"name": "PlanInCWM", "input": {}})
            assert response.status_code == 404
        assert c.tool("PlanInCWM", {})["error_type"] == "command_unavailable"
        assert c.store.summary() == before
        assert not (c.workdir / "goal.py").exists()
    collect(c)
    model(c.workdir)
    accepted = c.tool("UpdateCodeWorldModel", {})
    change = accepted["phase_change"]
    assert (change["from"], change["to"]) == ("CWM Modeling", "Active Exploration")
    notice = change["next_step"]
    assert present("UpdateCodeWorldModel", accepted, c.options, c.config.limits)["phase_change"] == change
    reminder = CwmSession(coordinator=c, tools=[]).reminder(1)
    assert notice == c.phase_description()
    assert reminder.endswith(c.phase_description())
    assert ("PlanInCWM" in notice) == enabled
    assert ("PlanInCWM" in reminder) == enabled
    exploration(c)
    result = c.tool("RunController", {})
    assert result["real_actions"] == 4 and not result.get("error"), result


@pytest.mark.integration
@pytest.mark.parametrize("cap", [2, 100, None])
def test_generated_data_api_hides_the_query_cap(rig, cap):
    c, _ = rig
    c.options.data_api.max_items = cap
    c.options.n_unique_observations_in_initial_collection = 4
    collect(c)
    ctx = FeatureContext("fake", "counter", str(c.workdir))
    source = next(
        f.content for f in templates(ctx, c.options) if f.relpath == "framework/data_api.py"
    )
    api = ModuleType("data_api")
    exec(compile(source, "data_api.py", "exec"), api.__dict__)
    api._query = lambda op, **args: c.data({"op": op, **args})
    # The caller never pages: every ID comes back, and loads of any size are batched under the cap.
    assert api.list_observation_ids() == [1, 2, 3, 4]
    assert api.list_transition_ids() == [1, 2, 3]
    assert [o["frame"][0] for o in api.load_observations([4, 1, 2, 3])] == [3, 0, 1, 2]
    assert api.load_observations(2) == api.load_observations("[2]")
    transition = api.load_transitions("[1:3]")[2]
    assert transition["observation_id"] == 3 and transition["next_observation_id"] == 4
    assert transition["next_observation"]["frame"][0] == 3 and transition["action"] == 1
    [episode] = api.list_episodes()
    assert episode["live"] and episode["n_steps"] == 3
    observations, actions = api.load_history(episode["episode_id"])
    assert [o["frame"][0] for o in observations] == [0, 1, 2, 3] and actions == [1, 1, 1]
    observations, actions = api.load_history(episode["episode_id"], step=1)
    assert len(observations) == 2 and actions == [1]
