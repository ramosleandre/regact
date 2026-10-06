"""Contracts found in the walkthrough: usable feedback, safe phase changes, real evidence."""

import base64
import json

import pytest
from fastapi.testclient import TestClient

from regact.agent.events import ToolCall, ToolResult, tool_result_images
from regact.obs.transcript import TranscriptWriter, event_to_json
from regact.protocols.cwm.session import CwmSession, CwmTool
from regact.tools.base import ToolContext
from regact.viz.reader import _group_turns
from test_cwm_protocol import accept, collect, exploration, model
from test_cwm_protocol import rig as rig

pytestmark = pytest.mark.integration


async def call(c, name, token=None):
    context = ToolContext(cwd=str(c.workdir), detail={"request_id": token} if token else {})
    output = await CwmTool(name, c).call({}, context)
    json_text = output.data[output.data.index("{") :]
    value = json.loads(json_text)
    assert json_text == json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    return value, value.get("phase_change")


async def test_acceptance_notice_and_retry_are_separate_and_not_duplicated(rig):
    c, _ = rig
    collect(c)
    model(c.workdir)
    result, change = await call(c, "UpdateCodeWorldModel", "stable-request")
    assert result["status"] == "Accepted"
    assert (change["from"], change["to"]) == ("CWM Modeling", "Active Exploration")
    assert (
        not {
            "phase",
            "exit_reason",
            "dataset_version",
            "bundle",
            "request_id",
            "accepted",
            "complete",
        }
        & result.keys()
    )
    before = c.store.db.execute("select count(*) from records").fetchone()[0]
    replay, _ = await call(c, "UpdateCodeWorldModel", "stable-request")
    assert replay == result  # a replay returns the stored result, phase change included
    assert c.store.db.execute("select count(*) from records").fetchone()[0] == before
    (c.workdir / "world_model/model_transition.py").write_text(
        "from world_model.model_state import State\ndef step(s,a): return State(s.n+2)\n"
    )
    refused, change = await call(c, "UpdateCodeWorldModel")
    assert refused["status"] == "Refused"
    assert refused["previously_accepted_cwm_version"] == result["cwm_version"]
    assert change is None
    assert c.phase == "Active Exploration"


async def test_http_notice_and_retry_id_are_transport_metadata(rig):
    c, server = rig
    collect(c)
    model(c.workdir)
    server.bind_control("counter", [CwmTool("UpdateCodeWorldModel", c)], cwd=str(c.workdir))
    with TestClient(server.app) as http:
        body = {"name": "UpdateCodeWorldModel", "input": {}}
        first = http.post(
            "/control/counter/tool", json=body, headers={"X-Regact-Request-ID": "repeat"}
        ).json()
        second = http.post(
            "/control/counter/tool", json=body, headers={"X-Regact-Request-ID": "repeat"}
        ).json()
    output = json.loads(first["output"])  # one JSON document, nothing printed after it
    assert output["status"] == "Accepted" and "messages" not in first
    assert first["output"].startswith('{\n  "status": "Accepted",')
    assert output["phase_change"]["next_step"] == c.phase_description()
    session = CwmSession(coordinator=c, tools=[])
    for reminder_count in (0, 1, 10):
        assert session.reminder(reminder_count) == (
            "Continue your work until the game is fully solved.\n"
            f"Current phase is {c.phase}: {c.phase_description()}"
        )
    assert "messages" not in second and json.loads(second["output"]) == json.loads(first["output"])
    assert CwmTool("PlanInCWM", c).input_schema["properties"] == {}
    rejected = await CwmTool("PlanInCWM", c).call(
        {"request_id": "public"}, ToolContext(cwd=str(c.workdir))
    )
    assert rejected.is_error and "no arguments" in rejected.data
    assert rejected.data.startswith('{\n  "status": "Refused",')


def test_generic_summary_and_image_sources(rig, monkeypatch):
    c, _ = rig
    collect(c)
    summary = c.data({"op": "summary"})
    assert summary["phase"] == "CWM Modeling"
    assert summary["n_started_episodes"] == 1
    assert summary["n_total_observations"] == 3
    assert not {"initial_collection", "dataset_version", "n_episodes"} & summary.keys()
    assert c.initial["info"]["milestones"] == []
    assert c.initial["reward"] == 0.0
    seen = []

    def png(_problem, obs):
        seen.append(obs)
        return b"fake PNG for source-routing test"

    monkeypatch.setattr("regact.protocols.cwm.viewer.png", png)
    did = c.store.diagnostic(
        {"kind": "comparison", "predicted": c.store.observation(2), "observed": c.initial}
    )
    c.data({"op": "image", "diagnostic_id": did, "which": None})
    assert seen[-1] == c.initial
    c.data({"op": "image", "transition_id": 1, "which": None})
    assert seen[-1] == c.store.observation(2)
    for args in (
        {},
        {"observation_id": 1, "transition_id": 1},
        {"observation_id": True},
        {"observation_id": 1, "which": "after"},
        {"diagnostic_id": did, "which": "after"},
    ):
        with pytest.raises(ValueError):
            c.data({"op": "image", **args})
    no_image = c.store.diagnostic({"kind": "compression_ratio"})
    with pytest.raises(ValueError, match="no observed observation"):
        c.data({"op": "image", "diagnostic_id": no_image})

    # Re-reading evidence is not a new occurrence. Resetting/repeating steps is.
    assert c.data({"op": "summary"}) == summary
    c._reset("exploration")
    c._step(1)
    repeated = c.data({"op": "summary"})
    assert repeated["n_unique_observations"] == summary["n_unique_observations"]
    assert repeated["n_unique_transitions"] == summary["n_unique_transitions"]
    assert repeated["n_total_observations"] == 5
    assert repeated["n_started_episodes"] == 2
    assert repeated["n_total_transitions"] == 3


async def test_milestone_survives_contradiction_and_is_not_announced_again(rig):
    c, _ = rig
    accept(c)
    c.env._milestone_detector = lambda env: ["checkpoint"] if env.last_obs.frame[0] >= 3 else []
    c.problem.milestone_kind = lambda _: "progress"
    exploration(c)
    result, change = await call(c, "RunController")
    assert result["stop_reason"] == "prediction_mismatch"
    assert result["new_milestones"][0]["name"] == "checkpoint"
    assert result["new_milestones"][0]["kind"] == "progress"
    assert "New real milestones" in result["message"]
    assert change["next_step"] == c.phase_description()
    assert CwmSession(coordinator=c, tools=[]).reminder(2).endswith(c.phase_description())
    render = c.workdir / "world_model/model_render.py"
    render.write_text(
        render.read_text().replace(
            '"milestones":[]', '"milestones":(["checkpoint"] if s.n>=3 else [])'
        )
    )
    accepted, change = await call(c, "UpdateCodeWorldModel")
    assert accepted["status"] == "Accepted" and change is not None
    again, change = await call(c, "RunController")
    assert again["real_actions"] == 4 and "new_milestones" not in again
    assert change is None


async def test_callback_budget_names_the_effective_limit(rig):
    c, _ = rig
    collect(c)
    model(c.workdir)
    c.options.execution.max_seconds_per_call = 0.2
    parser = c.workdir / "world_model/model_initial_state.py"
    parser.write_text("import time\ndef get_initial_state(obs):\n time.sleep(1)\n")
    result, change = await call(c, "UpdateCodeWorldModel")
    assert result["status"] == "Incomplete"
    assert result["error"]["budget"]["value"] == 0.2
    assert result["error"]["callback"] == "get_initial_state" and change is None


def test_planner_budget_counts_parse_step_and_render(rig):
    c, _ = rig
    accept(c)
    c.options.planner.max_cwm_calls_per_planner_call = 3
    (c.workdir / "goal.py").write_text('"""Reach four."""\ndef achieved(s): return s.n==4\n')
    result = c.tool("PlanInCWM", {})
    assert result["cwm_calls"] == 3 and result["n_states_searched"] == 2
    assert not result["candidate_found"]
    assert result["search_stop_reason"] == "max_cwm_calls_per_planner_call"


def test_image_transcript_preserves_bytes_deduplicated_and_text_shape(tmp_path):
    raw = b"actual backend image bytes"
    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(raw).decode(),
            },
        }
    ]
    images = tool_result_images(content)
    log = tmp_path / "transcript.jsonl"
    with TranscriptWriter(str(log)) as transcript:
        transcript.write(ToolCall("read", "Read", {"file_path": "mutable.png"}))
        transcript.write(ToolResult("read", "viewed", images=images))
        transcript.write(ToolResult("again", "viewed", images=images))
    events = [json.loads(line) for line in log.read_text().splitlines()]
    attachment = events[1]["images"][0]
    assert "data" not in attachment
    assert (tmp_path / "media" / attachment["filename"]).read_bytes() == raw
    assert len(list((tmp_path / "media").iterdir())) == 1
    assert "images" not in event_to_json(ToolResult("plain", "text"))
    # Image delivery is grounded in image blocks, never paths mentioned in text.
    assert tool_result_images("mutable.png") == []
    turns = _group_turns(events)
    assert turns[0].tools[0].images[0]["sha256"] == attachment["sha256"]


async def test_limit_between_prediction_and_real_step_preserves_complete_history(rig, monkeypatch):
    import time

    c, _ = rig
    accept(c)
    exploration(c)
    original = c._step

    def deadline_arrives(action):
        c.deadline = time.monotonic() - 1
        return original(action)

    monkeypatch.setattr(c, "_step", deadline_arrives)
    result, _ = await call(c, "RunController")
    assert result["task_stop"]["reason"] == "walltime_limit"
    assert result["real_actions"] == 0
    assert "error" not in result and "history_complete" not in result
    assert c.store.summary()["n_total_transitions"] == 2


def test_backend_image_survives_adapter_transcript_and_http(tmp_path):
    from regact.agent.claude_adapter import ClaudeAgent
    from regact.viz.app import build_app

    # Actual valid one-pixel PNG content, independent of a mutable image filename.
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
    )
    backend_result = {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "read",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(raw).decode(),
                            },
                        }
                    ],
                }
            ]
        },
    }
    event = ClaudeAgent.__new__(ClaudeAgent)._parse_events(backend_result)[0]
    task = tmp_path / "image-fixture"
    (task / "logs").mkdir(parents=True)
    (task / "logs/experiment_state.json").write_text("{}")
    with TranscriptWriter(str(task / "logs/transcript.jsonl")) as log:
        log.write(ToolCall("read", "Read", {"file_path": "original.png"}))
        log.write(event)
    import hashlib

    filename = hashlib.sha256(raw).hexdigest() + ".png"
    with TestClient(build_app(str(tmp_path))) as http:
        image = http.get(
            "/api/game/tool-image", params={"name": "image-fixture", "filename": filename}
        )
        assert image.status_code == 200 and image.content == raw
        invalid = http.get(
            "/api/game/tool-image", params={"name": "image-fixture", "filename": "../config.json"}
        )
        assert invalid.status_code == 422
    # The canonical text field does not copy a base64 blob into the conversation.
    assert base64.b64encode(raw).decode() not in event.output
