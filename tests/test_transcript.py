"""Unit tests: TranscriptWriter + event serialization (transcript.jsonl)."""

import json
from pathlib import Path

from regact.agent.events import AgentError, IterationComplete, ToolCall
from regact.obs.errors import ErrorCategory
from regact.obs.transcript import TranscriptWriter, event_from_json, event_to_json


def test_event_to_json_tool_call() -> None:
    payload = event_to_json(ToolCall("c1", "SubmitSolution", {"x": 1}))
    assert payload == {"type": "ToolCall", "id": "c1", "name": "SubmitSolution", "input": {"x": 1}}


def test_event_to_json_agent_error_renders_enum_value() -> None:
    payload = event_to_json(AgentError(ErrorCategory.AGENT_API, "429"))
    assert payload["type"] == "AgentError"
    assert payload["category"] == "agent_api"
    assert payload["message"] == "429"


def test_transcript_writes_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    with TranscriptWriter(str(path)) as writer:
        writer.write(ToolCall("c1", "SubmitSolution", {}))
        writer.write(IterationComplete("done"))
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["type"] == "ToolCall"
    assert json.loads(lines[1])["type"] == "IterationComplete"


def test_legacy_turn_complete_tag_reads_as_iteration_complete() -> None:
    """Pre-rename transcripts (bench 01) tagged this event 'TurnComplete'; readers still load it."""
    event = event_from_json({"type": "TurnComplete", "final_text": "done", "usage": None})
    assert event == IterationComplete("done")
