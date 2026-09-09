"""Unit tests: TranscriptWriter + event serialization (transcript.jsonl)."""

import json
from datetime import datetime
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


def test_written_events_carry_a_timestamp_and_still_round_trip(tmp_path) -> None:
    """Every timing question about a slow serve died on events carrying only {text, type}. The
    stamp is added at WRITE time - and the reader must drop it, or `cls(**fields)` raises TypeError
    and every stamped event is silently skipped."""
    import json as _json

    path = tmp_path / "transcript.jsonl"
    event = ToolCall("c1", "SubmitSolution", {"x": 1})
    with TranscriptWriter(str(path)) as writer:
        writer.write(event)

    record = _json.loads(path.read_text().strip())
    assert "ts" in record
    datetime.fromisoformat(record["ts"])  # parseable, so arithmetic needs no assumed rate
    assert event_from_json(record) == event  # the extra key must not drop the event
