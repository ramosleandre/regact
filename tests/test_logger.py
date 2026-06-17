"""Unit tests: RunLogger (events.jsonl + output.log)."""

import json
from pathlib import Path

from regact.obs.errors import ErrorCategory, LogComponent
from regact.obs.logger import RunLogger


def test_logger_writes_structured_and_human(tmp_path: Path) -> None:
    with RunLogger(str(tmp_path), task="g") as logger:
        logger.log(LogComponent.ORCHESTRATOR, "INFO", "session_start", phase="bootstrap", foo="bar")
        logger.log(
            LogComponent.AGENT,
            "ERROR",
            "agent_error",
            error_category=ErrorCategory.AGENT_API,
            message="429",
        )

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(events) == 2
    assert events[0]["component"] == "orchestrator"
    assert events[0]["task"] == "g"
    assert events[0]["detail"] == {"foo": "bar"}
    assert events[1]["error_category"] == "agent_api"

    human = (tmp_path / "output.log").read_text()
    assert "session_start" in human
    assert "agent_api" in human
