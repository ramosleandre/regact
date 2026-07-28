"""The canonical, provider-independent transcript writer.

The loop writes every normalized :class:`AgentEvent` here, one JSON object per
line, so the visualizer reads the same ``transcript.jsonl`` whether the backend
was Alan or Claude. This is the only place the event union is serialized for the
agent stream (structured ops logs go to :class:`RunLogger` instead).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import IO, Any

from regact.agent.events import (
    AgentError,
    AgentEvent,
    SystemPrompt,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnComplete,
    UserMessage,
)
from regact.obs.errors import ErrorCategory


class TranscriptWriter:
    """Append normalized agent events to ``transcript.jsonl``."""

    def __init__(self, path: str) -> None:
        # The writer owns this handle for its lifetime; close() / __exit__ release it.
        self._handle: IO[str] = open(path, "w", encoding="utf-8")  # noqa: SIM115

    def write(self, event: AgentEvent) -> None:
        self._handle.write(json.dumps(event_to_json(event)) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> TranscriptWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def event_to_json(event: AgentEvent) -> dict[str, object]:
    """Serialize one event, tagged by its type; enums rendered as their value."""
    payload = asdict(event)
    if isinstance(event, AgentError):
        payload["category"] = event.category.value
    return {"type": type(event).__name__, **payload}


_EVENT_TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        TextDelta,
        ThinkingDelta,
        ToolCall,
        ToolResult,
        TurnComplete,
        AgentError,
        SystemPrompt,
        UserMessage,
    )
}


def event_from_json(obj: dict[str, Any]) -> AgentEvent | None:
    """Rebuild an event from :func:`event_to_json` output; ``None`` if it is not one.

    Unknown tags and malformed payloads return ``None`` rather than raising, so a reader
    consuming a foreign or newer stream skips what it does not understand.
    """
    cls = _EVENT_TYPES.get(str(obj.get("type", "")))
    if cls is None:
        return None
    fields = {k: v for k, v in obj.items() if k != "type"}
    if cls is AgentError and "category" in fields:
        try:
            fields["category"] = ErrorCategory(fields["category"])
        except ValueError:
            return None
    try:
        return cls(**fields)  # type: ignore[no-any-return]
    except TypeError:  # missing/extra keys for this event type
        return None
