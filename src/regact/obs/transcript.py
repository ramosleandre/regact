"""The canonical, provider-independent transcript writer.

The loop writes every normalized :class:`AgentEvent` here, one JSON object per
line, so the visualizer reads the same ``transcript.jsonl`` whether the backend
was Alan or Claude. This is the only place the event union is serialized for the
agent stream (structured ops logs go to :class:`RunLogger` instead).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import IO, Any

from regact.agent.events import (
    AgentError,
    AgentEvent,
    IterationComplete,
    SystemPrompt,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    UserMessage,
)
from regact.obs.errors import ErrorCategory

# Written by TranscriptWriter, not part of any event; readers must drop it before rebuilding.
_TS_KEY = "ts"


class TranscriptWriter:
    """Append normalized agent events to ``transcript.jsonl``."""

    def __init__(self, path: str) -> None:
        # The writer owns this handle for its lifetime; close() / __exit__ release it.
        self._handle: IO[str] = open(path, "w", encoding="utf-8")  # noqa: SIM115

    def write(self, event: AgentEvent) -> None:
        # Stamped at write time, so the stream carries WHEN as well as what. Without it the only
        # timing signal is a file mtime, which dates the last write and nothing else - three
        # separate questions about a slow serve (was a run degraded from the start, how long did
        # one generation take, did a retry escalate) were unanswerable for exactly that reason.
        payload = dict(event_to_json(event))
        payload[_TS_KEY] = datetime.now(UTC).isoformat()
        self._handle.write(json.dumps(payload) + "\n")
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
        IterationComplete,
        AgentError,
        SystemPrompt,
        UserMessage,
    )
}
# Back-compat: pre-rename transcripts (bench 01) tagged the per-completion event "TurnComplete".
_EVENT_TYPES["TurnComplete"] = IterationComplete


def event_from_json(obj: dict[str, Any]) -> AgentEvent | None:
    """Rebuild an event from :func:`event_to_json` output; ``None`` if it is not one.

    Unknown tags and malformed payloads return ``None`` rather than raising, so a reader
    consuming a foreign or newer stream skips what it does not understand.
    """
    cls = _EVENT_TYPES.get(str(obj.get("type", "")))
    if cls is None:
        return None
    fields = {k: v for k, v in obj.items() if k not in ("type", _TS_KEY)}
    if cls is AgentError and "category" in fields:
        try:
            fields["category"] = ErrorCategory(fields["category"])
        except ValueError:
            return None
    try:
        return cls(**fields)  # type: ignore[no-any-return]
    except TypeError:  # missing/extra keys for this event type
        return None
