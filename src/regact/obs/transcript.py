"""The canonical, provider-independent transcript writer.

The loop writes every normalized :class:`AgentEvent` here, one JSON object per
line, so the visualizer reads the same ``transcript.jsonl`` whether the backend
was Alan or Claude. This is the only place the event union is serialized for the
agent stream (structured ops logs go to :class:`RunLogger` instead).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import IO

from regact.agent.events import AgentError, AgentEvent


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
