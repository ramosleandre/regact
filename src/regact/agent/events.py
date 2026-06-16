"""Normalized agent event stream.

Every adapter translates its backend's native output into this explicit union.
The orchestration loop consumes only these events, never backend message
classes — that is what keeps the loop provider-independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from regact.obs.errors import ErrorCategory


@dataclass
class TextDelta:
    """A chunk of assistant-visible text."""

    text: str


@dataclass
class ThinkingDelta:
    """A chunk of reasoning/thinking text."""

    text: str


@dataclass
class ToolCall:
    """The agent invoked a tool."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """The result handed back for a tool call."""

    id: str
    output: str
    is_error: bool = False


@dataclass
class TurnComplete:
    """The agent finished one turn."""

    final_text: str = ""
    usage: dict[str, Any] | None = None


@dataclass
class AgentError:
    """A backend/LLM error, normalized to a category."""

    category: ErrorCategory
    message: str


AgentEvent = TextDelta | ThinkingDelta | ToolCall | ToolResult | TurnComplete | AgentError
