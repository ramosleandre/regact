"""Normalized agent event stream.

Every adapter translates its backend's native output into this explicit union.
The orchestration loop consumes only these events, never backend message
classes — that is what keeps the loop provider-independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    images: list[dict[str, str]] = field(default_factory=list)
    executed: bool = True  # False: the agent refused to run the call; it spends no budget


def tool_result_images(content: Any) -> list[dict[str, str]]:
    """Preserve actual inline image blocks, never infer vision from a filename."""
    images = []
    if not isinstance(content, list):
        return images
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        source = block.get("source") if isinstance(block, dict) else getattr(block, "source", None)
        if kind == "image" and isinstance(source, dict) and source.get("type") == "base64":
            mime, data = source.get("media_type"), source.get("data")
            if mime in ("image/png", "image/jpeg", "image/webp", "image/gif") and isinstance(
                data, str
            ):
                images.append({"mime_type": mime, "data": data})
    return images


@dataclass
class IterationComplete:
    """The agent finished one iteration - one completion and its tool cycle."""

    final_text: str = ""
    usage: dict[str, Any] | None = None


@dataclass
class AgentError:
    """A backend/LLM error, normalized to a category."""

    category: ErrorCategory
    message: str


@dataclass
class SystemPrompt:
    """The system prompt the framework gave the agent (recorded once, for the viewer)."""

    text: str


@dataclass
class UserMessage:
    """A message the framework sent the agent (first message, keep-alive) — for the viewer."""

    text: str


AgentEvent = (
    TextDelta
    | ThinkingDelta
    | ToolCall
    | ToolResult
    | IterationComplete
    | AgentError
    | SystemPrompt
    | UserMessage
)
