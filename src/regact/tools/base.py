"""Framework tool base.

A ``Tool`` is the in-process front-end (used by Alan) for a framework action.
The same actions are reachable by any CLI agent through the workdir client CLI
(HTTP), so a ``Tool`` body typically just calls the orchestrator/EnvClient.
Concrete tools (submit/exit) live in sibling modules; this is only the contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

PermissionLevel = Literal["read", "write", "exec"]


@dataclass
class ToolContext:
    """Scoped state passed to a tool call."""

    cwd: str
    abort_signal: Any = None  # asyncio.Event-like
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolOutput:
    """Result of a tool execution."""

    data: Any
    is_error: bool = False


class Tool(ABC):
    """Abstract base for framework tools."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """OpenAI-compatible JSON Schema for the tool's inputs."""
        ...

    @abstractmethod
    async def call(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        """Execute the tool; signal failure with ``ToolOutput(is_error=True)``."""
        ...

    def permission_level(self, args: dict[str, Any]) -> PermissionLevel:
        return "write"

    def validate_input(self, args: dict[str, Any], context: ToolContext) -> str | None:
        """Return an error string for invalid input, else ``None``."""
        return None
