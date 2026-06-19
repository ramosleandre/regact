"""Flag (never block) suspicious tool calls, for logging and metrics.

A cheap string scan over the agent's tool-call arguments so the run can record what
the agent attempted (e.g. reading a forbidden path) and report an attempt count. It
is deliberately non-enforcing: confinement is the OS sandbox's job, so missing an
obfuscated attempt costs only a log entry. It reads the tool-call stream the run
already receives -- no process inspection.
"""

from __future__ import annotations

from typing import Any

from regact.security.policy import SecurityPolicy


def flag_tool_call(name: str, args: dict[str, Any], policy: SecurityPolicy) -> list[str]:
    """Return human-readable flags found in a tool call's textual arguments (empty == clean)."""
    blob = _flatten(args).lower()
    flags: list[str] = []
    for needle in policy.forbidden_path_substrings:
        if needle.lower() in blob:
            flags.append(f"tool {name!r}: references forbidden path {needle!r}")
    for module in policy.forbidden_imports:
        if f"import {module}" in blob or f"from {module}" in blob:
            flags.append(f"tool {name!r}: imports forbidden module {module!r}")
    return flags


def _flatten(value: Any) -> str:
    """Concatenate all string content in a nested args structure."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(v) for v in value)
    return ""
