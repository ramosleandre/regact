"""Guard one tool call against the policy (the regex-style safety net).

Backends enforce the workdir confinement natively (Claude deny-list +
permission-mode, Alan PreToolUse hook); this is the framework-side check that
flags what slips through — a tool whose textual arguments mention the game lib or
the on-disk game data. It is a string scan: cheap, never perfect, but it catches
the obvious attempts the loop can then block or log.
"""

from __future__ import annotations

from typing import Any

from regact.isolation.policy import SecurityPolicy


def guard_tool_call(name: str, args: dict[str, Any], policy: SecurityPolicy) -> list[str]:
    """Return violations found in a tool call's textual arguments (empty == clean)."""
    blob = _flatten(args).lower()
    violations: list[str] = []
    for needle in policy.forbidden_path_substrings:
        if needle.lower() in blob:
            violations.append(f"tool {name!r}: references forbidden path {needle!r}")
    for module in policy.forbidden_imports:
        if f"import {module}" in blob or f"from {module}" in blob:
            violations.append(f"tool {name!r}: imports forbidden module {module!r}")
    return violations


def _flatten(value: Any) -> str:
    """Concatenate all string content in a nested args structure."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(v) for v in value)
    return ""
