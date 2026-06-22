"""Flag (never block) cheat attempts, for logging and metrics.

Two complementary, non-enforcing signals — both just read the tool-call stream the run
already receives (no process inspection), and neither blocks:

* ``flag_tool_call`` — a keyword scan of a tool call's *arguments* for a forbidden path
  or module. Precise intent detection for the on-disk game data / escape modules.
* ``flag_os_denial`` — an *egress* denial in a tool *result*: a DNS failure or the
  proxy's 403 means the agent tried to reach an external host (the curls), which a
  keyword list cannot enumerate. We use only the network signatures: a sandbox's file
  denials are indistinguishable from benign friction (``ps``/``kill``/temp files all
  report "operation not permitted"), so the keyword scan owns the file/module side.
"""

from __future__ import annotations

from typing import Any

from regact.security.policy import SecurityPolicy

# Unambiguous egress denials: reaching an external host (no loopback/env call needs DNS,
# and a 403 here comes only from our allow-list proxy). Deliberately network-only.
_DENIAL_SIGNATURES = (
    "could not resolve host",  # DNS blocked — tried to reach an external host
    "nodename nor servname",  # same, macOS wording
    "403 forbidden",  # the egress proxy refused a non-allow-listed host
)


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


def flag_os_denial(output: str) -> bool:
    """True if a tool result reads like the agent was blocked from reaching an external host.

    The denial is the evidence — no guessing intent from the command: an external host
    was unreachable (DNS blocked) or refused by the egress allow-list proxy (403).
    """
    low = output.lower()
    return any(signature in low for signature in _DENIAL_SIGNATURES)


def _flatten(value: Any) -> str:
    """Concatenate all string content in a nested args structure."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(v) for v in value)
    return ""
