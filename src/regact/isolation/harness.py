"""The anti-cheat harness: one policy, two enforcement surfaces.

The harness owns the :class:`SecurityPolicy` and the framework-side checks that
are backend-independent:
  - ``scan_file`` — static AST scan of agent code (run before ``solution.py`` is
    executed), to reject a controller that imports the game lib or an escape hatch.
  - ``guard_tool_call`` — a string scan of a tool call's arguments, flagging the
    obvious attempts to reach the game lib or the on-disk game data.

Each adapter then translates the same policy to its backend's *native* confinement
(the strongest layer): :func:`claude_deny_settings` builds the Claude Code
``settings.json`` deny-list (file tools confined to the workdir); Alan uses a
PreToolUse hook. Behavioural shadow-replay is a controller-feature concern (it
needs to run a controller), not here, to keep this module agnostic.
"""

from __future__ import annotations

from typing import Any

from regact.isolation.policy import SecurityPolicy, default_policy
from regact.isolation.scan import scan_file
from regact.isolation.tool_guard import guard_tool_call


class AntiCheatHarness:
    """Owns the policy + the backend-independent checks."""

    def __init__(self, policy: SecurityPolicy | None = None) -> None:
        self.policy = policy or default_policy()

    def scan_file(self, path: str) -> list[str]:
        """Static violations in a ``.py`` file (empty == clean)."""
        return scan_file(path, self.policy)

    def guard_tool_call(self, name: str, args: dict[str, Any]) -> list[str]:
        """Violations in a tool call's textual arguments (empty == clean)."""
        return guard_tool_call(name, args, self.policy)


def claude_deny_settings(workdir: str, policy: SecurityPolicy | None = None) -> dict[str, Any]:
    """Build a Claude Code ``settings.json`` that denies reading the game data.

    Denies any path whose segments include a forbidden substring (the on-disk
    games), wherever it lives — using ``**/<name>/**`` so it matches regardless of
    the absolute location. Crucially it does NOT blanket-deny reads: the agent must
    stay free to read and edit its own workdir, or it cannot work. The HTTP boundary
    (no game object) and the submit-time AST scan are the other two layers.
    """
    policy = policy or default_policy()
    deny: list[str] = []
    for sub in sorted(policy.forbidden_path_substrings):
        if sub == "..":
            continue
        deny.append(f"Read(**/{sub}/**)")
    return {"permissions": {"deny": deny}}
