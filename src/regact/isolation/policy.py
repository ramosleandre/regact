"""The anti-cheat policy: what agent code must not do.

A single declarative policy, owned by the harness and translated by each adapter
to its backend's native enforcement. Forbidden imports/calls catch attempts to
reach the game object or escape the HTTP boundary from the submitted controller;
forbidden path substrings catch attempts to read the game data on disk.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SecurityPolicy:
    """What the agent's submitted code and tool calls are not allowed to do."""

    forbidden_imports: frozenset[str] = field(default_factory=frozenset)
    forbidden_calls: frozenset[str] = field(default_factory=frozenset)
    forbidden_path_substrings: frozenset[str] = field(default_factory=frozenset)


# Game libraries + introspection/escape hatches: a controller is a pure policy and
# needs none of these. Reaching the game lib or the on-disk game data is cheating.
_GAME_MODULES = ("arc_agi", "arcengine", "gymnasium", "minigrid")
_ESCAPE_MODULES = ("inspect", "importlib", "ctypes")  # extend as new vectors appear


def default_policy(*, extra_imports: Iterable[str] = ()) -> SecurityPolicy:
    """The policy applied to every run."""
    return SecurityPolicy(
        forbidden_imports=frozenset({*_GAME_MODULES, *_ESCAPE_MODULES, *extra_imports}),
        forbidden_calls=frozenset(
            {
                "eval",
                "exec",
                "compile",
                "__import__",
                "inspect.getsource",
                "importlib.import_module",
            }
        ),
        forbidden_path_substrings=frozenset({"environnement", "environment_files", ".."}),
    )
