"""The declarative anti-cheat policy: imports, calls, and path substrings to flag.

A single declarative policy consumed by two non-enforcing layers: the AST scan
(``scan.py``, applied by a feature to its agent-authored deliverable) and the
tool-call flagger (``detection.py``, which only logs/flags). OS-level confinement is
provided by ``runtime.py``; these lists are defense-in-depth and observability.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SecurityPolicy:
    """What the agent's submitted code and tool calls are flagged for."""

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
