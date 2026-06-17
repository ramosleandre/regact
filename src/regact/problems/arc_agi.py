"""ARC-AGI-3 problem (deferred to Block 8b).

The real port wraps the ``arc-agi`` arcade and loads the 25 games from a local
``environnement/`` directory in offline mode (downloaded server-side, never
reachable by the agent — it sees only the HTTP env client). Wiring the actual
``arcade.make(...)`` API and the local data layout waits until the download
script lands, so we only register the name here; building it raises a clear error.
Prompt text already lives in ``problems/prompts/arc_agi.md``.
"""

from __future__ import annotations

from typing import Any

from regact.problems.base import BaseProblem, register_problem


def _build(kwargs: dict[str, Any]) -> BaseProblem:
    raise NotImplementedError(
        "the ARC-AGI-3 problem arrives in Block 8b "
        "(needs the local game data in environnement/ and the arc-agi library)"
    )


# Registered so `build_problem('arc_agi')` fails loudly rather than 'unknown problem'.
register_problem("arc_agi", _build)
