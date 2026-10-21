"""Shared by the grid-game build scripts: a cache of searched levels and the dealing into packs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

CACHE_DIR = Path.home() / ".cache" / "regact" / "grid_game_solutions"


class SolutionCache:
    """Search results per level, kept across builds, so a rebuild only searches new levels.

    A level's entry holds its plan (or None) and the budget it was searched with; an unsolved
    entry is reused only when the new budget is not larger.
    """

    def __init__(self, game: str) -> None:
        self._path = CACHE_DIR / f"{game}.json"
        self._entries: dict[str, Any] = (
            json.loads(self._path.read_text()) if self._path.is_file() else {}
        )

    def solve(
        self, key: str, budget: float, search: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        """``search()`` returns ``{"plan": list | None, "exhausted": bool}``."""
        entry = self._entries.get(key)
        if entry is None or (
            entry["plan"] is None and not entry["exhausted"] and entry["budget"] < budget
        ):
            entry = {**search(), "budget": budget}
            self._entries[key] = entry
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._entries))
        return entry


def deal(levels: list[dict[str, Any]], n_packs: int) -> list[list[dict[str, Any]]]:
    """Packs of equal difficulty spread: levels sorted by optimal length are dealt in snake
    order (1..n, then n..1, ...), so no pack is systematically easier, and each pack is ordered by
    length (its easy-to-hard ramp)."""
    ranked = sorted(levels, key=lambda level: level["optimal"])
    packs: list[list[dict[str, Any]]] = [[] for _ in range(n_packs)]
    for i, level in enumerate(ranked):
        row, column = divmod(i, n_packs)
        packs[column if row % 2 == 0 else n_packs - 1 - column].append(level)
    return packs
