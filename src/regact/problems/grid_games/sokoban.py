"""Sokoban: packs of DeepMind Boxoban levels (Apache-2.0), played by a small in-house engine.

Each task is a pack of levels of increasing difficulty; the reference action count of a level
is its move-optimal solution, computed offline by ``scripts/grid_games/build_sokoban_packs.py``
and stored with the levels in ``data/sokoban_packs.json``. The agent sees colour ids only,
never the Sokoban symbols.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from regact.problems.base import register_problem
from regact.problems.grid_games.base import GridGame, GridGameProblem, GridTask

_PACKS = Path(__file__).parent / "data" / "sokoban_packs.json"

WALL, FLOOR, GOAL, BOX, BOX_ON_GOAL, PLAYER, PLAYER_ON_GOAL = 5, 0, 14, 12, 9, 8, 6
MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}  # up, down, left, right


def parse_level(rows: tuple[str, ...]) -> dict[str, Any]:
    """Walls, goals, boxes and player of a level written in the standard symbols
    (``#`` wall, ``.`` goal, ``$`` box, ``*`` box on goal, ``@`` player, ``+`` player on goal)."""
    walls, goals, boxes, player = set(), set(), set(), None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((r, c))
            if ch in ".+*":
                goals.add((r, c))
            if ch in "$*":
                boxes.add((r, c))
            if ch in "@+":
                player = (r, c)
    if player is None or not boxes or len(boxes) != len(goals):
        raise ValueError("a level needs one player and as many boxes as goals")
    return {
        "height": len(rows),
        "width": max(map(len, rows)),
        "walls": frozenset(walls),
        "goals": frozenset(goals),
        "boxes": frozenset(boxes),
        "player": player,
    }


class SokobanGame(GridGame):
    def __init__(self, levels: tuple[tuple[str, ...], ...]) -> None:
        self._levels = levels

    def load_level(self, index: int) -> None:
        self._level = parse_level(self._levels[index])
        self._boxes = set(self._level["boxes"])
        self._player = self._level["player"]

    def apply(self, action: int) -> None:
        dr, dc = MOVES[action]
        ahead = (self._player[0] + dr, self._player[1] + dc)
        if ahead in self._level["walls"]:
            return
        if ahead in self._boxes:
            beyond = (ahead[0] + dr, ahead[1] + dc)
            if beyond in self._level["walls"] or beyond in self._boxes:
                return
            self._boxes.remove(ahead)
            self._boxes.add(beyond)
        self._player = ahead

    def solved(self) -> bool:
        return bool(self._boxes == self._level["goals"])

    def grid(self) -> list[list[int]]:
        level = self._level
        rows = [[FLOOR] * level["width"] for _ in range(level["height"])]
        for r, c in level["walls"]:
            rows[r][c] = WALL
        for r, c in level["goals"]:
            rows[r][c] = GOAL
        for r, c in self._boxes:
            rows[r][c] = BOX_ON_GOAL if (r, c) in level["goals"] else BOX
        r, c = self._player
        rows[r][c] = PLAYER_ON_GOAL if (r, c) in level["goals"] else PLAYER
        return rows


@cache
def _load_tasks() -> dict[str, GridTask]:
    packs = json.loads(_PACKS.read_text(encoding="utf-8"))["packs"]
    return {
        pack["key"]: GridTask(
            key=pack["key"],
            levels=tuple(tuple(level["rows"]) for level in pack["levels"]),
            reference_actions=tuple(level["optimal"] for level in pack["levels"]),
        )
        for pack in packs
    }


class SokobanProblem(GridGameProblem):
    name = "sokoban"

    def tasks(self) -> dict[str, GridTask]:
        return _load_tasks()

    def make_game(self, task: GridTask) -> GridGame:
        return SokobanGame(task.levels)


register_problem("sokoban", lambda kwargs: SokobanProblem(**kwargs))
