"""BabaIsAI: packs of BabaIsAI levels (github.com/nacloos/baba-is-ai, MIT), a simplified
Baba Is You whose rules are word blocks on the board.

A level is an engine environment id plus a seed (the layout is procedural). Each task is a pack
of levels of increasing difficulty; the reference action count of a level is its shortest
solution, found by breadth-first search on the real engine by
``scripts/grid_games/build_babaisai_packs.py`` and stored with the levels in
``data/babaisai_packs.json``. Each cell shows a code for the object or word block on top of it
(the code table is in the same file); the engine's step limit is lifted, so a level ends only
when it is won.
"""

from __future__ import annotations

import json
import warnings
from functools import cache
from pathlib import Path
from typing import Any

from regact.envclient.obs import Obs
from regact.problems.base import register_problem
from regact.problems.grid_games.base import GridGame, GridGameProblem, GridTask

_PACKS = Path(__file__).parent / "data" / "babaisai_packs.json"
# Our directions (1 up, 2 down, 3 left, 4 right) -> the engine's (1 up, 2 right, 3 down, 4 left).
ENGINE_ACTION = {1: 1, 2: 3, 3: 4, 4: 2}
_NO_STEP_LIMIT = 10**9
_UNKNOWN_CODE = 99  # a cell kind absent from the code table (never seen while building the packs)


def make_engine(env_id: str, seed: int) -> Any:
    """The engine environment of one level, reset to its layout, with no step limit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import baba

        env = baba.make(env_id)
    env.max_steps = _NO_STEP_LIMIT
    env.reset(seed=seed)
    return env


def encoded_rows(env: Any) -> list[list[tuple[int, int]]]:
    """The (type, colour-or-word) pair of each cell's top object, row by row."""
    cells = env.grid.encode()  # indexed [x][y][channel]
    width, height = cells.shape[0], cells.shape[1]
    return [
        [(int(cells[x][y][0]), int(cells[x][y][1])) for x in range(width)] for y in range(height)
    ]


class BabaGame(GridGame):
    def __init__(self, levels: tuple[tuple[str, int], ...], codes: dict[tuple[int, int], int]):
        self._levels = levels
        self._codes = codes

    def load_level(self, index: int) -> None:
        self._env = make_engine(*self._levels[index])
        self._won = False

    def apply(self, action: int) -> None:
        _, reward, done, _ = self._env.step(ENGINE_ACTION[action])
        self._won = bool(done and reward > 0)

    def solved(self) -> bool:
        return self._won

    def grid(self) -> list[list[int]]:
        rows = encoded_rows(self._env)
        return [[self._codes.get(cell, _UNKNOWN_CODE) for cell in row] for row in rows]


@cache
def _load() -> tuple[dict[str, GridTask], dict[tuple[int, int], int], tuple[dict[str, Any], ...]]:
    data = json.loads(_PACKS.read_text(encoding="utf-8"))
    kinds = tuple(data["codes"])
    codes = {(kind["type"], kind["value"]): code for code, kind in enumerate(kinds)}
    tasks = {
        pack["key"]: GridTask(
            key=pack["key"],
            levels=tuple((level["env_id"], level["seed"]) for level in pack["levels"]),
            reference_actions=tuple(level["optimal"] for level in pack["levels"]),
        )
        for pack in data["packs"]
    }
    return tasks, codes, kinds


class BabaIsAIProblem(GridGameProblem):
    name = "babaisai"
    frame_note = (
        "each cell is an integer, one per kind of object or block (the same kind always has the "
        "same integer)"
    )

    def tasks(self) -> dict[str, GridTask]:
        return _load()[0]

    def make_game(self, task: GridTask) -> GridGame:
        return BabaGame(task.levels, _load()[1])

    def secret_modules(self) -> tuple[str, ...]:
        return ("baba",)

    def render_frame(self, obs: Any) -> Any | None:
        """One tile per code, each different from every other, and meaning nothing: the picture
        says no more than the integers (see :func:`code_tile`)."""
        import numpy as np

        from regact.problems.grid_games.base import _current_grid

        grid = _current_grid(obs.frame)
        if grid is None:
            return None
        return np.concatenate(
            [np.concatenate([code_tile(code) for code in row], axis=1) for row in grid], axis=0
        )

    def render_frame_for_viewer(self, obs: Obs, task_name: str) -> Any | None:
        """The engine's own tiles (word blocks carry their text): for people, not the agent."""
        import numpy as np

        from regact.problems.grid_games.base import _current_grid

        grid = _current_grid(obs.frame)
        if grid is None:
            return None
        tiles = [[_tile(code) for code in row] for row in grid]
        rows = [np.concatenate(row, axis=1) for row in tiles]
        return np.concatenate(rows, axis=0).astype(np.uint8)


# The agent's tile colours, assigned arbitrarily: no object keeps its engine colour, so a shared
# colour means a shared code and nothing else. Black is the empty cell.
_CODE_COLOURS = (9, 12, 11, 15, 6, 10, 13, 7, 1, 2, 3, 14, 0, 8)
_UNKNOWN_COLOUR = 6


@cache
def code_tile(code: int) -> Any:
    """The agent's tile of a code: codes 1-14 are flat colours, 15-28 the same colours with a
    black hole; the unknown code is magenta with a black cross."""
    import numpy as np

    from regact.problems.grid_games.base import _RENDER_SCALE as scale
    from regact.problems.palette import ARC_PALETTE

    tile = np.zeros((scale, scale, 3), dtype=np.uint8)
    tile[:] = ARC_PALETTE[5]
    if code == _UNKNOWN_CODE:
        tile[:] = ARC_PALETTE[_UNKNOWN_COLOUR]
        for i in range(scale):
            tile[i, i] = tile[i, scale - 1 - i] = ARC_PALETTE[5]
    elif code > 0:
        holed, index = divmod(code - 1, len(_CODE_COLOURS))
        if holed > 1:
            raise ValueError(f"no tile for code {code}: the code table outgrew the agent's tiles")
        tile[1:-1, 1:-1] = ARC_PALETTE[_CODE_COLOURS[index]]
        if holed:
            third = scale // 3
            tile[third:-third, third:-third] = ARC_PALETTE[5]
    return tile


@cache
def _tile(code: int) -> Any:
    """The engine's 32-pixel picture of a code's kind (``empty`` is a blank tile)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from baba import world_object as w
        from baba.grid import BabaIsYouGrid

    kinds = _load()[2]
    kind = kinds[code] if 0 <= code < len(kinds) else {"name": "empty:red"}
    type_name, value = kind["name"].split(":")
    obj: Any
    if type_name == "empty":
        obj = None
    elif type_name == "wall":
        obj = w.Wall(value)
    elif type_name == "rule_object":
        obj = w.RuleObject(value)
    elif type_name == "rule_is":
        obj = w.RuleIs()
    elif type_name == "rule_property":
        obj = w.RuleProperty(value)
    elif type_name == "rule_color":
        obj = w.RuleColor(value)
    else:
        obj = w.make_obj(type_name, value)
    return BabaIsYouGrid.render_tile(obj, tile_size=32)


register_problem("babaisai", lambda kwargs: BabaIsAIProblem(**kwargs))
