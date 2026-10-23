"""PuzzleScript: packs of levels from puzzlescript.net gallery games, played by the original engine.

PuzzleScript (increpare, MIT) is a language for turn-based grid puzzles; each game declares its
own objects, rewrite rules and win conditions, so every task has rules of its own. The engine runs
unmodified under Node (``puzzlescript_node/``); one Node process per environment.

The games are not redistributed: authors' gists carry no licence. ``data/puzzlescript_games.json``
lists each game's pinned source URL and SHA-256, the playable levels kept and their shortest
solutions; ``scripts/grid_games/install_puzzlescript.py`` fetches the engine and the games into a
local directory (``REGACT_PUZZLESCRIPT_DIR``, default ``~/.cache/regact/puzzlescript``).

A cell is the list of the ids of the objects in it (one per collision layer), so the frame shows
the whole level state; the agent never sees object names, titles or messages.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import cache
from pathlib import Path
from typing import Any

from regact.envclient.obs import Obs
from regact.obs.errors import ErrorCategory, RegactError
from regact.problems.base import register_problem
from regact.problems.grid_games.base import (
    GridGame,
    GridGameProblem,
    GridTask,
    _current_grid,
)
from regact.problems.palette import ARC_PALETTE

_GAMES = Path(__file__).parent / "data" / "puzzlescript_games.json"
_SERVER = Path(__file__).parent / "puzzlescript_node" / "serve.js"
ENGINE_COMMIT = "666f92c9c7dfe5cdf86c3e7a64374f7ad667b087"  # increpare/PuzzleScript, 2026-10-16
# Our directions (1 up, 2 down, 3 left, 4 right, 5 the game's "action") -> the engine's input codes.
ENGINE_DIR = {1: 0, 2: 2, 3: 1, 4: 3, 5: 4}
_BACKGROUND = 0  # the background object's id in every game of the set (checked by a test)
# The agent's picture: a tile per cell, wide enough for three objects side by side.
_TILE = 30
_OBJECT_COLOURS = (9, 12, 11, 15, 6, 10, 13, 7, 1, 2, 3, 14, 0, 8)  # black is the background
_SPRITE_SCALE = 5  # px per sprite pixel in the viewer's picture (a cell is 5x5 sprite pixels)


def data_dir() -> Path:
    """Where the engine and the game sources are installed."""
    default = Path.home() / ".cache" / "regact" / "puzzlescript"
    return Path(os.environ.get("REGACT_PUZZLESCRIPT_DIR") or default)


def engine_src() -> Path:
    return data_dir() / "engine" / ENGINE_COMMIT / "src"


def game_source(sha256: str) -> Path:
    return data_dir() / "games" / f"{sha256}.txt"


class PuzzleScriptGame(GridGame):
    """One task's levels, played by a Node subprocess running the engine."""

    def __init__(self, source: Path, levels: tuple[int, ...]) -> None:
        self._source = source
        self._levels = levels
        self._proc: subprocess.Popen[str] | None = None
        self._cells: list[list[list[int]]] = []
        self._won = False

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._proc is None:
            node = shutil.which("node")
            if node is None or not self._source.is_file() or not engine_src().is_dir():
                raise RegactError(
                    ErrorCategory.ENV_RUNTIME,
                    "PuzzleScript needs Node and the installed engine and games: run "
                    "`make install-puzzlescript`",
                )
            self._proc = subprocess.Popen(
                [node, str(_SERVER)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                env={**os.environ, "PS_SRC": str(engine_src())},
            )
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        reply = json.loads(self._proc.stdout.readline() or '{"error": "the engine exited"}')
        if "error" in reply:
            raise RegactError(ErrorCategory.ENV_RUNTIME, f"PuzzleScript: {reply['error']}")
        return reply

    def load_level(self, index: int) -> None:
        reply = self._call({"cmd": "load", "file": str(self._source), "level": self._levels[index]})
        self._cells, self._won = reply["cells"], False

    def apply(self, action: int) -> None:
        reply = self._call({"cmd": "step", "dir": ENGINE_DIR[action]})
        self._cells, self._won = reply["cells"], bool(reply["winning"])

    def solved(self) -> bool:
        return self._won

    def grid(self) -> list[list[Any]]:
        return self._cells

    def close(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self._proc = None


@cache
def _load() -> tuple[dict[str, GridTask], dict[str, dict[str, Any]]]:
    games = json.loads(_GAMES.read_text(encoding="utf-8"))["games"]
    tasks = {
        game["key"]: GridTask(
            key=game["key"],
            levels=tuple(level["index"] for level in game["levels"]),
            reference_actions=tuple(level["optimal"] for level in game["levels"]),
        )
        for game in games
    }
    return tasks, {game["key"]: game for game in games}


def _rgb(colour: str) -> tuple[int, int, int]:
    value = colour.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    try:
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    except ValueError:  # an unresolved colour name: draw it grey
        return (128, 128, 128)


@cache
def _sprites(task_name: str) -> tuple[list[tuple[list, list]], list[int], tuple[int, int, int]]:
    """Each object's (colours, 5x5 matrix of colour indices, -1 = transparent), each object's
    layer, and the background colour, from the compiled game."""
    tasks, games = _load()
    game = PuzzleScriptGame(game_source(games[task_name]["sha256"]), tasks[task_name].levels)
    try:
        game.load_level(0)
        info = game._call({"cmd": "info"})
    finally:
        game.close()
    sprites = [
        ([_rgb(c) for c in sprite["colors"]], sprite["matrix"]) for sprite in info["sprites"]
    ]
    return sprites, info["layers"], _rgb(info["background"])


def object_stripe(stripe: Any, obj: int) -> None:
    """Draw one object into its stripe of a tile: ids 1-14 a flat colour, 15-28 the same colour
    with a black hole, inside a one-pixel black border."""
    holed, index = divmod(obj - 1, len(_OBJECT_COLOURS))
    if not 0 <= holed <= 1:
        raise ValueError(f"no picture for object id {obj}: the game has too many objects")
    height, width = stripe.shape[:2]
    stripe[1:-1, 1:-1] = ARC_PALETTE[_OBJECT_COLOURS[index]]
    if holed:
        stripe[height // 3 : -(height // 3), width // 3 : width - width // 3] = ARC_PALETTE[5]


class PuzzleScriptProblem(GridGameProblem):
    name = "puzzlescript"
    frame_note = (
        "each cell is the list of the integer ids of the objects in it (one object per layer, so "
        "a cell can hold several; the same object always has the same id)"
    )
    action_note = (
        "Actions 1-4 are directional (typically up, down, left, right); action 5, when "
        "available, is the game's own extra action. Discover their effects by interaction."
    )

    def tasks(self) -> dict[str, GridTask]:
        return _load()[0]

    def task_actions(self, task_name: str) -> tuple[int, ...]:
        return tuple(_load()[1][task_name]["actions"])

    def make_game(self, task: GridTask) -> GridGame:
        return PuzzleScriptGame(game_source(_load()[1][task.key]["sha256"]), task.levels)

    def render_frame(self, obs: Obs) -> Any | None:
        """The background black; each other object of a cell a vertical stripe, in id order,
        drawn by :func:`object_stripe`. Every cell content has its own tile."""
        import numpy as np

        grid = _current_grid(obs.frame, depth=3)
        if grid is None:
            return None
        out = np.zeros((len(grid) * _TILE, len(grid[0]) * _TILE, 3), dtype=np.uint8)
        for r, row in enumerate(grid):
            for c, objects in enumerate(row):
                shown = [obj for obj in objects if obj != _BACKGROUND]
                tile = out[r * _TILE : (r + 1) * _TILE, c * _TILE : (c + 1) * _TILE]
                for k, obj in enumerate(shown):
                    left = k * _TILE // len(shown)
                    object_stripe(tile[:, left : (k + 1) * _TILE // len(shown)], obj)
        return out

    def render_frame_for_viewer(self, obs: Obs, task_name: str) -> Any | None:
        """The game's own 5x5 sprites, layer by layer: for people, not the agent."""
        import numpy as np

        grid = _current_grid(obs.frame, depth=3)
        if grid is None or task_name not in self.tasks():
            return self.render_frame(obs)
        sprites, layers, background = _sprites(task_name)
        px = _SPRITE_SCALE
        out = np.zeros((len(grid) * 5 * px, len(grid[0]) * 5 * px, 3), dtype=np.uint8)
        for r, row in enumerate(grid):
            for c, objects in enumerate(row):
                tile = np.empty((5, 5, 3), dtype=np.uint8)
                tile[:] = background
                for obj in sorted(objects, key=lambda o: layers[o]):
                    colours, matrix = sprites[obj]
                    for y in range(5):
                        for x in range(5):
                            index = matrix[y][x] if matrix else 0
                            if index >= 0:
                                tile[y, x] = colours[index]
                block = np.repeat(np.repeat(tile, px, axis=0), px, axis=1)
                out[r * 5 * px : (r + 1) * 5 * px, c * 5 * px : (c + 1) * 5 * px] = block
        return out

    def render_obs_text(self, obs: Obs) -> str | None:
        grid = _current_grid(obs.frame, depth=3)
        if grid is None:
            return None
        info = obs.info or {}
        header = (
            f"state={info.get('state', '?')}  "
            f"levels_completed={info.get('levels_completed', 0)}/{info.get('win_levels', '?')}  "
            f"available_actions={obs.available_actions}"
        )
        return header + "\n\n" + "\n".join(json.dumps(row) for row in grid)


register_problem("puzzlescript", lambda kwargs: PuzzleScriptProblem(**kwargs))
