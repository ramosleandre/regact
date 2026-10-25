"""Baba Is You: the campaign levels of the game by Hempuli, played by regact's own engine.

The levels are the game's (extracted from its files by github.com/stared/baba-is-harbor, each
with a solution checked on the real game). They are fetched at install and pinned by SHA-256,
never redistributed; ``data/babaisyou_levels.json`` holds their URLs, hashes and each level's
reference (the winning move of that checked solution, replayed through this engine).

Nothing is hidden from the agent: a cell lists what a player sees on it, objects by name and
facing (``"baba:right"``) and word blocks in capitals (``"BABA"``, ``"IS"``, ``"YOU"``), and
``obs.info["active_rules"]`` lists the rules in force, as the game lights the words that form
them. The difficulty is the reasoning, not the reading. The engine package is hidden from the
agent's sandbox (``secret_modules``), as a human player never sees the game's code.
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path
from typing import Any

from regact.config.schema import InfoMode
from regact.envclient.obs import Obs
from regact.problems.base import register_problem
from regact.problems.grid_games.babaisyou import engine
from regact.problems.grid_games.base import GridGame, GridGameProblem, GridTask

_DATA = Path(__file__).parents[1] / "data" / "babaisyou_levels.json"
# Solutions for the levels harbor has none for (worlds 4-9), kept in this hidden package.
SOLUTIONS = Path(__file__).parent / "solutions.json"
# Our actions -> the game's inputs: 1 up, 2 down, 3 left, 4 right, 5 wait (the game's idle key).
INPUT = {1: "u", 2: "d", 3: "l", 4: "r", 5: engine.WAIT}
FACING = {engine.RIGHT: "right", engine.UP: "up", engine.LEFT: "left", engine.DOWN: "down"}


def data_dir() -> Path:
    """Where the levels are installed (``make install-babaisyou``)."""
    default = Path.home() / ".cache" / "regact" / "babaisyou"
    return Path(os.environ.get("REGACT_BABAISYOU_DIR") or default)


def level_path(sha256: str) -> Path:
    return data_dir() / "levels" / f"{sha256}.json"


def solution_path(sha256: str) -> Path:
    """A level's solution checked on the real game (for the tests; the agent never sees it)."""
    return data_dir() / "solutions" / f"{sha256}.sh"


@cache
def _load() -> tuple[dict[str, GridTask], dict[str, dict[str, Any]]]:
    data = json.loads(_DATA.read_text(encoding="utf-8"))
    tasks = {
        pack["key"]: GridTask(
            key=pack["key"],
            levels=tuple(level["sha256"] for level in pack["levels"]),
            reference_actions=tuple(level["reference"] for level in pack["levels"]),
        )
        for pack in data["packs"]
    }
    return tasks, {level["sha256"]: level for pack in data["packs"] for level in pack["levels"]}


@cache
def _level(sha256: str) -> dict[str, Any]:
    path = level_path(sha256)
    if not path.is_file():
        from regact.obs.errors import ErrorCategory, RegactError

        raise RegactError(
            ErrorCategory.ENV_RUNTIME,
            "Baba Is You levels are not installed: run make install-babaisyou",
        )
    level: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return level


def cell_entry(o: engine.Obj) -> str:
    """How an object reads on the board: a word in capitals, a thing by name and facing."""
    if o.is_text:
        return o.word.upper()
    if o.name == engine.EDGE:
        return engine.EDGE
    return f"{o.name}:{FACING[o.dir]}"


def board(state: engine.State) -> list[list[list[str]]]:
    props, _ = engine.object_properties(state)
    rows: list[list[list[str]]] = [[[] for _ in range(state.width)] for _ in range(state.height)]
    for o in state.objects:
        if "hide" not in props[o.id]:
            rows[o.y][o.x].append(cell_entry(o))
    for row in rows:  # the engine's own order within a tile means nothing to a player
        for cell in row:
            cell.sort()
    return rows


def rule_text(rule: engine.Rule) -> str:
    def word(name: str, negated: bool) -> str:
        return ("NOT " if negated else "") + name

    parts = [word(rule.subject, rule.subject_not)]
    for c in rule.conditions:
        parts.append(word(c.kind, c.negated))
        parts.append(" AND ".join(word(n, negated) for n, negated in c.args))
    parts += [rule.verb, word(rule.target, rule.target_not)]
    return " ".join(parts).upper()


class BabaIsYouGame(GridGame):
    def __init__(self, levels: tuple[str, ...]) -> None:
        self._levels = levels
        self._state: engine.State | None = None

    def load_level(self, index: int) -> None:
        self._state = engine.load(_level(self._levels[index]))

    def apply(self, action: int) -> None:
        assert self._state is not None
        self._state = engine.step(self._state, INPUT[action])

    def solved(self) -> bool:
        return self._state is not None and self._state.won

    def grid(self) -> list[list[Any]]:
        assert self._state is not None
        return board(self._state)

    def info(self) -> dict[str, Any]:
        assert self._state is not None
        return {"active_rules": sorted(rule_text(r) for r in self._state.rules)}


class BabaIsYouProblem(GridGameProblem):
    name = "babaisyou"
    actions = (1, 2, 3, 4, 5)
    # The reference is a solution checked on the real game, not a proven optimum: the scores
    # keep ARC's names (relative to a human-made solution), not ROAE/LROAE.
    main_metrics = ("rhae", "lrhae_uncapped", "mean_relative_actions")
    frame_note = (
        "each cell is the list of what is on that tile: objects by name and the way they face "
        '(e.g. "baba:right", "wall:down"), word blocks in capital letters (e.g. "BABA", "IS", '
        '"YOU"), and "edge" for the level border; `obs.info["active_rules"]` lists the rules '
        'currently in force (e.g. "BABA IS YOU")'
    )
    action_note = (
        "Actions 1-4 move up, down, left and right; action 5 waits one turn (nothing you control "
        "moves, but the rest of the level takes its turn)."
    )

    def tasks(self) -> dict[str, GridTask]:
        return _load()[0]

    def build_prompt(self, task_name: str, **kwargs: Any) -> str:
        prompt = super().build_prompt(task_name, **kwargs)
        if kwargs.get("info_mode") is InfoMode.MINIMAL:
            return prompt
        # Nothing is masked here: a person playing knows which game it is.
        return prompt.replace(
            "This is an interactive puzzle game played on a small grid.",
            "This is an interactive puzzle game played on a small grid: the campaign levels of "
            "the game Baba Is You.",
            1,
        )

    def make_game(self, task: GridTask) -> GridGame:
        return BabaIsYouGame(task.levels)

    def secret_modules(self) -> tuple[str, ...]:
        return ("regact.problems.grid_games.babaisyou",)

    def derived_submission_metrics(
        self, task_name: str, raw_episodes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        from regact.problems.arc_agi.scoring import rhae_from_results

        rhae = rhae_from_results(
            {"episodes": raw_episodes}, baseline_actions=list(self._reference(task_name))
        )
        return rhae.scores() if rhae is not None else {}

    def derived_trace_metrics(
        self, task_name: str, trace: list[tuple[int, dict[str, Any]]]
    ) -> dict[str, Any]:
        from regact.problems.arc_agi.scoring import actions_per_level_from_trace, rhae_score

        reference = self._reference(task_name)
        levels = [(actions, int(info.get("levels_completed", 0) or 0)) for actions, info in trace]
        per_level = actions_per_level_from_trace(levels)
        return rhae_score(
            baseline_actions=reference,
            actions_per_level=per_level,
            levels_completed=len(per_level),
            total_levels=len(reference),
        ).scores()

    def render_frame(self, obs: Obs) -> Any | None:
        """Each tile with its entries written in it (several entries share the tile in bands):
        word blocks white on black, objects black on a colour of their own."""
        return _render(obs.frame)

    def render_obs_text(self, obs: Obs) -> str | None:
        rows = obs.frame if isinstance(obs.frame, list) else None
        if not rows:
            return None
        info = obs.info or {}
        lines = [
            f"levels_completed={info.get('levels_completed', 0)}/{info.get('win_levels', '?')}",
            "active rules: " + " | ".join(info.get("active_rules", [])),
        ]
        for y, row in enumerate(rows):
            for x, cell in enumerate(row):
                shown = [e for e in cell if e != engine.EDGE]
                if shown:
                    lines.append(f"({x},{y}) " + " ".join(shown))
        return "\n".join(lines)


_TILE = 40


def _colour(name: str) -> tuple[int, int, int]:
    import hashlib

    digest = hashlib.sha256(name.encode()).digest()
    return (110 + digest[0] % 140, 110 + digest[1] % 140, 110 + digest[2] % 140)


def _render(frame: Any) -> Any | None:
    if not isinstance(frame, list) or not frame:
        return None
    import numpy as np
    from PIL import Image, ImageDraw

    height, width = len(frame), len(frame[0])
    image = Image.new("RGB", (width * _TILE, height * _TILE), (20, 20, 24))
    pen = ImageDraw.Draw(image)
    for y, row in enumerate(frame):
        for x, cell in enumerate(row):
            entries = [e for e in cell if isinstance(e, str)]
            if not entries:
                continue
            band = _TILE // len(entries)
            for k, entry in enumerate(entries):
                top, left = y * _TILE + k * band, x * _TILE
                box = (left, top, left + _TILE - 1, top + band - 1)
                if entry == engine.EDGE:
                    pen.rectangle(box, fill=(60, 60, 64))
                    continue
                name, _, facing = entry.partition(":")
                if entry.isupper():  # a word block
                    pen.rectangle(box, fill=(0, 0, 0), outline=(235, 235, 235))
                    pen.text((left + 2, top + 1), entry[:6], fill=(255, 255, 255))
                else:
                    pen.rectangle(box, fill=_colour(name))
                    pen.text((left + 2, top + 1), name[:6], fill=(0, 0, 0))
                    if facing:
                        marker = {"right": ">", "left": "<", "up": "^", "down": "v"}[facing]
                        pen.text((left + _TILE - 9, top + 1), marker, fill=(0, 0, 0))
    return np.asarray(image)


register_problem("babaisyou", lambda kwargs: BabaIsYouProblem(**kwargs))
