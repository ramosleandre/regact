"""A Python engine for Baba Is You levels (the commercial game by Hempuli), rule for rule.

A level is a grid of objects ``{name, x, y, dir}``; ``text_*`` objects are words. Words in a
row (left to right) or a column (top to bottom) form rules: ``NOUN IS PROPERTY``,
``NOUN IS NOUN`` (a transformation), ``NOUN HAS NOUN``, with ``AND`` and ``NOT``. Each turn
the YOU objects step (pushing PUSH objects, blocked by STOP), MOVE objects step, the rules
are read again, transformations apply, then same-tile interactions resolve (SINK, DEFEAT,
HOT/MELT, OPEN/SHUT, WEAK, BONUS) and a YOU on a WIN wins. FLOAT objects only interact with
FLOAT objects. Fidelity is checked by replaying solutions verified on the real game.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

# Facing as stored in level files, and the move each input makes.
RIGHT, UP, LEFT, DOWN = 0, 1, 2, 3
STEP = {RIGHT: (1, 0), UP: (0, -1), LEFT: (-1, 0), DOWN: (0, 1)}
INPUT_DIR = {"r": RIGHT, "u": UP, "l": LEFT, "d": DOWN}
WAIT = "p"

VERBS = ("is", "has")
PROPERTIES = frozenset(
    {
        "you", "win", "stop", "push", "defeat", "sink", "hot", "melt", "open", "shut",
        "float", "move", "weak", "hide", "bonus", "pull", "red", "blue", "tele", "shift",
        "swap", "end", "done", "safe", "fall", "more", "power", "sleep", "word",
    }
)  # fmt: skip
EDGE = "edge"  # the level's border tiles: always solid, never named by a rule


@dataclass
class Obj:
    id: int
    name: str
    x: int
    y: int
    dir: int = DOWN

    @property
    def is_text(self) -> bool:
        return self.name.startswith("text_")

    @property
    def word(self) -> str:
        return self.name[5:]


@dataclass
class Rule:
    subject: str  # a noun, "text", or "level"
    verb: str  # "is" | "has"
    target: str  # a property or a noun
    subject_not: bool = False
    target_not: bool = False


@dataclass
class State:
    width: int
    height: int
    objects: list[Obj]
    next_id: int
    won: bool = False
    rules: list[Rule] = field(default_factory=list)


def load(level: dict[str, Any]) -> State:
    # The game's object list: columns left to right, top to bottom within a column; units act
    # in this order, which decides who pushes first.
    ordered = sorted(level["objects"], key=lambda o: (int(o["x"]), int(o["y"])))
    objects = [
        Obj(i, str(o["name"]), int(o["x"]), int(o["y"]), int(o.get("dir", DOWN)))
        for i, o in enumerate(ordered)
    ]
    state = State(int(level["width"]), int(level["height"]), objects, len(objects))
    _settle_rules(state)
    return state


# --------------------------------------------------------------------------------- rules


def _is_noun(word: str) -> bool:
    return word not in PROPERTIES and word not in VERBS and word not in ("and", "not")


def _parse_line(words: list[str]) -> list[Rule]:
    """Every rule a line of consecutive words spells; sentences may share words."""
    rules: list[Rule] = []
    for start in range(len(words)):
        if start and words[start - 1] == "not":
            continue  # "NOT BABA IS ..." must not also read as "BABA IS ..."
        i, subjects = start, []
        while True:
            negated = False
            while i < len(words) and words[i] == "not":
                negated, i = not negated, i + 1
            if i >= len(words) or not _is_noun(words[i]):
                subjects = []
                break
            subjects.append((words[i], negated))
            i += 1
            if i < len(words) and words[i] == "and":
                i += 1
                continue
            break
        if not subjects or i >= len(words) or words[i] not in VERBS:
            continue
        verb, i = words[i], i + 1
        targets = []
        while True:
            negated = False
            while i < len(words) and words[i] == "not":
                negated, i = not negated, i + 1
            if i >= len(words) or words[i] in VERBS or words[i] == "and":
                break
            target = words[i]
            if verb == "has" and not _is_noun(target):
                break
            targets.append((target, negated))
            i += 1
            if i < len(words) - 1 and words[i] == "and" and words[i + 1] not in VERBS:
                i += 1
                continue
            break
        for subject, subject_not in subjects:
            for target, target_not in targets:
                rules.append(Rule(subject, verb, target, subject_not, target_not))
    return rules


def parse_rules(state: State) -> list[Rule]:
    grid: dict[tuple[int, int], list[str]] = {}
    for o in state.objects:
        if o.is_text:
            grid.setdefault((o.x, o.y), []).append(o.word)
    rules: list[Rule] = []
    seen: set[tuple[Any, ...]] = set()
    for dx, dy in ((1, 0), (0, 1)):
        for x, y in sorted(grid):
            if (x - dx, y - dy) in grid:
                continue  # not the start of a line
            line, cx, cy = [], x, y
            while (cx, cy) in grid:
                line.append(grid[(cx, cy)][0])
                cx, cy = cx + dx, cy + dy
            for rule in _parse_line(line):
                key = (rule.subject, rule.verb, rule.target, rule.subject_not, rule.target_not)
                if key not in seen:
                    seen.add(key)
                    rules.append(rule)
    return rules


def _names_in_level(state: State) -> set[str]:
    return {o.name for o in state.objects if not o.is_text and o.name != EDGE}


def _subjects(rule: Rule, state: State) -> set[str]:
    """The object names a rule's subject covers ("text_*" for TEXT)."""
    if rule.subject == "level":
        return {"level"}
    if rule.subject == "text":
        named = {o.name for o in state.objects if o.is_text}
        return set() if rule.subject_not else named
    if rule.subject_not:
        return _names_in_level(state) - {rule.subject}
    return {rule.subject}


def properties(state: State) -> dict[str, set[str]]:
    """Each object name's properties under the current rules (NOT wins over a plain rule)."""
    has: dict[str, set[str]] = {}
    denied: dict[str, set[str]] = {}
    for rule in state.rules:
        if rule.verb != "is" or rule.target not in PROPERTIES:
            continue
        for name in _subjects(rule, state):
            (denied if rule.target_not else has).setdefault(name, set()).add(rule.target)
    for o in state.objects:
        if o.is_text:
            has.setdefault(o.name, set()).add("push")
    return {name: props - denied.get(name, set()) for name, props in has.items()}


def _transformations(state: State) -> dict[str, str]:
    keep = {r.subject for r in state.rules if r.verb == "is" and r.subject == r.target}
    denied = {
        (name, r.target)
        for r in state.rules
        if r.verb == "is" and r.target_not
        for name in _subjects(r, state)
    }
    out: dict[str, str] = {}
    for rule in state.rules:
        if rule.verb != "is" or rule.target_not or rule.target in PROPERTIES:
            continue
        if rule.target == rule.subject or rule.subject == "level":
            continue
        for name in _subjects(rule, state):
            noun = name[5:] if name.startswith("text_") else name
            if noun in keep or (name, rule.target) in denied or name in out:
                continue
            out[name] = "text_" + noun if rule.target == "text" else rule.target
    return out


def _settle_rules(state: State) -> None:
    """Read the rules, apply transformations, and read again until nothing changes."""
    for _ in range(10):
        state.rules = parse_rules(state)
        changes = _transformations(state)
        changed = False
        for o in state.objects:
            if o.name in changes and changes[o.name] != o.name:
                o.name = changes[o.name]
                changed = True
        if not changed:
            return
    state.rules = parse_rules(state)


# ------------------------------------------------------------------------------ movement
# Follows the Baba Is You wiki's "Order of Operations": units move in object-list order, and a
# move into a tile checks OPEN/SHUT, then WEAK, then PUSH, then STOP.


def _props_of(o: Obj, props: dict[str, set[str]]) -> set[str]:
    return props.get(o.name, set())


def _floats(o: Obj, props: dict[str, set[str]]) -> bool:
    return "float" in _props_of(o, props)


def _opens(a: Obj, b: Obj, props: dict[str, set[str]]) -> bool:
    pa, pb = _props_of(a, props), _props_of(b, props)
    return ("open" in pa and "shut" in pb) or ("shut" in pa and "open" in pb)


def _solid(o: Obj, props: dict[str, set[str]]) -> bool:
    return o.name == EDGE or bool({"stop", "push"} & _props_of(o, props))


def _try_move(state: State, o: Obj, d: int, props: dict[str, set[str]], doomed: set[int]) -> bool:
    """Move ``o`` one step in direction ``d``, pushing what is in the way; False if blocked.
    Objects destroyed on contact (OPEN/SHUT, a WEAK object that cannot move) go to ``doomed``."""
    dx, dy = STEP[d]
    nx, ny = o.x + dx, o.y + dy
    if not (0 <= nx < state.width and 0 <= ny < state.height):
        return False
    ahead = [
        t for t in state.objects if t.x == nx and t.y == ny and t is not o and t.id not in doomed
    ]
    for t in ahead:
        if t.name != EDGE and _solid(t, props) and _opens(o, t, props):
            doomed.update((o.id, t.id))  # a key into a door: both are gone, the move happened
            return True
    for t in ahead:
        if t.name == EDGE:
            return False
        p_t = _props_of(t, props)
        if (
            "weak" in p_t
            and "stop" in p_t
            and "push" not in p_t
            and _floats(t, props) == _floats(o, props)
        ):
            continue  # a weak solid object does not block: it breaks on the later overlap check
        if "push" in p_t:
            if not _try_move(state, t, d, props, doomed):
                if "weak" in p_t:
                    doomed.add(t.id)  # a weak object that cannot be pushed breaks instead
                    continue
                return False
            continue
        if "stop" in p_t:
            return False
    o.x, o.y = nx, ny
    o.dir = d
    return True


def _move_phase(state: State, key: str, direction: int | None) -> None:
    props = properties(state)
    doomed: set[int] = set()
    for o in [o for o in state.objects if key in _props_of(o, props)]:
        if o.id in doomed:
            continue
        d = direction if key == "you" else o.dir
        assert d is not None
        o.dir = d
        if _try_move(state, o, d, props, doomed):
            continue
        if key == "move":  # a MOVE object that bumps into something turns around and goes back
            o.dir = (d + 2) % 4
            if not _try_move(state, o, o.dir, props, doomed) and "weak" in _props_of(o, props):
                doomed.add(o.id)
        elif "weak" in _props_of(o, props):
            doomed.add(o.id)
    _destroy(state, doomed)


# ---------------------------------------------------------------------------- interactions


def _destroy(state: State, doomed: set[int]) -> None:
    if not doomed:
        return
    has = [r for r in state.rules if r.verb == "has" and not r.target_not]
    survivors, spawned = [], []
    for o in state.objects:
        if o.id not in doomed:
            survivors.append(o)
            continue
        for rule in has:
            if o.name in _subjects(rule, state):
                noun = o.word if o.is_text else o.name
                name = "text_" + noun if rule.target == "text" else rule.target
                spawned.append(Obj(state.next_id, name, o.x, o.y, o.dir))
                state.next_id += 1
    state.objects = survivors + spawned


# Same-tile interactions, in the game's order; each step deletes before the next one looks.
PHASES = ("sink", "weak", "melt", "defeat", "open", "bonus")


def _phase(state: State, kind: str, props: dict[str, set[str]]) -> set[int]:
    def p(o: Obj) -> set[str]:
        return _props_of(o, props)

    level = props.get("level", set())  # the level is on every tile at once, on its own float
    level_float = "float" in level
    cells: dict[tuple[int, int], list[Obj]] = {}
    for o in state.objects:
        cells.setdefault((o.x, o.y), []).append(o)
    doomed: set[int] = set()
    for o in state.objects:
        mine = p(o)
        floats = "float" in mine
        others = [
            b
            for b in cells[(o.x, o.y)]
            if b is not o and b.name != EDGE and ("float" in p(b)) == floats
        ]
        on_level = level if level_float == floats else set()
        if kind == "sink" and "sink" in mine and others:
            doomed.add(o.id)
            doomed.update(b.id for b in others)
        elif kind == "weak" and "weak" in mine and others:
            doomed.add(o.id)
        elif kind == "melt" and "melt" in mine:
            if "hot" in mine or "hot" in on_level or any("hot" in p(b) for b in others):
                doomed.add(o.id)
        elif kind == "defeat" and "you" in mine:
            if "defeat" in mine or "defeat" in on_level or any("defeat" in p(b) for b in others):
                doomed.add(o.id)
        elif kind == "open" and ({"open", "shut"} & mine):
            if {"open", "shut"} <= mine:
                doomed.add(o.id)
            elif "open" in mine:
                shut = [b for b in others if "shut" in p(b) and b.id not in doomed]
                if shut:
                    doomed.update((o.id, shut[0].id))
        elif kind == "bonus" and "bonus" in mine:
            if any("you" in p(b) for b in others) or "you" in on_level:
                doomed.add(o.id)
    return doomed


def _interactions(state: State) -> None:
    for kind in PHASES:
        _destroy(state, _phase(state, kind, properties(state)))


def _check_win(state: State) -> bool:
    props = properties(state)
    level = props.get("level", set())
    level_float = "float" in level
    for o in state.objects:
        mine = props.get(o.name, set())
        if "you" not in mine:
            continue
        floats = "float" in mine
        if "win" in level and level_float == floats:
            return True
        for t in state.objects:
            t_props = props.get(t.name, set())
            if t.x == o.x and t.y == o.y and "win" in t_props and ("float" in t_props) == floats:
                return True
    if "you" in level:  # the level is YOU: any same-float WIN object wins
        return any(
            "win" in props.get(t.name, set())
            and ("float" in props.get(t.name, set())) == level_float
            for t in state.objects
        )
    return False


# ----------------------------------------------------------------------------------- turn


def step(state: State, move: str) -> State:
    """The state after one input: ``l``/``r``/``u``/``d`` or ``p`` (wait). Pure."""
    s = copy.deepcopy(state)
    if s.won:
        return s
    if move != WAIT:
        _move_phase(s, "you", INPUT_DIR[move])
    _move_phase(s, "move", None)
    _settle_rules(s)
    _interactions(s)
    s.won = _check_win(s)  # in the same step as the interactions, before the rules are re-read
    _settle_rules(s)
    return s


def play(level: dict[str, Any], moves: str) -> tuple[State, int | None]:
    """Replay ``moves``; the 1-based index of the move that won, or None."""
    state = load(level)
    for i, move in enumerate(moves, start=1):
        state = step(state, move)
        if state.won:
            return state, i
    return state, None
