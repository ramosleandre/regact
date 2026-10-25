"""A Python engine for Baba Is You levels (the commercial game by Hempuli), rule for rule.

A level is a grid of objects ``{name, x, y, dir}``; ``text_*`` objects are words. Words in a
row (left to right) or a column (top to bottom) form rules: ``NOUN IS PROPERTY``,
``NOUN IS NOUN`` (a transformation), ``NOUN HAS NOUN``, ``NOUN MAKE NOUN``, with ``AND``,
``NOT`` and the conditions ``ON``, ``NEAR`` and ``FACING``. A turn follows the wiki's "Order of
Operations": YOU, MOVE and SHIFT movement, transformations, TELE, the SHIFT turn, FALL,
directions, then the same-tile interactions (MORE, SINK, WEAK, MELT, DEFEAT, OPEN/SHUT, MAKE,
BONUS) and the win check. FLOAT objects only interact with FLOAT objects. Fidelity is checked
by replaying solutions verified on the real game.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Any

# Facing as stored in level files, and the move each input makes.
RIGHT, UP, LEFT, DOWN = 0, 1, 2, 3
STEP = {RIGHT: (1, 0), UP: (0, -1), LEFT: (-1, 0), DOWN: (0, 1)}
INPUT_DIR = {"r": RIGHT, "u": UP, "l": LEFT, "d": DOWN}
DIRECTIONS = {"right": RIGHT, "up": UP, "left": LEFT, "down": DOWN}
WAIT = "p"

VERBS = ("is", "has", "make")
CONDITIONS = ("on", "near", "facing")
PROPERTIES = frozenset(
    {
        "you", "win", "stop", "push", "defeat", "sink", "hot", "melt", "open", "shut",
        "float", "move", "weak", "hide", "bonus", "pull", "red", "blue", "tele", "shift",
        "swap", "end", "done", "safe", "fall", "more", "power", "sleep", "word", "best",
        "right", "up", "left", "down",
    }
)  # fmt: skip
EDGE = "edge"  # the level's border tiles: always solid, never named by a rule
# Nouns that name no single kind of object.
TEXT, LEVEL, EMPTY, ALL, GROUP = "text", "level", "empty", "all", "group"

Props = dict[str, int]  # a property and how many rules give it (MOVE and SHIFT stack)


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


@dataclass(frozen=True)
class Condition:
    kind: str  # "on" | "near" | "facing"
    args: tuple[tuple[str, bool], ...]  # (noun or direction, negated), all must hold
    negated: bool = False


@dataclass
class Rule:
    subject: str  # a noun, or text / level / empty / all / group
    verb: str  # "is" | "has" | "make"
    target: str  # a property or a noun
    subject_not: bool = False
    target_not: bool = False
    conditions: tuple[Condition, ...] = ()


@dataclass
class State:
    width: int
    height: int
    objects: list[Obj]
    next_id: int
    won: bool = False
    rules: list[Rule] = field(default_factory=list)
    turn: int = 0
    seed: int = 0  # the game draws a teleport destination at random among 3+ teleporters


def load(level: dict[str, Any], seed: int = 0) -> State:
    # The game's object list: columns left to right, top to bottom within a column; units act
    # in this order, which decides who pushes first.
    ordered = sorted(level["objects"], key=lambda o: (int(o["x"]), int(o["y"])))
    objects = [
        Obj(i, str(o["name"]), int(o["x"]), int(o["y"]), int(o.get("dir", DOWN)))
        for i, o in enumerate(ordered)
    ]
    state = State(int(level["width"]), int(level["height"]), objects, len(objects), seed=seed)
    _settle_rules(state)
    return state


# --------------------------------------------------------------------------------- rules


def _is_noun(word: str) -> bool:
    return (
        word not in PROPERTIES
        and word not in VERBS
        and word not in CONDITIONS
        and word not in ("and", "not")
    )


def _negations(words: list[str], i: int) -> tuple[bool, int]:
    negated = False
    while i < len(words) and words[i] == "not":
        negated, i = not negated, i + 1
    return negated, i


def _list(words: list[str], i: int, accept: Any) -> tuple[list[tuple[str, bool]], int]:
    """``[NOT]* W (AND [NOT]* W)*`` for words ``accept`` takes; stops before an AND that
    leads to a condition or a verb (the caller's continuation)."""
    items: list[tuple[str, bool]] = []
    while True:
        negated, j = _negations(words, i)
        if j >= len(words) or not accept(words[j]):
            return items, i
        items.append((words[j], negated))
        i = j + 1
        if i + 1 < len(words) and words[i] == "and":
            _, k = _negations(words, i + 1)
            if k < len(words) and accept(words[k]):
                i += 1
                continue
        return items, i


def _parse_from(words: list[str], i: int) -> tuple[list[Rule], int]:
    """The rules of the sentence starting at ``i``, and the index of its verb."""
    subjects, i = _list(words, i, _is_noun)
    if not subjects:
        return [], i
    conditions: list[Condition] = []
    while True:
        negated, j = _negations(words, i)
        if j >= len(words) or words[j] not in CONDITIONS:
            break
        kind = words[j]
        accept = (lambda w: _is_noun(w) or w in DIRECTIONS) if kind == "facing" else _is_noun
        args, k = _list(words, j + 1, accept)
        if not args:
            return [], i
        conditions.append(Condition(kind, tuple(args), negated))
        i = k
        if i + 1 < len(words) and words[i] == "and":
            _, k = _negations(words, i + 1)
            if k < len(words) and words[k] in CONDITIONS:
                i += 1
                continue
        break
    if i >= len(words) or words[i] not in VERBS:
        return [], i
    verb = words[i]
    if verb == "is":
        targets, _ = _list(words, i + 1, lambda w: _is_noun(w) or w in PROPERTIES)
    else:
        targets, _ = _list(words, i + 1, _is_noun)
    rules = [
        Rule(subject, verb, target, subject_not, target_not, tuple(conditions))
        for subject, subject_not in subjects
        for target, target_not in targets
    ]
    return rules, i


def _parse_line(words: list[str]) -> list[Rule]:
    """Every rule a line of consecutive words spells; sentences may share words, but none
    starts inside the subject or conditions of an earlier one ("KEKE ON GRASS IS DEFEAT" does
    not also say "GRASS IS DEFEAT"; "ROCK IS BABA IS YOU" says both)."""
    rules: list[Rule] = []
    inside: set[int] = set()
    for start in range(len(words)):
        if start in inside:
            continue
        found, verb_at = _parse_from(words, start)
        if found:
            rules.extend(found)
            inside.update(range(start + 1, verb_at))
    return rules


_MAX_READINGS = 16  # a line with stacked words reads every combination, up to this many


def _readings(cells: list[list[str]]) -> list[list[str]]:
    readings: list[list[str]] = [[]]
    for cell in cells:
        if len(readings) * len(cell) > _MAX_READINGS:
            cell = cell[:1]
        readings = [[*r, w] for r in readings for w in cell]
    return readings


def _rule_key(rule: Rule) -> tuple[Any, ...]:
    return (
        rule.subject, rule.verb, rule.target, rule.subject_not, rule.target_not, rule.conditions
    )  # fmt: skip


def parse_rules(state: State) -> list[Rule]:
    """The rules the words spell; objects that are WORD (under the current rules) count as
    their own word. A rule spelled by two separate lines counts twice (MOVE, SHIFT stack)."""
    grid: dict[tuple[int, int], list[str]] = {}
    word_objects = set()
    if any(r.target == "word" for r in state.rules):
        props, _ = object_properties(state)
        word_objects = {o.id for o in state.objects if "word" in props[o.id]}
    for o in state.objects:
        if o.is_text:
            grid.setdefault((o.x, o.y), []).append(o.word)
        elif o.id in word_objects:
            grid.setdefault((o.x, o.y), []).append(o.name)
    rules: list[Rule] = []
    for dx, dy in ((1, 0), (0, 1)):
        for x, y in sorted(grid):
            if (x - dx, y - dy) in grid:
                continue  # not the start of a line
            cells, cx, cy = [], x, y
            while (cx, cy) in grid:
                cells.append(grid[(cx, cy)])
                cx, cy = cx + dx, cy + dy
            seen: set[tuple[Any, ...]] = set()
            for words in _readings(cells):
                for rule in _parse_line(words):
                    if _rule_key(rule) not in seen:
                        seen.add(_rule_key(rule))
                        rules.append(rule)
    return rules


def _reparse(state: State) -> None:
    """Read the rules again; WORD objects can spell rules, so read until the rules agree."""
    for _ in range(4):
        rules = parse_rules(state)
        same = [_rule_key(r) for r in rules] == [_rule_key(r) for r in state.rules]
        state.rules = rules
        if same or not any(r.target == "word" for r in rules):
            return


# ----------------------------------------------------------------------- who has what


def _noun(o: Obj) -> str:
    return o.word if o.is_text else o.name


def _nouns_in_level(state: State) -> list[str]:
    """ALL: every noun present as an object or a word (not TEXT, LEVEL, EMPTY, ALL, GROUP)."""
    special = {TEXT, LEVEL, EMPTY, ALL, GROUP, EDGE}
    nouns = {o.name for o in state.objects if not o.is_text}
    nouns |= {o.word for o in state.objects if o.is_text and _is_noun(o.word)}
    return sorted(nouns - special)


class _View:
    """Who is where, and who is in the GROUP, for one evaluation of the rules."""

    def __init__(self, state: State) -> None:
        self.state = state
        self.cells: dict[tuple[int, int], list[Obj]] = {}
        for o in state.objects:
            self.cells.setdefault((o.x, o.y), []).append(o)
        self.group: set[int] = set()
        group_rules = [r for r in state.rules if r.verb == "is" and r.target == GROUP]
        if group_rules:
            members = {o.id for r in group_rules if not r.target_not for o in self._covered(r)}
            denied = {o.id for r in group_rules if r.target_not for o in self._covered(r)}
            self.group = members - denied

    def _covered(self, rule: Rule) -> list[Obj]:
        return [o for o in self.state.objects if self.covers(rule, o)]

    def is_(self, o: Obj, noun: str) -> bool:
        """Whether ``o`` is what ``noun`` names (in a condition or as a subject)."""
        if o.name == EDGE:
            return False
        if noun == TEXT:
            return o.is_text
        if noun == ALL:
            return not o.is_text and o.name != EMPTY
        if noun == GROUP:
            return o.id in self.group
        return not o.is_text and o.name == noun

    def covers(self, rule: Rule, o: Obj) -> bool:
        if o.name == EDGE or rule.subject == LEVEL:
            return False
        if rule.subject_not:
            if o.is_text or o.name == EMPTY or self.is_(o, rule.subject):
                return False
        elif not self.is_(o, rule.subject):
            return False
        return all(self.holds(c, o) for c in rule.conditions)

    def holds(self, c: Condition, o: Obj) -> bool:
        if c.kind == "facing":
            dx, dy = STEP[o.dir]
            ahead = self.cells.get((o.x + dx, o.y + dy), [])
            met = all(
                (o.dir == DIRECTIONS[n]) != negated
                if n in DIRECTIONS
                else any(self.is_(b, n) for b in ahead) != negated
                for n, negated in c.args
            )
        else:
            if c.kind == "on":
                around = [b for b in self.cells.get((o.x, o.y), []) if b is not o]
            else:
                around = [
                    b
                    for dx in (-1, 0, 1)
                    for dy in (-1, 0, 1)
                    for b in self.cells.get((o.x + dx, o.y + dy), [])
                    if b is not o
                ]
            met = all(any(self.is_(b, n) for b in around) != negated for n, negated in c.args)
        return met != c.negated


def object_properties(state: State) -> tuple[dict[int, Props], Props]:
    """Each object's properties under the current rules (NOT wins over a plain rule), and the
    level's own (LEVEL IS ...). Conditions are checked against the board as it is now."""
    view = _View(state)
    plain: dict[tuple[str, bool], tuple[Props, set[str]]] = {}  # from unconditional rules
    conditional = []
    prop_rules = [r for r in state.rules if r.verb == "is" and r.target in PROPERTIES]
    level: Props = {}
    level_denied = set()
    for r in prop_rules:
        if r.subject == LEVEL and not r.subject_not:
            if r.target_not:
                level_denied.add(r.target)
            else:
                level[r.target] = level.get(r.target, 0) + 1
        elif r.conditions:
            conditional.append(r)
    out: dict[int, Props] = {}
    for o in state.objects:
        if o.name == EDGE:
            out[o.id] = {}
            continue
        key = (o.name, o.id in view.group)  # objects alike share their unconditional rules
        if key not in plain:
            have: Props = {"push": 1} if o.is_text else {}  # the game's base rule TEXT IS PUSH
            denied: set[str] = set()
            for r in prop_rules:
                if not r.conditions and view.covers(r, o):
                    if r.target_not:
                        denied.add(r.target)
                    else:
                        have[r.target] = have.get(r.target, 0) + 1
            plain[key] = (have, denied)
        have, denied = plain[key]
        props = dict(have)
        if conditional:
            denied = set(denied)
            for r in conditional:
                if view.covers(r, o):
                    if r.target_not:
                        denied.add(r.target)
                    else:
                        props[r.target] = props.get(r.target, 0) + 1
        out[o.id] = {p: n for p, n in props.items() if p not in denied}
    return out, {p: n for p, n in level.items() if p not in level_denied}


# --------------------------------------------------------------------- transformations


def _resolve(target: str, o: Obj) -> str | None:
    """The object name a rule's target makes of ``o``; None for targets that name no object."""
    if target == TEXT:
        return None if o.is_text else "text_" + o.name
    if target in (LEVEL, GROUP, ALL, EMPTY):
        return None
    return target


def _transform(state: State, only: set[int] | None = None) -> bool:
    """Apply X IS Y (and X IS TEXT / EMPTY / ALL, EMPTY IS Y) once; True if anything changed."""
    rules = [
        r
        for r in state.rules
        if r.verb == "is" and r.target not in PROPERTIES and r.target != GROUP
    ]
    if not rules:
        return False
    view = _View(state)
    nouns = _nouns_in_level(state)
    changes: list[tuple[Obj, list[str]]] = []
    for o in state.objects:
        if o.name == EDGE or (only is not None and o.id not in only):
            continue
        mine = [r for r in rules if view.covers(r, o)]
        if not mine:
            continue
        if any(r.target == r.subject and not r.target_not for r in mine):
            continue  # X IS X: the object keeps its kind
        if any(r.target == _noun(o) and not r.target_not and not o.is_text for r in mine):
            continue
        denied = {r.target for r in mine if r.target_not}
        targets: list[str] = []
        for r in mine:
            if r.target_not or r.target in denied:
                continue
            if r.target == ALL:
                targets.extend(nouns)
            elif r.target == EMPTY:
                targets.append(EMPTY)
            else:
                name = _resolve(r.target, o)
                if name is not None:
                    targets.append(name)
        targets = list(dict.fromkeys(targets))
        if o.name in targets:  # X IS ALL: it stays itself and the other kinds appear beside it
            targets.remove(o.name)
            targets.insert(0, o.name)
        if targets and targets != [o.name]:
            changes.append((o, targets))
    spawned: list[Obj] = []
    removed: set[int] = set()
    for o, targets in changes:
        first, *rest = targets
        for name in rest:
            if name != EMPTY:
                spawned.append(Obj(state.next_id, name, o.x, o.y, o.dir))
                state.next_id += 1
        if first == EMPTY:
            removed.add(o.id)
        else:
            o.name = first
    empty_rules = [
        r
        for r in rules
        if r.subject == EMPTY and not r.subject_not and not r.target_not and not r.conditions
    ]
    if empty_rules and not any(r.target == EMPTY for r in empty_rules):
        filled = {(o.x, o.y) for o in state.objects}
        for y in range(state.height):
            for x in range(state.width):
                if (x, y) in filled:
                    continue
                for r in empty_rules:
                    if r.target not in (TEXT, LEVEL, GROUP, ALL):
                        spawned.append(Obj(state.next_id, r.target, x, y, DOWN))
                        state.next_id += 1
    if removed:
        state.objects = [o for o in state.objects if o.id not in removed]
    state.objects.extend(spawned)
    return bool(changes or spawned)


def _settle_rules(state: State, only: set[int] | None = None) -> None:
    """Read the rules, transform once, read them again. A rule a transformation spells only
    acts next turn (the game stopped chaining them in 2020)."""
    _reparse(state)
    if _transform(state, only):
        _reparse(state)


# ------------------------------------------------------------------------------ movement
# Follows the Baba Is You wiki's "Order of Operations": units move in object-list order, and a
# move into a tile checks OPEN/SHUT, then SWAP, then WEAK, then PUSH, then STOP and PULL.


class _Turn:
    """The properties of one movement or interaction stage (the game reads them per stage)."""

    def __init__(self, state: State) -> None:
        self.state = state
        self.props, self.level = object_properties(state)
        self.doomed: set[int] = set()
        self.moved: dict[int, int] = {}  # who moved which way this round (pushed or on its own)
        self.moving: set[int] = set()  # the push chain being resolved

    def p(self, o: Obj) -> Props:
        return self.props.get(o.id, {})

    def floats(self, o: Obj) -> bool:
        return "float" in self.p(o)

    def solid(self, o: Obj) -> bool:
        return o.name == EDGE or bool({"stop", "push", "pull"} & self.p(o).keys())

    def opens(self, a: Obj, b: Obj) -> bool:
        pa, pb = self.p(a), self.p(b)
        return ("open" in pa and "shut" in pb) or ("shut" in pa and "open" in pb)

    def at(self, x: int, y: int, but: Obj | None = None) -> list[Obj]:
        return [
            t
            for t in self.state.objects
            if t.x == x and t.y == y and t is not but and t.id not in self.doomed
        ]

    def try_move(self, o: Obj, d: int) -> bool:
        """Move ``o`` one step in direction ``d``, pushing what is in the way and pulling what
        is behind; False if blocked. Objects destroyed on contact (OPEN/SHUT, a WEAK object
        that cannot move) go to ``doomed``."""
        self.moving.add(o.id)
        try:
            return self._try_move(o, d)
        finally:
            self.moving.discard(o.id)

    def _try_move(self, o: Obj, d: int) -> bool:
        state = self.state
        dx, dy = STEP[d]
        nx, ny = o.x + dx, o.y + dy
        if not (0 <= nx < state.width and 0 <= ny < state.height):
            return False
        ahead = self.at(nx, ny, but=o)
        for t in ahead:
            if t.name != EDGE and (self.solid(t) or t.name == EMPTY) and self.opens(o, t):
                self.doomed.update((o.id, t.id))  # a key into a door: both are gone
                return True
        if any(t.name == EDGE for t in ahead):
            return False
        swaps = []
        for t in ahead:
            p_t = self.p(t)
            if "swap" in self.p(o) or "swap" in p_t:
                swaps.append(t)
                continue
            if (
                "weak" in p_t
                and "stop" in p_t
                and "push" not in p_t
                and self.floats(t) == self.floats(o)
            ):
                continue  # a weak solid object does not block: it breaks on the later overlap check
            if "push" in p_t:
                if self.moved.get(t.id) == d:
                    continue  # already pushed this way with its stack: it moves once
                if not self.try_move(t, d):
                    if "weak" in p_t:
                        self.doomed.add(t.id)  # a weak object that cannot be pushed breaks instead
                        continue
                    return False
                continue
            if "stop" in p_t or "pull" in p_t:
                return False
        ox, oy = o.x, o.y
        o.x, o.y = nx, ny
        o.dir = d
        self.moved[o.id] = d
        for t in swaps:
            t.x, t.y = ox, oy
        for t in self.at(ox - dx, oy - dy):
            if "pull" not in self.p(t) or t.name == EDGE or t.id in self.moving:
                continue  # what is itself being pushed along is not pulled as well
            if self.moved.get(t.id) != d:
                self.try_move(t, d)
        return True


def _add_empties(state: State) -> bool:
    """While EMPTY has properties (EMPTY IS YOU / PUSH / OPEN, EMPTY HAS ...), every free tile
    holds an "empty" object for one movement stage; the game recomputes them after each step."""
    if not any(
        r.subject == EMPTY
        and not r.subject_not
        and (r.verb == "has" or (r.verb == "is" and r.target in PROPERTIES))
        for r in state.rules
    ):
        return False
    filled = {(o.x, o.y) for o in state.objects}
    for y in range(state.height):
        for x in range(state.width):
            if (x, y) not in filled:
                state.objects.append(Obj(state.next_id, EMPTY, x, y))
                state.next_id += 1
    return True


def _with_empties(stage: Any, state: State, *args: Any) -> None:
    added = _add_empties(state)
    stage(state, *args)
    if added:
        state.objects = [o for o in state.objects if o.name != EMPTY]


def _move_you(state: State, direction: int) -> None:
    turn = _Turn(state)
    pending = [o for o in state.objects if "you" in turn.p(o) and "sleep" not in turn.p(o)]
    for o in pending:
        o.dir = direction
    while pending:  # a mover blocked by another mover tries again once that one has moved
        blocked = []
        for o in pending:
            if o.id in turn.doomed or turn.moved.get(o.id) == direction:
                continue  # pushed this way by another mover already: a row of movers moves as one
            if not turn.try_move(o, direction):
                blocked.append(o)
        if len(blocked) == len(pending):
            break
        pending = blocked
    for o in pending:
        if "weak" in turn.p(o) and turn.moved.get(o.id) != direction:
            turn.doomed.add(o.id)
    _destroy(state, turn.doomed)


def _move_move(state: State) -> None:
    """MOVE objects step forward, turning around once when blocked; stacked MOVE steps again."""
    turn = _Turn(state)
    movers = [o for o in state.objects if "move" in turn.p(o) and "sleep" not in turn.p(o)]
    turned: set[int] = set()
    for round_ in range(max((turn.p(o)["move"] for o in movers), default=0)):
        turn.moved = {}
        for o in movers:
            if o.id in turn.doomed or turn.p(o)["move"] <= round_:
                continue
            if turn.moved.get(o.id) == o.dir:
                continue
            if turn.try_move(o, o.dir):
                continue
            if o.id not in turned:
                turned.add(o.id)
                o.dir = (o.dir + 2) % 4
                if turn.try_move(o, o.dir):
                    continue
            if "weak" in turn.p(o):
                turn.doomed.add(o.id)
    _destroy(state, turn.doomed)


def _move_shift(state: State) -> None:
    """Each SHIFT object carries the other same-float objects on its tile the way it faces."""
    turn = _Turn(state)
    plan: dict[int, list[int]] = {}  # object id -> [direction, steps]
    for s in [o for o in state.objects if "shift" in turn.p(o)]:
        for t in turn.at(s.x, s.y, but=s):
            if t.name == EDGE or turn.floats(t) != turn.floats(s):
                continue
            if t.id in plan:
                plan[t.id][1] += turn.p(s)["shift"]  # the first SHIFT sets the direction
            else:
                plan[t.id] = [s.dir, turn.p(s)["shift"]]
    if not plan:
        return
    by_id = {o.id: o for o in state.objects}
    for round_ in range(max(steps for _, steps in plan.values())):
        turn.moved = {}
        for oid, (d, steps) in plan.items():
            if steps > round_ and oid not in turn.doomed and turn.moved.get(oid) != d:
                o = by_id[oid]
                if not turn.try_move(o, d) and "weak" in turn.p(o):
                    turn.doomed.add(o.id)
    _destroy(state, turn.doomed)


def _tele(state: State) -> None:
    """Objects on a TELE object go to another object of its kind (at random among several)."""
    turn = _Turn(state)
    kinds: dict[str, list[Obj]] = {}
    for o in state.objects:
        if "tele" in turn.p(o):
            kinds.setdefault(o.name, []).append(o)
    rng = random.Random(f"{state.seed}:{state.turn}")
    moved: set[int] = set()
    for name, pads in kinds.items():
        if len(pads) < 2:
            continue
        for pad in pads:
            px, py = pad.x, pad.y
            for t in turn.at(px, py, but=pad):
                if t.name == EDGE or t.name == name or t.id in moved:
                    continue
                if turn.floats(t) != turn.floats(pad):
                    continue
                others = [q for q in pads if q is not pad]
                dest = others[0] if len(others) == 1 else rng.choice(others)
                t.x, t.y = dest.x, dest.y
                moved.add(t.id)


def _shift_turn(state: State) -> None:
    """After moving, an object on a SHIFT object faces the way it faces."""
    turn = _Turn(state)
    shifts = [o for o in state.objects if "shift" in turn.p(o)]
    if not shifts:
        return
    turned: set[int] = set()
    for s in shifts:
        for t in turn.at(s.x, s.y, but=s):
            if t.name != EDGE and t.id not in turned and turn.floats(t) == turn.floats(s):
                t.dir = s.dir
                turned.add(t.id)


def _fall(state: State) -> None:
    """FALL objects drop until something solid (or the edge) is below; they push nothing."""
    turn = _Turn(state)
    for o in [o for o in state.objects if "fall" in turn.p(o)]:
        while o.id not in turn.doomed:
            below = o.y + 1
            if below >= state.height:
                break
            under = turn.at(o.x, below, but=o)
            if any(t.name == EDGE for t in under):
                break
            shut = [t for t in under if turn.solid(t) and turn.opens(o, t)]
            if shut:
                turn.doomed.update((o.id, shut[0].id))
                break
            if any(turn.solid(t) for t in under):
                break
            o.y = below
    _destroy(state, turn.doomed)


def _face(state: State) -> None:
    """UP / DOWN / LEFT / RIGHT turn an object; among several, the most given wins, then the
    first one clockwise from where it faces."""
    turn = _Turn(state)
    for o in state.objects:
        given = {DIRECTIONS[p]: n for p, n in turn.p(o).items() if p in DIRECTIONS}
        if not given:
            continue
        best = max(given.values())
        d = o.dir
        for _ in range(4):
            if given.get(d) == best:
                o.dir = d
                break
            d = (d - 1) % 4  # clockwise: right, down, left, up


# ---------------------------------------------------------------------------- interactions


def _destroy(state: State, doomed: set[int]) -> None:
    if not doomed:
        return
    has = [r for r in state.rules if r.verb == "has" and not r.target_not]
    view = _View(state) if has else None
    survivors, spawned = [], []
    for o in state.objects:
        if o.id not in doomed:
            survivors.append(o)
            continue
        for rule in has:
            assert view is not None
            if view.covers(rule, o):
                name = _resolve(rule.target, o)
                if name is not None:
                    spawned.append(Obj(state.next_id, name, o.x, o.y, o.dir))
                    state.next_id += 1
    state.objects = survivors + spawned


def _more(state: State, turn: _Turn) -> None:
    """A MORE object copies itself onto each free side (no copy there, nothing solid)."""
    for o in [o for o in state.objects if "more" in turn.p(o)]:
        for dx, dy in STEP.values():
            x, y = o.x + dx, o.y + dy
            if not (0 <= x < state.width and 0 <= y < state.height):
                continue
            there = turn.at(x, y)
            if any(t.name == o.name or turn.solid(t) for t in there):
                continue
            copy_ = Obj(state.next_id, o.name, x, y, o.dir)
            state.next_id += 1
            state.objects.append(copy_)
            turn.props[copy_.id] = dict(turn.p(o))


def _make(state: State) -> None:
    rules = [r for r in state.rules if r.verb == "make" and not r.target_not]
    if not rules:
        return
    view = _View(state)
    for r in rules:
        for o in [o for o in state.objects if view.covers(r, o)]:
            name = _resolve(r.target, o)
            if name is None or any(t.name == name for t in view.cells.get((o.x, o.y), [])):
                continue
            made = Obj(state.next_id, name, o.x, o.y, o.dir)
            state.next_id += 1
            state.objects.append(made)
            view.cells.setdefault((o.x, o.y), []).append(made)


# Same-tile interactions, in the game's order; each step deletes before the next one looks.
PHASES = ("more", "sink", "weak", "melt", "defeat", "open", "make", "bonus")


def _phase(state: State, kind: str, turn: _Turn) -> set[int]:
    p = turn.p
    level = turn.level  # the level is on every tile at once, on its own float
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
        on_level = level if level_float == floats else {}
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
        elif kind == "open" and ({"open", "shut"} & mine.keys()):
            if {"open", "shut"} <= mine.keys():
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
        turn = _Turn(state)
        if kind == "more":
            _more(state, turn)
        elif kind == "make":
            _make(state)
        else:
            _destroy(state, _phase(state, kind, turn))


def _check_win(state: State) -> bool:
    props, level = object_properties(state)
    level_float = "float" in level
    for o in state.objects:
        mine = props[o.id]
        if "you" not in mine:
            continue
        floats = "float" in mine
        if "win" in level and level_float == floats:
            return True
        for t in state.objects:
            t_props = props[t.id]
            if t.x == o.x and t.y == o.y and "win" in t_props and ("float" in t_props) == floats:
                return True
    if "you" in level:  # the level is YOU: any same-float WIN object wins
        return any(
            "win" in props[t.id] and ("float" in props[t.id]) == level_float for t in state.objects
        )
    return False


# ----------------------------------------------------------------------------------- turn


def step(state: State, move: str) -> State:
    """The state after one input: ``l``/``r``/``u``/``d`` or ``p`` (wait). Pure."""
    s = replace(state, objects=[replace(o) for o in state.objects], rules=list(state.rules))
    if s.won:
        return s
    s.turn += 1
    first_new = s.next_id
    if move != WAIT:
        _with_empties(_move_you, s, INPUT_DIR[move])
    _with_empties(_move_move, s)
    _with_empties(_move_shift, s)
    _settle_rules(s)
    _tele(s)
    _shift_turn(s)
    _fall(s)
    _reparse(s)
    _face(s)
    _fall(s)
    _reparse(s)
    _interactions(s)
    s.won = _check_win(s)  # in the same step as the interactions, before the rules are re-read
    _settle_rules(s, only={o.id for o in s.objects if o.id >= first_new})  # made this turn
    return s


def play(level: dict[str, Any], moves: str, seed: int = 0) -> tuple[State, int | None]:
    """Replay ``moves``; the 1-based index of the move that won, or None."""
    state = load(level, seed)
    for i, move in enumerate(moves, start=1):
        state = step(state, move)
        if state.won:
            return state, i
    return state, None
