"""Build ``problems/grid_games/data/babaisai_packs.json`` (needs ``make install-babaisai``).

Usage:
    python scripts/grid_games/build_babaisai_packs.py [--seed 0] [--budget 300]

A level is an engine environment id plus a seed. Candidates are drawn in turn from each skill
family of ``FAMILIES`` (the engine's 43 generators, grouped; the one-move ``you_win`` ones are left
out) and solved by breadth-first search on the real engine. A candidate is cut when the search
does not solve it within the budget, when it is solved in fewer than ``MIN_OPTIMAL`` moves, or
when an object starts hidden under another; ``selection`` in the file counts these per family. The
first ``POOL`` levels kept are dealt into packs of ``PACK_SIZE`` (sorted by shortest solution,
dealt in snake order, each pack ordered by length), so every pack has the same spread. Every plan
is replayed through regact's environment before the file is written; search results are cached
across builds. The code table lists every (type, colour-or-word) pair seen on top of a cell in
any searched state, with a colour for the video; ``empty`` is code 0.
BabaIsAI: https://github.com/nacloos/baba-is-ai (MIT)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
import warnings
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from _common import SolutionCache, deal  # noqa: E402

from regact.problems.grid_games import babaisai  # noqa: E402
from regact.problems.grid_games.base import GridGameEnv  # noqa: E402

OUT = Path(babaisai.__file__).parent / "data" / "babaisai_packs.json"
# Skill family -> generator-id prefixes (after "env/"); two-room ids carry "two_room-".
FAMILIES = {
    "goto_win": ("goto_win",),
    "make_win": ("make_win",),
    "two_room-goto_win": ("two_room-goto_win", "two_room-maybe_break_stop-goto_win"),
    "two_room-break_stop-goto_win": ("two_room-break_stop-goto_win",),
    "two_room-make_you": ("two_room-make_you",),
    "two_room-make_win": ("two_room-make_win", "two_room-make_wall_win"),
    "two_room-break_stop-make_win": ("two_room-break_stop-make_win",),
}
VERSION = "babaisai-bench-v1"
MIN_OPTIMAL = 3
POOL = 64
PACK_SIZE = 8
WORKERS = 2  # nice 19; the laptop also serves audio
# Engine colour names -> ARC palette ids, for the video.
OBJECT_COLOUR = {
    "red": 8,
    "green": 14,
    "blue": 9,
    "purple": 15,
    "yellow": 11,
    "grey": 3,
    "white": 1,
}
WORD_COLOURS = (6, 12, 10, 7, 13, 2, 11, 14, 9, 8, 15, 3)


def digest(env) -> bytes:
    """The full state (every object of every cell, top included), hashed."""
    stacks = tuple(tuple(o.encode() for o in cell if o is not None) for cell in env.grid.grid)
    return hashlib.blake2b(repr(stacks).encode(), digest_size=16).digest()


def solve(env_id: str, seed: int, budget: float, seen_pairs: set) -> list[int] | None:
    """A shortest plan in our action ids, or None; adds every top-of-cell pair seen.

    Breadth-first, storing a digest per state and a parent pointer per node: a node's state is
    rebuilt by replaying its path, which keeps memory flat (a stored engine copy per node does not).
    """

    def fresh():
        env = babaisai.make_engine(env_id, seed)
        env._np_random = None  # gym 0.25's generator: no deepcopy under numpy 2; unused after reset
        return env

    parents, moves = [-1], [0]
    seen, queue, t0 = {digest(fresh())}, deque([0]), time.time()
    actions = list(babaisai.ENGINE_ACTION.items())
    while queue:
        current = queue.popleft()
        path, node = [], current
        while node > 0:
            path.append(moves[node])
            node = parents[node]
        path.reverse()
        base = fresh()
        for action in path:
            base.step(babaisai.ENGINE_ACTION[action])
        seen_pairs.update(cell for row in babaisai.encoded_rows(base) for cell in row)
        for i, (action, engine_action) in enumerate(actions):
            nxt = base if i == len(actions) - 1 else copy.deepcopy(base)
            _, reward, done, _ = nxt.step(engine_action)
            if done and reward > 0:
                return [*path, action]
            key = digest(nxt)
            if key not in seen:
                seen.add(key)
                parents.append(current)
                moves.append(action)
                queue.append(len(moves) - 1)
        if time.time() - t0 > budget:
            return None
    return None


def hidden_at_reset(env_id: str, seed: int) -> bool:
    """Whether some object starts hidden under another (walls stacked on walls do not count)."""
    env = babaisai.make_engine(env_id, seed)
    for cell in env.grid.grid:
        objects = [o for o in cell if o is not None]
        if len(objects) > 1 and not all(o.type in ("wall", "fwall") for o in objects):
            return True
    return False


def kind(pair: tuple[int, int]) -> dict:
    from baba import world_object as w

    # Invert the live tables: baba adds the word ids to COLOR_TO_IDX after building IDX_TO_COLOR.
    type_name = {i: n for n, i in w.OBJECT_TO_IDX.items()}[pair[0]]
    value_name = {i: n for n, i in w.COLOR_TO_IDX.items()}[pair[1]]
    word = type_name.startswith("rule")
    if word:
        colour = WORD_COLOURS[pair[1] % len(WORD_COLOURS)]
    elif type_name == "empty":
        colour = 5
    elif type_name == "wall" and value_name == "grey":
        colour = 4
    else:
        colour = OBJECT_COLOUR.get(value_name, 13)
    name = f"{type_name}:{value_name}"
    return {"type": pair[0], "value": pair[1], "name": name, "word": word, "colour": colour}


def family_of(generator: str) -> str | None:
    name = generator.removeprefix("env/")
    matches = [
        (len(prefix), family)
        for family, prefixes in FAMILIES.items()
        for prefix in prefixes
        if name == prefix or name.startswith(prefix + "-")
    ]
    return max(matches)[1] if matches else None


def search(job: tuple[str, int, float]) -> dict:
    env_id, seed, budget = job
    pairs: set = set()
    plan = solve(env_id, seed, budget, pairs)
    return {"plan": plan, "exhausted": False, "pairs": sorted(pairs)}


def candidates(seed: int) -> list[tuple[str, str, int]]:
    """(family, generator, seed) drawn in turn from each family; far more than ever needed."""
    from baba import registration

    generators = sorted(i for i in registration.registry if "#" not in i)
    pools = {f: [g for g in generators if family_of(g) == f] for f in FAMILIES}
    rng, out, seen = random.Random(seed), [], set()
    while len(out) < 40 * POOL:
        for family, pool in pools.items():
            level = (rng.choice(pool), rng.randrange(10_000))
            if level not in seen:
                seen.add(level)
                out.append((family, *level))
    return out


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--budget", type=float, default=300, help="search seconds per level")
    opts = args.parse_args()
    cache, kept, pairs = SolutionCache("babaisai"), [], set()
    selection: dict[str, Counter] = {family: Counter() for family in FAMILIES}
    queue = candidates(opts.seed)
    with ProcessPoolExecutor(WORKERS) as workers:
        while len(kept) < POOL:
            batch, queue = queue[:WORKERS], queue[WORKERS:]
            pending = []
            for family, env_id, seed in batch:
                selection[family]["drawn"] += 1
                if hidden_at_reset(env_id, seed):
                    selection[family]["hidden object"] += 1
                    continue
                job = (env_id, seed, opts.budget)
                known = cache.cached(f"{env_id}@{seed}", opts.budget)
                pending.append((family, env_id, seed, known or workers.submit(search, job)))
            for family, env_id, seed, result in pending:  # in draw order: reproducible
                search_ = (lambda r=result: r) if isinstance(result, dict) else result.result
                entry = cache.solve(f"{env_id}@{seed}", opts.budget, search_)
                pairs.update(tuple(pair) for pair in entry["pairs"])
                plan = entry["plan"]
                if plan is None:
                    selection[family]["unsolved in budget"] += 1
                elif len(plan) < MIN_OPTIMAL:
                    selection[family]["too short"] += 1
                elif len(kept) < POOL:
                    selection[family]["kept"] += 1
                    level = {"env_id": env_id, "seed": seed, "family": family}
                    kept.append({**level, "optimal": len(plan), "plan": plan})
                    print(f"{len(kept)}/{POOL} {env_id}@{seed} optimal {len(plan)}", flush=True)
    packs = [
        {"key": f"babaisai_p{i + 1:02d}", "levels": levels}
        for i, levels in enumerate(deal(kept, POOL // PACK_SIZE))
    ]
    empty = next(pair for pair in pairs if kind(pair)["name"].startswith("empty"))
    table = [kind(empty)] + [kind(pair) for pair in sorted(pairs - {empty})]
    summary = {family: dict(counts) for family, counts in selection.items()}
    OUT.write_text(
        json.dumps(
            {
                "version": VERSION,
                "source": "https://github.com/nacloos/baba-is-ai (MIT), commit 40ae1b2",
                "built_by": "scripts/grid_games/build_babaisai_packs.py",
                "seed": opts.seed,
                "search_budget_s": opts.budget,
                "selection": summary,
                "codes": table,
                "packs": packs,
            },
            indent=1,
        )
        + "\n"
    )
    babaisai._load.cache_clear()
    problem = babaisai.BabaIsAIProblem()
    for pack in packs:  # every stored plan must solve its level through the regact environment
        for level in pack["levels"]:
            game = babaisai.BabaGame(((level["env_id"], level["seed"]),), babaisai._load()[1])
            env = GridGameEnv(game, n_levels=1, actions=problem.actions)
            env.reset()
            outcomes = [env.step(a)[2] for a in level["plan"]]
            assert outcomes[-1] and not any(outcomes[:-1]), (pack["key"], level["env_id"])
    print(f"wrote {OUT} ({VERSION}, {len(table)} codes); selection: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
