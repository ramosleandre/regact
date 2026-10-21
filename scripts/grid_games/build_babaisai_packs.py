"""Build ``problems/grid_games/data/babaisai_packs.json`` (needs ``make install-babaisai``).

Usage:
    python scripts/grid_games/build_babaisai_packs.py [--packs 8] [--seed 0]

Each pack holds 3 single-room and 4 two-room levels (an engine environment id plus a seed),
ordered by shortest solution. Every level is solved by breadth-first search on the real engine
and its plan is replayed through regact's grid-game environment before it is kept; levels not
solved within the search budget, solved in fewer than ``MIN_OPTIMAL`` moves, or with an object
hidden under another at reset, are skipped.
The code table lists every (type, colour-or-word) pair seen on top of a cell in any searched
state, with a colour for the video; ``empty`` is code 0.
BabaIsAI: https://github.com/nacloos/baba-is-ai (MIT).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
import warnings
from collections import deque
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from regact.problems.grid_games import babaisai  # noqa: E402
from regact.problems.grid_games.base import GridGameEnv  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src/regact/problems/grid_games/data/babaisai_packs.json"
MIX = (("single", 3), ("two_room", 4))
MAX_SECONDS = 20.0
MIN_OPTIMAL = 3
# Engine colour names -> ARC palette ids, for the video.
OBJECT_COLOUR = {"red": 8, "green": 14, "blue": 9, "purple": 15, "yellow": 11, "grey": 3, "white": 1}
WORD_COLOURS = (6, 12, 10, 7, 13, 2, 11, 14, 9, 8, 15, 3)


def stacks(env) -> tuple:
    """The full state: every object of every cell, top included."""
    return tuple(tuple(o.encode() for o in cell if o is not None) for cell in env.grid.grid)


def solve(env_id: str, seed: int, seen_pairs: set) -> list[int] | None:
    """A shortest plan in our action ids, or None; adds every top-of-cell pair seen."""
    start = babaisai.make_engine(env_id, seed)
    start._np_random = None  # gym 0.25's generator does not deepcopy under numpy 2; unused after reset
    queue, seen, t0 = deque([(start, [])]), {stacks(start)}, time.time()
    while queue:
        env, plan = queue.popleft()
        seen_pairs.update(cell for row in babaisai.encoded_rows(env) for cell in row)
        for action, engine_action in babaisai.ENGINE_ACTION.items():
            nxt = copy.deepcopy(env)
            _, reward, done, _ = nxt.step(engine_action)
            if done and reward > 0:
                return plan + [action]
            key = stacks(nxt)
            if key not in seen:
                seen.add(key)
                queue.append((nxt, plan + [action]))
        if time.time() - t0 > MAX_SECONDS:
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
    return {"type": pair[0], "value": pair[1], "name": f"{type_name}:{value_name}", "word": word, "colour": colour}


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--packs", type=int, default=8)
    args.add_argument("--seed", type=int, default=0)
    opts = args.parse_args()
    from baba import registration

    ids = sorted(i for i in registration.registry if "#" not in i)
    pools = {
        "single": [i for i in ids if "two_room" not in i],
        "two_room": [i for i in ids if "two_room" in i],
    }
    rng, used, pairs, packs = random.Random(opts.seed), set(), set(), []
    for p in range(opts.packs):
        levels = []
        for pool, count in MIX:
            found = 0
            while found < count:
                env_id, seed = rng.choice(pools[pool]), rng.randrange(10_000)
                if (env_id, seed) in used or hidden_at_reset(env_id, seed):
                    continue
                used.add((env_id, seed))
                plan = solve(env_id, seed, pairs)
                if plan is None or len(plan) < MIN_OPTIMAL:
                    continue
                levels.append({"env_id": env_id, "seed": seed, "optimal": len(plan), "plan": plan})
                found += 1
                print(f"pack {p + 1}: {env_id}@{seed} optimal {len(plan)}", flush=True)
        levels.sort(key=lambda level: level["optimal"])
        packs.append({"key": f"babaisai_p{p + 1:02d}", "levels": levels})
    empty = next(pair for pair in pairs if kind(pair)["name"].startswith("empty"))
    table = [kind(empty)] + [kind(pair) for pair in sorted(pairs - {empty})]
    OUT.write_text(
        json.dumps(
            {
                "source": "https://github.com/nacloos/baba-is-ai (MIT), commit 40ae1b2",
                "built_by": "scripts/grid_games/build_babaisai_packs.py",
                "seed": opts.seed,
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
    print(f"wrote {OUT} ({len(table)} codes)")


if __name__ == "__main__":
    main()
