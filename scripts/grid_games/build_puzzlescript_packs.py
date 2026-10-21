"""Build ``problems/grid_games/data/puzzlescript_games.json`` (needs Node and the engine).

Usage:
    python scripts/grid_games/build_puzzlescript_packs.py [--budget 60] [--retry-budget 600]

For each game of ``GAMES`` (puzzlescript.net gallery games that pass the strict filter: no
``random``, ``realtime``, ``again``, ``rigid`` or ``checkpoint``), the first ``MAX_LEVELS``
playable levels in the designer's order are searched breadth-first for a shortest solution, in
order: ``--budget`` seconds, then ``--retry-budget`` for a level not solved. A game with a level
still unsolved is left out (``dropped``), so no game is cut short. A level shown to have no
solution at all (an ending screen drawn as a level) is not a level and is skipped. A game needs
``MIN_LEVELS`` levels. Every plan is replayed through regact's environment before the file is
written; search results are cached across builds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from _common import SolutionCache

from regact.problems.grid_games import puzzlescript as ps
from regact.problems.grid_games.base import GridGameEnv

OUT = ps._GAMES
VERSION = "puzzlescript-bench-v1"
BFS = Path(__file__).with_name("puzzlescript_bfs.js")
MAX_LEVELS = 10
MIN_LEVELS = 3
WORKERS = 2  # nice 19; the laptop also serves audio
_GIST = "https://gist.githubusercontent.com"
# (title, pinned source URL, SHA-256 of the source); the title is for the docs, never the agent.
GAMES = (
    (
        "Microban",
        f"{_GIST}/anonymous/6841219/raw/f18eda4277c7d75e353ec544d934de43c32674ca/script.txt",
        "f3b2815e76d4192a325f069e50dc5bd1d5aa336e72b79b606320cf30a0b0cc13",
    ),
    (
        "Kettle",
        f"{_GIST}/anonymous/6878681/raw/678615a38d9306d4e31428d3bd71bf703dbd021d/script.txt",
        "df99ad2980afbe2e8b71c11272b54a43cbf7e7f2c9907a7e706e8b9c85ae18ef",
    ),
    (
        "Take Heart Lass",
        f"{_GIST}/anonymous/6902186/raw/f742954232ba84637fd168768c45ec535af5a04d/script.txt",
        "acfdc7fde4c15511b8a2e82cc65a8c368c0dffb96423b93b8391ad6a88991032",
    ),
    (
        "Lime Rick",
        f"{_GIST}/anonymous/6841210/raw/a77d16c7750bc4af92a9d6082a4f57f3c95649cc/script.txt",
        "e8625400924fe39cbe93687fcc3427059c819980fb707b8afc8899ab5ed22901",
    ),
    (
        "Modality",
        f"{_GIST}/anonymous/6888061/raw/ef61308850f9a1eac5a836a4c0e75c8768759984/script.txt",
        "924a27515c6169a07800e79b1c32872773bbe17db60bc40a8794bc59a0d1aad8",
    ),
    (
        "Long Haul Space Flight",
        f"{_GIST}/pap4qlxxlevb/44a295d7a2ab17a2048f555d72781a29/raw/64ea98681d0f66a9adc8e016b96176987a89879e/script.txt",
        "410e2786d68ecd65604bc6e86d62d2fcf800f6fcdfd12997ba32d6d7537b3d22",
    ),
    (
        "Collapsable Sokoban",
        f"{_GIST}/anonymous/fd4e3445b63068676f72/raw/ae01866dfa030732be1a92a1c6d09804527fd6e7/script.txt",
        "e8810a7845ce13d62d3c7418e592eb29437a12ea91bcc561c688567a7021468e",
    ),
    (
        "Love and Pieces",
        f"{_GIST}/anonymous/6846450/raw/c332bb089cdf9b644b8f357bb441d4356e61c81a/script.txt",
        "f4a145c90a6867521c0923ab379b53fa282c88127902773f2a0ca91e2e44331b",
    ),
    (
        "Cake Monsters",
        f"{_GIST}/MattRix/fcf4b43ee6c679ef9389/raw/6a4812d392bd2c2abae6a33d1225a755bb752669/script.txt",
        "2dc20187b2be5d387f36dc9ff25bacbc1a87f8c98acd55e2b3c4ea90470803d2",
    ),
    (
        "Stairways",
        f"{_GIST}/anonymous/21e3784aaa7675dc4b08/raw/403938ee58b3d577d7a006ea2192c2c044b22a72/script.txt",
        "9a196c4cd07bc778b27bf6975bb3a267b6cc07a7f05724cf04acdd0f4fe266d7",
    ),
    (
        "Singleton Traffic",
        f"{_GIST}/Tophwells/eb0f9102f85b5ea536b1/raw/8e977c64f6d6cf79987742b57ffbe553a64ddddb/script.txt",
        "0eb5a411bd0d16e36d6a8532485eda33398183d48401ed6ad108cdc1b73a8438",
    ),
    (
        "Heroes of Sokoban",
        f"{_GIST}/anonymous/6860122/raw/d3e54f0ff2c897bc63ec246b3d61726537511cf1/script.txt",
        "81a1d27115fc40c04f27bb39d0ef12ebffbea513e3fe9f550b5173bb683728a3",
    ),
    (
        "Midas",
        f"{_GIST}/anonymous/6878337/raw/bd5300fc0772ab1e4bd4ae61fb52f9d48728f655/script.txt",
        "1375abffab652e1cacfefa2f151d3d57bb028bca3c0eb3bac522519d9f2ee091",
    ),
    (
        "Soliquid",
        f"{_GIST}/klianc09/4adc7b17c2c0449c3a8b6c730e8cbfc9/raw/60e69002ca96c4ff20102863a6ef3f4bd85ca9cf/script.txt",
        "12494c4bdc0cedd6583f5b3ad48cf3c9125f0981c6eeeedada5d7818eb7d6242",
    ),
    (
        "2D Whale World",
        f"{_GIST}/anonymous/6841165/raw/eae110632df83ac16737955a2b2641cc2b39d4e3/script.txt",
        "d341f7c681d41677f9f85d9cc5196db0b7a3f29408c7d8c8f7bb48f563d1d297",
    ),
    (
        "Smother",
        f"{_GIST}/increpare/6decdfe5107e7f2beee586e01680e850/raw/669974a75d314936ef2e61e3817c26bcc6082299/script.txt",
        "1cb8ab1caf6a507640fa5c739c4c7d0a073479a176aeb1c056a78d8935dae89b",
    ),
    (
        "Manic Ammo",
        f"{_GIST}/anonymous/7057244/raw/4d2c8a71dbf0de7784e4ff222308436d8402b0da/script.txt",
        "66a18f3c19d27587faa98bbaa7bce6dab7fce81751bd26ba21bb6b3b5deda79b",
    ),
)
OUR_ACTION = {engine: ours for ours, engine in ps.ENGINE_DIR.items()}


def node(*args: str, budget_s: float = 30) -> dict:
    out = subprocess.run(
        ["nice", "-n", "19", "node", "--max-old-space-size=3000", str(BFS), *args],
        capture_output=True,
        text=True,
        timeout=budget_s + 120,
        env={**os.environ, "PS_SRC": str(ps.engine_src())},
    )
    if not out.stdout.strip():
        return {"plan": None, "exhausted": False, "error": out.stderr[-300:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


def fetch(url: str, sha256: str) -> Path:
    path = ps.game_source(sha256)
    if not path.is_file():
        import urllib.request

        body = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(body).hexdigest() != sha256:
            raise SystemExit(f"{url} does not match its recorded SHA-256")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return path


def search_game(game: tuple, budgets: tuple[float, float], cache: SolutionCache) -> dict:
    """The game's levels in order; stops at the first one still unsolved after both budgets."""
    title, url, sha256, source, levels, action = game
    kept, skipped = [], []
    for level in levels:
        for budget in budgets:
            entry = cache.solve(
                f"{sha256}:{level}",
                budget,
                lambda budget=budget, level=level: node(
                    str(source), str(level), str(int(budget * 1000)), budget_s=budget
                ),
            )
            if entry["plan"] is not None or entry["exhausted"]:
                break
        if entry["plan"] is not None:
            plan = [OUR_ACTION[d] for d in entry["plan"]]
            kept.append({"index": level, "optimal": len(plan), "plan": plan})
        elif entry["exhausted"]:
            skipped.append({"index": level, "reason": "no solution exists (an ending screen)"})
        else:
            return {"title": title, "dropped": f"level {level} unsolved in {budgets[-1]:.0f} s"}
    if len(kept) < MIN_LEVELS:
        return {"title": title, "dropped": f"only {len(kept)} levels"}
    return {
        "title": title,
        "url": url,
        "sha256": sha256,
        "actions": [1, 2, 3, 4, 5] if action else [1, 2, 3, 4],
        "levels": kept,
        "skipped": skipped,
    }


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--budget", type=float, default=60)
    args.add_argument("--retry-budget", type=float, default=600)
    args.add_argument("--workers", type=int, default=WORKERS)
    opts = args.parse_args()
    games = []
    for title, url, sha256 in GAMES:
        source = fetch(url, sha256)
        info = node(str(source), "info")
        games.append((title, url, sha256, source, info["playable"][:MAX_LEVELS], info["action"]))
    cache = SolutionCache("puzzlescript")
    with ThreadPoolExecutor(opts.workers) as pool:
        results = list(
            pool.map(lambda g: search_game(g, (opts.budget, opts.retry_budget), cache), games)
        )
    out = [r for r in results if "dropped" not in r]
    dropped = [r for r in results if "dropped" in r]
    for r in results:
        print(f"{r['title']}: {r.get('dropped') or [k['optimal'] for k in r['levels']]}")
    for i, game in enumerate(out, start=1):
        game["key"] = f"puzzlescript_{i:02d}"
    OUT.write_text(
        json.dumps(
            {
                "version": VERSION,
                "engine_commit": ps.ENGINE_COMMIT,
                "built_by": "scripts/grid_games/build_puzzlescript_packs.py",
                "search_budget_s": [opts.budget, opts.retry_budget],
                "dropped": dropped,
                "games": [{"key": g.pop("key"), **g} for g in out],
            },
            indent=1,
        )
        + "\n"
    )
    ps._load.cache_clear()
    for game in out:  # every stored plan must solve its level through the regact environment
        for level in game["levels"]:
            env = GridGameEnv(
                ps.PuzzleScriptGame(ps.game_source(game["sha256"]), (level["index"],)),
                n_levels=1,
                actions=tuple(game["actions"]),
            )
            env.reset()
            outcomes = [env.step(a)[2] for a in level["plan"]]
            env.close()
            assert outcomes[-1] and not any(outcomes[:-1]), (game["title"], level["index"])
    print(f"wrote {OUT} ({VERSION}: {len(out)} games, {len(dropped)} dropped)")


if __name__ == "__main__":
    main()
