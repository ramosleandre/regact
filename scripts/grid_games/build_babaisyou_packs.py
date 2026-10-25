"""Build ``problems/grid_games/data/babaisyou_levels.json`` from baba-is-harbor's levels.

Usage:
    python scripts/grid_games/build_babaisyou_packs.py [--source <local clone>]

baba-is-harbor (github.com/stared/baba-is-harbor, Apache-2.0) extracted the campaign levels of
Baba Is You (worlds 0 to 9) from the game's own files. This script pins them by URL and SHA-256
(the level designs are the game's, so they are fetched at install, never redistributed) and
gives each level a reference: the move that wins along a known solution, replayed through
regact's engine.

- Worlds 0-3: harbor's solutions, checked on the real game. A level is kept only if its
  solution wins here too.
- Worlds 4-9: harbor has no solutions; ours (``babaisyou/solutions.json``) were found by
  following each level's solution on the Baba Is You wiki in regact's engine.

One task per world, its levels in the game's numbering, bonus and extra levels last (the
game's own unlock order agrees: they open last).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from regact.problems.grid_games import babaisyou as bbu
from regact.problems.grid_games.babaisyou import engine

VERSION = "babaisyou-bench-v2"
COMMIT = "2adb73ad35d139ab490e2a8513631c43dec16095"
_RAW = f"https://raw.githubusercontent.com/stared/baba-is-harbor/{COMMIT}"
_LEVELS = "tasks_campaign/main-00-09/environment/levels"
WORLDS = {
    "00": "map",
    "01": "the lake",
    "02": "solitary island",
    "03": "temple ruins",
    "04": "forest of fall",
    "05": "deep forest",
    "06": "rocket trip",
    "07": "flower garden",
    "08": "chasm",
    "09": "volcanic cavern",
}
EXCLUDED = {
    "stage-04-level-08": "corrupt at the source: harbor decoded three of its custom words "
    "(LEAF, FOLIAGE, DEFEAT) as objects, so its rules differ from the game's",
}


def _read(source: Path | None, relative: str) -> bytes | None:
    if source is not None:
        path = source / relative
        return path.read_bytes() if path.is_file() else None
    try:
        with urllib.request.urlopen(f"{_RAW}/{relative}", timeout=60) as response:
            body: bytes = response.read()
            return body
    except urllib.error.HTTPError:
        return None


def _harbor_moves(script: bytes) -> str:
    return script.decode().split('echo -n "')[1].split('"')[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=None, help="a local clone at COMMIT")
    args = parser.parse_args()
    names = json.loads(_read(args.source, "level_map.json") or b"{}")["levels"]
    ours = json.loads(bbu.SOLUTIONS.read_text(encoding="utf-8"))
    ids = [n for n in names if n.split("-")[1] in WORLDS]
    packs: dict[str, list[dict[str, object]]] = {world: [] for world in WORLDS}
    rejected = dict(EXCLUDED)
    for lid in sorted(ids, key=lambda t: (t.split("-")[2] != "level", t)):
        level_bytes = _read(args.source, f"{_LEVELS}/{lid}.json")
        if level_bytes is None or lid in EXCLUDED:
            continue
        entry: dict[str, object] = {
            "id": lid,
            "title": names[lid]["title"],
            "url": f"{_RAW}/{_LEVELS}/{lid}.json",
            "sha256": hashlib.sha256(level_bytes).hexdigest(),
        }
        harbor = _read(args.source, f"tasks/{lid}/solution/solve.sh")
        if harbor is not None:
            moves = _harbor_moves(harbor)
            entry |= {
                "solution": "real game",
                "solution_url": f"{_RAW}/tasks/{lid}/solution/solve.sh",
                "solution_sha256": hashlib.sha256(harbor).hexdigest(),
            }
        elif lid in ours:
            moves = ours[lid]
            entry["solution"] = "wiki-guided, this engine"
        else:
            rejected[lid] = "no known solution"
            continue
        _, won = engine.play(json.loads(level_bytes), moves)
        if won is None:
            rejected[lid] = "its solution does not win in this engine"
            continue
        entry["reference"] = won
        packs[lid.split("-")[1]].append(entry)
    data = {
        "version": VERSION,
        "source": f"github.com/stared/baba-is-harbor at {COMMIT} (levels from Baba Is You)",
        "built_by": "scripts/grid_games/build_babaisyou_packs.py",
        "rejected": rejected,
        "packs": [
            {"key": f"babaisyou_{world}", "stage": title, "levels": packs[world]}
            for world, title in WORLDS.items()
        ],
    }
    bbu._DATA.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    kept = sum(len(p) for p in packs.values())
    print(f"{kept} levels kept, {len(rejected)} rejected {sorted(rejected)} -> {bbu._DATA}")


if __name__ == "__main__":
    main()
