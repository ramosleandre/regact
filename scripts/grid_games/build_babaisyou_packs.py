"""Build ``problems/grid_games/data/babaisyou_levels.json`` from baba-is-harbor's levels.

Usage:
    python scripts/grid_games/build_babaisyou_packs.py [--source <local clone>]

baba-is-harbor (github.com/stared/baba-is-harbor, Apache-2.0) extracted the campaign levels of
Baba Is You from the game's own files, each with a solution checked on the real game. This
script pins them by URL and SHA-256 (the level designs are the game's, so they are fetched at
install, never redistributed), replays every solution through regact's engine, and keeps a
level only if the solution wins there too. A level's reference is the move that wins (the
solutions sometimes keep moving after the win). One task per campaign stage, its levels in the
game's numbering, bonus levels last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from regact.problems.grid_games import babaisyou as bbu
from regact.problems.grid_games.babaisyou import engine

VERSION = "babaisyou-bench-v1"
COMMIT = "2adb73ad35d139ab490e2a8513631c43dec16095"
_RAW = f"https://raw.githubusercontent.com/stared/baba-is-harbor/{COMMIT}"
STAGES = {"00": "map", "01": "the lake", "02": "solitary island", "03": "temple ruins"}


def _read(source: Path | None, relative: str) -> bytes:
    if source is not None:
        return (source / relative).read_bytes()
    with urllib.request.urlopen(f"{_RAW}/{relative}", timeout=60) as response:
        return response.read()


def _solution(script: bytes) -> str:
    return script.decode().split('echo -n "')[1].split('"')[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=None, help="a local clone at COMMIT")
    args = parser.parse_args()
    names = json.loads(_read(args.source, "level_map.json"))["levels"]
    tasks = sorted(n for n in names if n.split("-")[1] in STAGES)
    packs: dict[str, list[dict[str, object]]] = {stage: [] for stage in STAGES}
    rejected = []
    for task in sorted(tasks, key=lambda t: ("bonus" in t, t)):
        level_path = f"tasks/{task}/environment/level.json"
        solution_path = f"tasks/{task}/solution/solve.sh"
        level_bytes, solution_bytes = (
            _read(args.source, level_path),
            _read(args.source, solution_path),
        )
        moves = _solution(solution_bytes)
        _, won = engine.play(json.loads(level_bytes), moves)
        if won is None:
            rejected.append(task)
            continue
        packs[task.split("-")[1]].append(
            {
                "id": task,
                "title": names[task]["title"],
                "url": f"{_RAW}/{level_path}",
                "sha256": hashlib.sha256(level_bytes).hexdigest(),
                "solution_url": f"{_RAW}/{solution_path}",
                "solution_sha256": hashlib.sha256(solution_bytes).hexdigest(),
                "reference": won,
            }
        )
    data = {
        "version": VERSION,
        "source": f"github.com/stared/baba-is-harbor at {COMMIT} (levels from Baba Is You)",
        "built_by": "scripts/grid_games/build_babaisyou_packs.py",
        "rejected": rejected,
        "packs": [
            {"key": f"babaisyou_{stage}", "stage": title, "levels": packs[stage]}
            for stage, title in STAGES.items()
        ],
    }
    bbu._DATA.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    kept = sum(len(p) for p in packs.values())
    print(f"{kept} levels kept, {len(rejected)} rejected {rejected} -> {bbu._DATA}")


if __name__ == "__main__":
    main()
