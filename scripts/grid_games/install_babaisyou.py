"""Fetch the Baba Is You levels (``make install-babaisyou``).

Usage:
    python scripts/grid_games/install_babaisyou.py

Downloads every level listed in ``problems/grid_games/data/babaisyou_levels.json``, and the
solutions checked on the real game (worlds 0-3), from their pinned URLs into
``REGACT_BABAISYOU_DIR`` (default ``~/.cache/regact/babaisyou``). A file whose SHA-256 differs
from the recorded one is refused. The levels are the game's designs: they are fetched, not
redistributed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from regact.problems.grid_games import babaisyou as bbu


def fetch(url: str, sha256: str, path: Path) -> None:
    if path.is_file():
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        body = response.read()
    if hashlib.sha256(body).hexdigest() != sha256:
        raise SystemExit(f"{url} changed since the levels were pinned; refusing it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


if __name__ == "__main__":
    data = json.loads(bbu._DATA.read_text(encoding="utf-8"))
    levels = [level for pack in data["packs"] for level in pack["levels"]]
    for level in levels:
        fetch(level["url"], level["sha256"], bbu.level_path(level["sha256"]))
        if "solution_url" in level:
            fetch(
                level["solution_url"],
                level["solution_sha256"],
                bbu.solution_path(level["solution_sha256"]),
            )
    print(f"{len(levels)} levels -> {bbu.data_dir()}")
