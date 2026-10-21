"""Install the PuzzleScript engine and games (``make install-puzzlescript``).

Usage:
    python scripts/grid_games/install_puzzlescript.py

Downloads the engine's ``src/js`` at the pinned commit and every game listed in
``problems/grid_games/data/puzzlescript_games.json`` from its pinned URL into
``REGACT_PUZZLESCRIPT_DIR`` (default ``~/.cache/regact/puzzlescript``). A game whose SHA-256 differs
from the recorded one is refused. The games are fetched, not redistributed: their authors'
gists carry no licence. Run it where there is internet (on a cluster: a login node).
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from regact.problems.grid_games import puzzlescript as ps

_ENGINE_TARBALL = f"https://codeload.github.com/increpare/PuzzleScript/tar.gz/{ps.ENGINE_COMMIT}"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def install_engine() -> None:
    target = ps.engine_src().parent
    if (target / "src" / "js" / "engine.js").is_file():
        return
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(fileobj=io.BytesIO(fetch(_ENGINE_TARBALL)), mode="r:gz") as archive:
        for member in archive.getmembers():
            relative = Path(*Path(member.name).parts[1:])
            if member.isfile() and (
                relative.parts[:2] == ("src", "js") or str(relative) == "LICENSE"
            ):
                (tmp / relative).parent.mkdir(parents=True, exist_ok=True)
                (tmp / relative).write_bytes(archive.extractfile(member).read())  # type: ignore[union-attr]
    tmp.rename(target)
    print(f"engine {ps.ENGINE_COMMIT[:7]} -> {target}")


def install_games() -> None:
    games = json.loads(ps._GAMES.read_text(encoding="utf-8"))["games"]
    for game in games:
        path = ps.game_source(game["sha256"])
        if path.is_file():
            continue
        body = fetch(game["url"])
        if hashlib.sha256(body).hexdigest() != game["sha256"]:
            raise SystemExit(f"{game['key']}: the source at {game['url']} changed; refusing it")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    print(f"{len(games)} games -> {ps.data_dir() / 'games'}")


if __name__ == "__main__":
    install_engine()
    install_games()
