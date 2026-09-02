"""Local viewer for a regact experiment: `make viz PATH=experiments/<run>`.

A small FastAPI app + a vanilla-JS SPA. Reads the canonical artifacts (no DB):
one game or many from an experiment dir, the conversation (turns), the proxy
metrics, and the controller videos when present.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from regact.viz import reader
from regact.viz.metrics import game_metrics

_STATIC = Path(__file__).parent / "static"


class _NoCacheStatic(StaticFiles):
    """Serve the viz assets with ``Cache-Control: no-cache`` so the browser revalidates (ETag/304)
    every request. The viz is edited live - app.js, the icon registry, and hand-added icon PNGs all
    change under a running server; without this a heuristically-cached copy silently goes stale."""

    async def get_response(self, path: str, scope: Any) -> Any:
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


def _settings_dir() -> Path:
    # Per-interface viz settings (colors, order, toggles) live here, one JSON per graph scope, so
    # they survive across sessions/browsers and viz updates. Overridable for tests via env var.
    return Path(
        os.environ.get("REGACT_VIZ_SETTINGS_DIR") or (Path.home() / ".regact" / "viz_settings")
    )


def _settings_path(scope: str) -> Path:
    """Flat, traversal-safe filename for one interface's settings (the graph `under` scope)."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", scope) or "_root"
    return _settings_dir() / (safe + ".json")


def _experiment_of(game_relpath: str) -> str:
    """Fallback experiment id when config.experiment_name is absent: for a
    ``<experiment>/<timestamp>/<task>`` run path it is the grandparent; else the top component."""
    parts = game_relpath.split("/")
    if len(parts) >= 3:
        return parts[-3]
    return parts[0] if parts else game_relpath


def build_app(experiment_dir: str) -> FastAPI:
    app = FastAPI(title="regact viz")
    root = Path(experiment_dir)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/tree")
    def tree() -> dict[str, Any]:
        # Cheap folder scan (no metric parsing) - the browsable index for a many-experiment root.
        return {"root": root.name, "tree": reader.build_tree(experiment_dir)}

    @app.get("/api/games")
    def games(under: str = "") -> dict[str, Any]:
        # ``under`` scopes the (parsed) list to one subtree (a run/experiment), so the browser never
        # parses the whole root at once. Empty = every game (kept for a flat single-run root).
        names = reader.list_games(experiment_dir)
        if under:
            names = [n for n in names if n == under or n.startswith(under + "/")]
        out = []
        for name in names:
            game = reader.load_game(experiment_dir, name)
            out.append(
                {
                    "name": name,
                    # Grouping keys for the cross-run graphs: an experiment is one agent+problem
                    # config (config.experiment_name), and one experiment holds many runs of many
                    # tasks (some tasks repeated across timestamps -> aggregated in the UI).
                    "experiment": game.config.get("experiment_name") or _experiment_of(name),
                    "task": game.state.get("task_name") or name.rsplit("/", 1)[-1],
                    "state": game.state,
                    "metrics": game_metrics(game),
                }
            )
        return {"experiment": root.name, "games": out}

    def _require_game(name: str) -> None:
        # ``name`` is a query param (a run's path relative to the root, possibly nested for a
        # sweep). Validating against list_games both 404s the unknown and blocks path traversal.
        if name not in reader.list_games(experiment_dir):
            raise HTTPException(status_code=404, detail=f"unknown game {name!r}")

    @app.get("/api/game")
    def game(name: str) -> dict[str, Any]:
        _require_game(name)
        view = reader.load_game(experiment_dir, name)
        return {
            "name": name,
            "state": view.state,
            "config": view.config,
            "turns": [dataclasses.asdict(t) for t in view.turns],
            "submissions": [dataclasses.asdict(s) for s in view.submissions],
            "metrics": game_metrics(view),
        }

    @app.get("/api/game/artifacts")
    def artifacts(name: str) -> dict[str, Any]:
        _require_game(name)
        view = reader.load_game(experiment_dir, name)
        return {
            "files": [dataclasses.asdict(a) for a in reader.list_artifacts(experiment_dir, name)],
            "submissions": [dataclasses.asdict(s) for s in view.submissions],
        }

    @app.get("/api/game/logs")
    def logs(name: str) -> dict[str, Any]:
        _require_game(name)
        return reader.load_logs(experiment_dir, name)

    @app.get("/video")
    def video(game: str, submission: str, filename: str) -> FileResponse:
        # game/submission/filename are all query params (game may be a nested sweep path).
        if not filename.endswith(".mp4"):
            raise HTTPException(status_code=400, detail="only .mp4")
        _require_game(game)
        path = (root / game / "workdir" / "submissions" / submission / filename).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise HTTPException(status_code=404, detail="video not found")
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "no-store"})

    @app.get("/api/settings")
    def get_settings(scope: str = "") -> dict[str, Any]:
        # This interface's saved settings (empty dict if none yet). Scope = the graph `under` path.
        path = _settings_path(scope)
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
        return {}

    @app.put("/api/settings")
    async def put_settings(scope: str, request: Request) -> dict[str, str]:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="settings must be a JSON object")
        _settings_dir().mkdir(parents=True, exist_ok=True)
        _settings_path(scope).write_text(json.dumps(body, indent=2), encoding="utf-8")
        return {"status": "saved"}

    app.mount("/static", _NoCacheStatic(directory=str(_STATIC)), name="static")
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="regact.viz")
    parser.add_argument("--experiment", required=True, help="Path to an experiment dir.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8030)
    args = parser.parse_args(argv)

    import uvicorn

    print(f"regact viz → http://{args.host}:{args.port}  (experiment: {args.experiment})")
    uvicorn.run(build_app(args.experiment), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
