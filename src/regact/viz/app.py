"""Local viewer for a regact experiment: `make viz PATH=experiments/<run>`.

A small FastAPI app + a vanilla-JS SPA. Reads the canonical artifacts (no DB):
one game or many from an experiment dir, the conversation (turns), the proxy
metrics, and the controller videos when present.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from regact.viz import reader
from regact.viz.metrics import game_metrics

_STATIC = Path(__file__).parent / "static"
# Rendered task previews, keyed by task name (None = tried and can't render). Deterministic per
# task, so this process-wide cache means each task renders at most once.
_PREVIEW_CACHE: dict[str, bytes | None] = {}


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

    @app.get("/api/task_preview")
    def task_preview(task: str) -> Response:
        # Small rendered thumbnail of a task's env (the Graphs x-axis previews). Rendered on first
        # request and cached; 404 for a task we can't render (the UI just hides the image then).
        if task not in _PREVIEW_CACHE:
            try:
                from regact.viz.task_preview import render_task_png

                _PREVIEW_CACHE[task] = render_task_png(task)
            except Exception:
                _PREVIEW_CACHE[task] = None
        png = _PREVIEW_CACHE[task]
        if png is None:
            raise HTTPException(status_code=404, detail="no preview for this task")
        return Response(
            content=png, media_type="image/png", headers={"Cache-Control": "max-age=86400"}
        )

    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
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
