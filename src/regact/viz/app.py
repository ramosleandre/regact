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
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from regact.viz import reader
from regact.viz.metrics import game_metrics

_STATIC = Path(__file__).parent / "static"


def build_app(experiment_dir: str) -> FastAPI:
    app = FastAPI(title="regact viz")
    root = Path(experiment_dir)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/games")
    def games() -> dict[str, Any]:
        out = []
        for name in reader.list_games(experiment_dir):
            game = reader.load_game(experiment_dir, name)
            out.append({"name": name, "state": game.state, "metrics": game_metrics(game)})
        return {"experiment": root.name, "games": out}

    @app.get("/api/game/{name}")
    def game(name: str) -> dict[str, Any]:
        if name not in reader.list_games(experiment_dir):
            raise HTTPException(status_code=404, detail=f"unknown game {name!r}")
        view = reader.load_game(experiment_dir, name)
        return {
            "name": name,
            "state": view.state,
            "config": view.config,
            "turns": [dataclasses.asdict(t) for t in view.turns],
            "submissions": [dataclasses.asdict(s) for s in view.submissions],
            "metrics": game_metrics(view),
        }

    @app.get("/api/game/{name}/artifacts")
    def artifacts(name: str) -> dict[str, Any]:
        if name not in reader.list_games(experiment_dir):
            raise HTTPException(status_code=404, detail=f"unknown game {name!r}")
        view = reader.load_game(experiment_dir, name)
        return {
            "files": [dataclasses.asdict(a) for a in reader.list_artifacts(experiment_dir, name)],
            "submissions": [dataclasses.asdict(s) for s in view.submissions],
        }

    @app.get("/api/game/{name}/logs")
    def logs(name: str) -> dict[str, Any]:
        if name not in reader.list_games(experiment_dir):
            raise HTTPException(status_code=404, detail=f"unknown game {name!r}")
        return reader.load_logs(experiment_dir, name)

    @app.get("/video/{game}/{submission}/{filename}")
    def video(game: str, submission: str, filename: str) -> FileResponse:
        if not filename.endswith(".mp4"):
            raise HTTPException(status_code=400, detail="only .mp4")
        path = root / game / "workdir" / "submissions" / submission / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="video not found")
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "no-store"})

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
