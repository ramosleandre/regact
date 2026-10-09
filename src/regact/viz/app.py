"""Local viewer for a regact experiment: `make viz PATH=experiments/<run>`.

A small FastAPI app + a vanilla-JS SPA. Reads the canonical artifacts (no DB):
one game or many from an experiment dir, the conversation (turns), the proxy
metrics, and the controller videos when present.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
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
    from regact.protocols.cwm import viewer as cwm_viewer
    from regact.protocols.cwm.worker import WorkerError

    playback = cwm_viewer.Playback()

    def metrics(view: reader.GameView, name: str) -> dict[str, Any]:
        result = game_metrics(view)
        protocol_name = view.config.get("protocol", {}).get("name")
        if protocol_name in ("cwm", "vanilla"):
            path = root / name / "cwm" / "status.json"
            if path.is_file():
                state = json.loads(path.read_text())
                latest = state.get("latest_exploration") or {}
                result.update(
                    protocol=protocol_name,
                    final_aggregate=latest.get("aggregate", {}),
                    score_source="latest real exploration",
                    best_exploration_aggregate=(state.get("best_exploration") or {}).get(
                        "aggregate", {}
                    ),
                    env_moves=state.get("n_total_transitions", state.get("n_step_events", 0))
                    + state.get("reset_actions", 0),
                    success_rate=latest.get("aggregate", {}).get("success_rate"),
                )
                problem = reader.problem_of(view.config)
                task = reader.task_name(name, view.state)
                with contextlib.suppress(Exception):  # a viewer renders without them
                    result["derived_metrics"] = problem.derived_trace_metrics(
                        task, cwm_viewer.info_trace(root / name)
                    )
                with contextlib.suppress(Exception):
                    result["progress"] = cwm_viewer.progress(
                        root / name, problem, task, state.get("progress")
                    )
        return result

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
                    "agent": {
                        key: (game.config.get("agent") or {}).get(key) for key in ("name", "model")
                    },
                    "task": game.state.get("task_name") or name.rsplit("/", 1)[-1],
                    "state": game.state,
                    "metrics": metrics(game, name),
                }
            )
        return {"experiment": root.name, "games": out}

    def _require_game(name: str) -> None:
        # Validate this run directly: frame playback must not scan the experiment tree.
        candidate = (root / name).resolve()
        if (
            Path(name).is_absolute()
            or not candidate.is_relative_to(root.resolve())
            or not (candidate / "logs/experiment_state.json").is_file()
        ):
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
            "metrics": metrics(view, name),
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

    def cwm_call(name: str, operation: Callable[[Path], Any]) -> Any:
        _require_game(name)
        try:
            return operation(root / name)
        except (ValueError, OSError, WorkerError) as exc:
            raise HTTPException(422, detail=str(exc)) from exc

    @app.get("/api/game/cwm")
    def cwm(name: str, before: int = 0) -> Any:
        return cwm_call(name, lambda task: cwm_viewer.inspect(task, before=before))

    @app.get("/api/game/cwm/evidence")
    def cwm_evidence(name: str, kind: str, identifier: int) -> Any:
        return cwm_call(name, lambda task: cwm_viewer.evidence(task, kind, identifier))

    @app.get("/api/game/cwm/evidence-image")
    def cwm_evidence_image(name: str, kind: str, identifier: int, side: str = "observed") -> Any:
        def read(task: Path) -> Any:
            evidence = cwm_viewer.evidence(task, kind, identifier)
            if kind == "observation":
                obs = evidence
            elif kind == "diagnostic" and side in (
                "predicted",
                "observed",
                "first_output",
                "second_output",
            ):
                obs = evidence.get(side)
            else:
                raise ValueError("unsupported evidence image source")
            if not isinstance(obs, dict) or "frame" not in obs:
                raise ValueError("this evidence has no observation image for that side")
            return Response(
                cwm_viewer.png(cwm_viewer.problem_for(task), obs), media_type="image/png"
            )

        return cwm_call(name, read)

    @app.get("/api/game/tool-image")
    def tool_image(name: str, filename: str) -> Any:
        _require_game(name)
        if not re.fullmatch(r"[0-9a-f]{64}\.(png|jpg|webp|gif)", filename):
            raise HTTPException(422, detail="invalid image identifier")
        directory = (root / name / "logs" / "media").resolve()
        path = directory / filename
        if (
            path.is_symlink()
            or not path.is_file()
            or not directory.is_relative_to((root / name).resolve())
        ):
            raise HTTPException(404, detail="recorded tool image unavailable")
        return FileResponse(path, headers={"Cache-Control": "private, max-age=31536000, immutable"})

    @app.get("/api/game/cwm/source")
    def cwm_source(name: str, bundle: str, filename: str = "") -> Any:
        return cwm_call(name, lambda task: cwm_viewer.source(task, bundle, filename))

    @app.post("/api/game/cwm/load")
    def cwm_load(name: str, kind: str, identifier: int) -> Any:
        return cwm_call(name, lambda task: playback.load(task, kind, identifier))

    @app.get("/api/game/cwm/frame")
    def cwm_frame(
        name: str, kind: str, identifier: int, index: int = 0, image: bool = False
    ) -> Any:
        def read(task: Path) -> Any:
            frame = playback.frame(task, kind, identifier, index)
            if image:
                return Response(
                    cwm_viewer.png(cwm_viewer.problem_for(task), frame["obs"]),
                    media_type="image/png",
                    headers={"Cache-Control": "no-store"},
                )
            return frame

        return cwm_call(name, read)

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
                return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
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
