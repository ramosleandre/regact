"""The HTTP env server: the env behind localhost HTTP.

Holds the game sessions and exposes them over HTTP; the agent never imports the
game, it only sees these endpoints. The real ``game_id`` is used as-is (no
aliasing). Inspired by arc-3-agents-baseline1 ``server.py``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from regact.env.session import EnvSession
from regact.env.wrapped_env import WrappedEnv


class EnvServer:
    """A localhost HTTP server fronting one or more :class:`EnvSession` objects.

    Routes (JSON envelopes ``{obs, action_count}``):
      ``POST /env/{game_id}/reset`` ``{seed?}`` · ``POST /env/{game_id}/step`` ``{action}``
      ``GET /env/{game_id}/current`` · ``GET /env/{game_id}/last-step``
      ``POST /env/{game_id}/stop``
    """

    def __init__(self, *, host: str = "127.0.0.1", port: int = 0) -> None:
        self._host = host
        self._port = port
        self._sessions: dict[str, EnvSession] = {}
        self.app = self._build_app()

    def register(self, game_id: str, session: EnvSession) -> str:
        """Attach a session under its real ``game_id``; return that id."""
        self._sessions[game_id] = session
        return game_id

    def _session(self, game_id: str) -> EnvSession:
        try:
            return self._sessions[game_id]
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown game {game_id!r}") from None

    def _require_live(self, game_id: str) -> WrappedEnv:
        session = self._session(game_id)
        if session.live is None:
            raise HTTPException(
                status_code=409, detail=f"game {game_id!r} not started; reset first"
            )
        return session.live

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="regact env server")

        @app.post("/env/{game_id}/reset")
        def reset(game_id: str, body: dict[str, Any]) -> dict[str, Any]:
            session = self._session(game_id)
            env = session.make()
            obs = env.reset(seed=body.get("seed"))
            return {"obs": obs.to_json(), "action_count": env.action_count}

        @app.post("/env/{game_id}/step")
        def step(game_id: str, body: dict[str, Any]) -> dict[str, Any]:
            env = self._require_live(game_id)
            obs = env.step(body.get("action"))
            return {"obs": obs.to_json(), "action_count": env.action_count}

        @app.get("/env/{game_id}/current")
        def current(game_id: str) -> dict[str, Any]:
            env = self._require_live(game_id)
            if env.last_obs is None:
                raise HTTPException(status_code=409, detail="no observation yet")
            return {"obs": env.last_obs.to_json(), "action_count": env.action_count}

        @app.get("/env/{game_id}/last-step")
        def last_step(game_id: str) -> dict[str, int]:
            return {"action_count": self._require_live(game_id).action_count}

        @app.post("/env/{game_id}/stop")
        def stop(game_id: str) -> dict[str, bool]:
            self._session(game_id).close()
            return {"ok": True}

        return app

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    async def serve(self) -> None:
        """Run a real uvicorn server (for non-test use)."""
        import uvicorn

        config = uvicorn.Config(self.app, host=self._host, port=self._port, log_level="warning")
        await uvicorn.Server(config).serve()
