"""How the orchestrator exposes the env server for one task.

Two transports behind one context manager:
  - **in-process** (FastAPI ``TestClient``): for the ``scripted`` backend, which
    never runs real subprocess scripts — no socket, fast, flake-free in CI.
  - **real uvicorn** (ephemeral port): for any real agent, whose exploration
    scripts are subprocesses that hit ``http://127.0.0.1:<port>`` via the baked
    ``make_env``. This is the faithful "HTTP only" path.

``serve_env`` yields an :class:`EnvConnection` (the base url to bake into the
workdir + a connected :class:`EnvClient` the orchestrator's executor drives) and
tears the server down on exit.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from regact.env.server import EnvServer
from regact.envclient.client import EnvClient

# Placeholder baked into a scripted run's workdir; that backend never calls it.
_IN_PROCESS_URL = "http://127.0.0.1:0"


@dataclass
class EnvConnection:
    """What a task needs to talk to its env: the URL to bake + a live client."""

    base_url: str  # baked into framework/make_env.py (used by subprocess agents)
    client: EnvClient  # the orchestrator's own client (drives eval)


@asynccontextmanager
async def serve_env(
    server: EnvServer, game_id: str, *, in_process: bool
) -> AsyncIterator[EnvConnection]:
    """Front ``server`` for one game over the chosen transport; yield the connection."""
    if in_process:
        from fastapi.testclient import TestClient

        client = EnvClient(TestClient(server.app), game_id)
        try:
            yield EnvConnection(base_url=_IN_PROCESS_URL, client=client)
        finally:
            client.close()
        return

    async with _running_uvicorn(server) as base_url:
        client = EnvClient.connect(base_url, game_id)
        try:
            yield EnvConnection(base_url=base_url, client=client)
        finally:
            client.close()


@asynccontextmanager
async def _running_uvicorn(server: EnvServer) -> AsyncIterator[str]:
    """Run ``server.app`` on an ephemeral port, on its own thread.

    A separate thread (with its own event loop) is required: the orchestrator
    drives the env through a *synchronous* ``EnvClient``, so a server sharing the
    caller's event loop would deadlock (the sync request blocks the loop the
    server needs to answer on).
    """
    import uvicorn

    config = uvicorn.Config(server.app, host="127.0.0.1", port=0, log_level="warning", ws="none")
    uv_server = uvicorn.Server(config)
    thread = threading.Thread(target=uv_server.run, daemon=True)
    thread.start()
    try:
        while not uv_server.started:
            await asyncio.sleep(0.02)
        port = uv_server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        uv_server.should_exit = True
        await asyncio.to_thread(thread.join, 5)
