"""A localhost allow-listing egress proxy.

The agent runs with ``HTTPS_PROXY``/``HTTP_PROXY`` pointed here and, under the OS
sandbox (``deny_egress``), no direct external route — so its only way out is this
proxy, which tunnels ``CONNECT`` requests to an allow-list of hosts and refuses
everything else. The proxy runs in the orchestrator (outside the sandbox), so it
reaches the allowed hosts on the agent's behalf.

Agnostic: it takes only host names. The allow-list comes from the loaded agent's
``host_egress_hosts()`` (e.g. codex: api.openai.com / chatgpt.com); the proxy never
knows which agent or game it serves.
"""

from __future__ import annotations

import asyncio
import contextlib

_BUF = 65536


class EgressProxy:
    """A CONNECT proxy that only tunnels to ``allow_hosts`` (and their subdomains)."""

    def __init__(self, allow_hosts: list[str], *, host: str = "127.0.0.1") -> None:
        self._allow = frozenset(h.lower() for h in allow_hosts)
        self._host = host
        self._server: asyncio.AbstractServer | None = None
        self.port = 0

    async def start(self) -> int:
        """Bind on an ephemeral loopback port and return it."""
        self._server = await asyncio.start_server(self._handle, self._host, 0)
        self.port = int(self._server.sockets[0].getsockname()[1])
        return self.port

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()

    def allows(self, host: str) -> bool:
        """True if ``host`` is an allowed host or a subdomain of one."""
        host = host.lower().rsplit(":", 1)[0]
        return host in self._allow or any(host.endswith("." + h) for h in self._allow)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            parts = (await reader.readline()).decode("latin-1").split()
            while (await reader.readline()) not in (b"\r\n", b"\n", b""):
                pass  # drain the rest of the request headers
            if len(parts) >= 2 and parts[0].upper() == "CONNECT":
                host, _, port = parts[1].partition(":")
                if self.allows(host):
                    await self._tunnel(reader, writer, host, int(port or "443"))
                    return
                await _reply(writer, b"HTTP/1.1 403 Forbidden\r\n\r\n")
            else:
                # plain HTTP (no TLS) is not used by the LLM APIs; refuse to keep it simple.
                await _reply(writer, b"HTTP/1.1 501 Not Implemented\r\n\r\n")
        except Exception:
            with contextlib.suppress(Exception):
                writer.close()

    async def _tunnel(
        self,
        creader: asyncio.StreamReader,
        cwriter: asyncio.StreamWriter,
        host: str,
        port: int,
    ) -> None:
        try:
            sreader, swriter = await asyncio.open_connection(host, port)
        except OSError:
            await _reply(cwriter, b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return
        cwriter.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await cwriter.drain()
        await asyncio.gather(_pipe(creader, swriter), _pipe(sreader, cwriter))


async def _reply(writer: asyncio.StreamWriter, data: bytes) -> None:
    with contextlib.suppress(Exception):
        writer.write(data)
        await writer.drain()
        writer.close()


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    with contextlib.suppress(OSError):
        while data := await reader.read(_BUF):
            writer.write(data)
            await writer.drain()
    with contextlib.suppress(OSError):
        writer.close()
