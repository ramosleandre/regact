"""Carry host-loopback TCP services across a network-namespaced sandbox.

A sandbox that unshares the network namespace gets a private loopback: every
route out is gone by construction, including the host's ``127.0.0.1`` — so the
host-side services a confined process legitimately needs (the env server, the
egress proxy, a local model endpoint) vanish along with the internet. Pathname
unix sockets are mount-namespace objects, invisible to network isolation, so a
socket file bound into the sandbox still reaches a listener outside. This module
turns that property into a transparent bridge:

  host side      :class:`LoopbackMirror` — one socket file per mirrored port,
                 each forwarding to ``127.0.0.1:<port>`` on the host.
  sandbox side   ``python -m regact.security.netbridge --mirror P=SOCK -- argv``
                 binds ``127.0.0.1:P`` *inside* the namespace, relays every
                 connection to the socket file, then runs ``argv`` as its child.

A mirrored port keeps its host number on both sides, so a URL minted outside the
sandbox stays valid inside it — nothing is rewritten. Only the mirrored ports
exist in the namespace; every other destination stays unreachable. Which ports a
given child gets is the caller's choice per wrap, so different subprocesses of
one run can see different services.

Agnostic: only ports, paths, and an argv — no agent, environment, or feature
types. The child command is always launched from an argv list (no shell is ever
involved, so there is no injection surface), and this process never writes to
stdout (stdio belongs to the child, which may speak a framed protocol over it);
diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import shutil
import signal
import sys
import tempfile
from collections.abc import Sequence

_BUF = 65536

Stream = tuple[asyncio.StreamReader, asyncio.StreamWriter]


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Copy one direction until EOF, then half-close the write side."""
    try:
        while data := await reader.read(_BUF):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except OSError:
        pass


async def _splice(a: Stream, b: Stream) -> None:
    """Relay both directions until each hits EOF, then close both endpoints."""
    await asyncio.gather(_pipe(a[0], b[1]), _pipe(b[0], a[1]))
    for _, writer in (a, b):
        with contextlib.suppress(OSError):
            writer.close()


class LoopbackMirror:
    """Host side of the bridge: one socket file per mirrored loopback port.

    :meth:`mirror` creates ``<socket_dir>/p<port>.sock`` forwarding every
    connection to host ``127.0.0.1:<port>``. A sandbox launcher makes the socket
    file visible inside (an ``allow_rw`` path) and prepends :meth:`argv_prefix`
    to the child argv so the same port comes back up in the namespace.
    """

    def __init__(self) -> None:
        # Its own short-lived dir: AF_UNIX paths have a ~100-char limit, so the
        # sockets cannot live under a deep run directory.
        self._dir = tempfile.mkdtemp(prefix="regact-net-")
        self._servers: dict[int, asyncio.AbstractServer] = {}

    @property
    def socket_dir(self) -> str:
        return self._dir

    def socket_path(self, port: int) -> str:
        return os.path.join(self._dir, f"p{port}.sock")

    async def mirror(self, port: int) -> str:
        """Expose host ``127.0.0.1:port`` as a socket file; return its path (idempotent)."""
        path = self.socket_path(port)
        if port in self._servers:
            return path

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                remote = await asyncio.open_connection("127.0.0.1", port)
            except OSError:
                writer.close()
                return
            await _splice((reader, writer), remote)

        self._servers[port] = await asyncio.start_unix_server(handle, path=path)
        return path

    def argv_prefix(self, ports: Sequence[int]) -> list[str]:
        """The in-sandbox launcher argv for ``ports``, to prepend to a child argv."""
        prefix = [sys.executable, "-m", "regact.security.netbridge"]
        for port in ports:
            prefix += ["--mirror", f"{port}={self.socket_path(port)}"]
        return [*prefix, "--"]

    async def close(self) -> None:
        for server in self._servers.values():
            server.close()
            with contextlib.suppress(Exception):
                await server.wait_closed()
        self._servers.clear()
        shutil.rmtree(self._dir, ignore_errors=True)


# --- in-sandbox entry ------------------------------------------------------- #


def _parse(argv: list[str]) -> tuple[list[tuple[int, str]], list[str]]:
    """Split at the first ``--`` (everything after is the child argv), parse the rest."""
    if "--" in argv:
        split = argv.index("--")
        own, command = argv[:split], argv[split + 1 :]
    else:
        own, command = argv, []
    parser = argparse.ArgumentParser(
        prog="regact.netbridge",
        description="Bind loopback ports that relay to unix sockets, then run a command.",
    )
    parser.add_argument(
        "--mirror",
        action="append",
        default=[],
        metavar="PORT=SOCKET",
        help="repeatable: serve 127.0.0.1:PORT by relaying to the SOCKET file",
    )
    args = parser.parse_args(own)
    mirrors = []
    for spec in args.mirror:
        port, _, path = spec.partition("=")
        mirrors.append((int(port), path))
    return mirrors, command


async def _serve_mirror(port: int, path: str) -> asyncio.AbstractServer:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            remote = await asyncio.open_unix_connection(path)
        except OSError:
            writer.close()
            return
        await _splice((reader, writer), remote)

    return await asyncio.start_server(handle, "127.0.0.1", port)


async def _amain(mirrors: list[tuple[int, str]], command: list[str]) -> int:
    if not command:
        print("netbridge: no command after --", file=sys.stderr)
        return 2
    servers = []
    for port, path in mirrors:
        try:
            servers.append(await _serve_mirror(port, path))
        except OSError as exc:
            print(f"netbridge: cannot bind 127.0.0.1:{port}: {exc}", file=sys.stderr)
            return 1

    proc = await asyncio.create_subprocess_exec(*command)

    def _forward_terminate() -> None:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError, ValueError):
            loop.add_signal_handler(sig, _forward_terminate)
    code = await proc.wait()
    for server in servers:
        server.close()
    return code if code >= 0 else 128 - code  # signal death -> the usual 128+N


def main(argv: list[str] | None = None) -> int:
    mirrors, command = _parse(sys.argv[1:] if argv is None else argv)
    return asyncio.run(_amain(mirrors, command))


if __name__ == "__main__":
    raise SystemExit(main())
