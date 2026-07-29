"""The loopback mirror: socket forwarding on both sides + the in-sandbox launcher.

Subprocesses here are spawned from argv lists (never a shell string), so there is
no injection surface; the launcher under test enforces the same rule.
"""

import asyncio
import os
import socket
import sys

import regact
from regact.security.netbridge import LoopbackMirror

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(regact.__file__)))


async def _echo_server() -> tuple[asyncio.AbstractServer, int]:
    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(await reader.read(64))
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


def test_mirror_forwards_socket_file_to_host_tcp() -> None:
    async def check() -> None:
        server, port = await _echo_server()
        mirror = LoopbackMirror()
        path = await mirror.mirror(port)
        assert path == await mirror.mirror(port)  # idempotent per port

        reader, writer = await asyncio.open_unix_connection(path)
        writer.write(b"ping")
        await writer.drain()
        assert await reader.readexactly(4) == b"ping"
        writer.close()

        await mirror.close()
        server.close()

    asyncio.run(check())


def test_socket_dir_is_removed_on_close() -> None:
    mirror = LoopbackMirror()
    assert os.path.isdir(mirror.socket_dir)
    asyncio.run(mirror.close())
    assert not os.path.exists(mirror.socket_dir)


def test_argv_prefix_names_each_port_and_ends_with_separator() -> None:
    mirror = LoopbackMirror()
    prefix = mirror.argv_prefix([8100, 8200])
    assert prefix[0] == sys.executable
    assert prefix[-1] == "--"
    assert f"8100={mirror.socket_path(8100)}" in prefix
    assert f"8200={mirror.socket_path(8200)}" in prefix
    asyncio.run(mirror.close())


def test_launcher_relays_a_port_to_the_socket_file() -> None:
    """End to end without a sandbox: TCP echo <- socket file <- launcher port <- client."""

    async def check() -> None:
        server, echo_port = await _echo_server()
        mirror = LoopbackMirror()
        path = await mirror.mirror(echo_port)

        # Outside a namespace the echo port is taken, so relay on another free one.
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        relay_port = probe.getsockname()[1]
        probe.close()

        client = (
            "import socket\n"
            f'sock = socket.create_connection(("127.0.0.1", {relay_port}), 5)\n'
            'sock.sendall(b"ping")\n'
            'assert sock.recv(4) == b"ping"\n'
        )
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "regact.security.netbridge",
            "--mirror",
            f"{relay_port}={path}",
            "--",
            sys.executable,
            "-c",
            client,
            env={**os.environ, "PYTHONPATH": _SRC},
        )
        assert await proc.wait() == 0

        await mirror.close()
        server.close()

    asyncio.run(check())


def test_launcher_propagates_the_child_exit_code() -> None:
    async def check() -> None:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "regact.security.netbridge",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
            env={**os.environ, "PYTHONPATH": _SRC},
        )
        assert await proc.wait() == 7

    asyncio.run(check())
