"""The egress proxy: allow-list matching + CONNECT tunnel-vs-refuse over real sockets."""

import asyncio

from regact.security.egress_proxy import EgressProxy


def test_allows_exact_host_and_subdomains_only() -> None:
    p = EgressProxy(["api.openai.com", "chatgpt.com"])
    assert p.allows("api.openai.com")
    assert p.allows("api.openai.com:443")  # port stripped
    assert p.allows("x.chatgpt.com")  # subdomain
    assert not p.allows("arcprize.org")
    assert not p.allows("openai.com")  # parent is not a member
    assert not p.allows("evil-chatgpt.com")  # not a subdomain


def test_connect_refuses_disallowed_and_tunnels_allowed() -> None:
    async def check() -> None:
        # an "allowed" target: a local echo server
        async def echo(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
            w.write(await r.read(64))
            await w.drain()
            w.close()

        echo_srv = await asyncio.start_server(echo, "127.0.0.1", 0)
        echo_port = echo_srv.sockets[0].getsockname()[1]

        proxy = EgressProxy(["127.0.0.1"])  # allow the loopback echo host
        port = await proxy.start()

        # disallowed host -> 403, no tunnel
        r, w = await asyncio.open_connection("127.0.0.1", port)
        w.write(b"CONNECT arcprize.org:443 HTTP/1.1\r\n\r\n")
        await w.drain()
        assert b"403" in await r.readline()
        w.close()

        # allowed host -> 200 then a transparent tunnel
        r, w = await asyncio.open_connection("127.0.0.1", port)
        w.write(f"CONNECT 127.0.0.1:{echo_port} HTTP/1.1\r\n\r\n".encode())
        await w.drain()
        assert b"200" in await r.readline()
        await r.readline()  # the blank line ending the proxy's 200 response
        w.write(b"ping")
        await w.drain()
        assert await r.readexactly(4) == b"ping"
        w.close()

        await proxy.close()
        echo_srv.close()

    asyncio.run(check())
