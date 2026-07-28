"""Child-side runner: drive an ``alancode`` agent from a parent process.

Spawned by :class:`regact.agent.alan_subprocess.AlanSubprocessAgent` as
``python -m regact.agent.alan_runner``, inside the OS sandbox. It speaks
newline-delimited JSON: commands in on stdin, normalized ``AgentEvent``s out on
stdout in the same shape as ``transcript.jsonl`` (:func:`event_to_json`), so the
parent rebuilds exactly the stream the in-process adapter would have yielded.

Two framing rules keep the protocol unambiguous: every event line carries its
event ``type``, and the runner's own control frames use a leading underscore
(``_ready`` / ``_turn_end`` / ``_fatal``), which no event type uses.

Framework tools are deliberately NOT registered natively here. A subprocess agent
reaches them over the workdir control CLI (``control_actions == "client_cli"``),
so this sandboxed child never holds a Python handle to the orchestrator's tools.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from regact.obs.transcript import event_to_json

# Control frames the runner emits alongside events; underscore-prefixed so they can
# never collide with an AgentEvent type name.
READY = "_ready"
TURN_END = "_turn_end"
FATAL = "_fatal"


def _write(frame: dict[str, Any]) -> None:
    """Emit one JSON frame on stdout (line-delimited, flushed so the parent sees it live)."""
    sys.stdout.write(json.dumps(frame) + "\n")
    sys.stdout.flush()


async def _read_command() -> dict[str, Any] | None:
    """Read one command frame from stdin; ``None`` on EOF or an unparsable line."""
    line = await asyncio.to_thread(sys.stdin.readline)
    if not line:
        return None  # parent closed the pipe
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _build(frame: dict[str, Any]) -> Any:
    """Construct the alancode agent from the ``start`` frame.

    Secrets arrive here on stdin rather than in argv, so an api_key is never visible
    in the host's process list.
    """
    from regact.agent.alan_adapter import build_alan_agent

    return build_alan_agent(
        cwd=str(frame.get("cwd") or "."),
        model=frame.get("model"),
        base_url=frame.get("base_url"),
        api_key=frame.get("api_key"),
        system_prompt=frame.get("system_prompt"),
        extra_tools=[],  # client_cli: the agent calls framework tools over the control CLI
        args=dict(frame.get("args") or {}),
    )


async def _run_turn(agent: Any, message: str) -> None:
    """Stream one turn's events, then close it with a ``_turn_end`` frame."""
    from regact.agent.alan_adapter import map_alan_events

    try:
        async for native in agent.query_events_async(message):
            for event in map_alan_events(native):
                _write(dict(event_to_json(event)))
    except Exception as exc:  # a backend fault must reach the parent, not kill the turn silently
        _write({"type": FATAL, "message": f"{type(exc).__name__}: {exc}"})
    finally:
        _write({"type": TURN_END})


async def _serve() -> int:
    """Command loop: build on ``start``, then run turns until ``close`` or EOF."""
    agent: Any = None
    while True:
        command = await _read_command()
        if command is None:
            break
        kind = command.get("cmd")
        if kind == "start":
            agent = _build(command)
            _write({"type": READY})
        elif kind == "send" and agent is not None:
            await _run_turn(agent, str(command.get("message", "")))
        elif kind == "inject" and agent is not None:
            agent.inject_message(str(command.get("message", "")))
        elif kind == "close":
            break
    if agent is not None:
        await agent.close()
    return 0


def main() -> int:
    try:
        return asyncio.run(_serve())
    except Exception as exc:  # startup faults (e.g. alancode missing) must be reportable
        _write({"type": FATAL, "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
