"""Unit tests: the out-of-process Alan backend and the wire format it rides on.

None of these need ``alancode`` — they cover the parts regact owns: the event
round-trip that the child/parent protocol depends on, the registry wiring, the
declared capabilities, and how a child frame becomes an event.
"""

from __future__ import annotations

import json
import sys

import pytest

import regact.agent.alan_subprocess as alan_subprocess
from regact.agent.alan_runner import FATAL, READY, TURN_END, _run_turn
from regact.agent.alan_subprocess import AlanSubprocessAgent
from regact.agent.base import build_agent
from regact.agent.events import (
    AgentError,
    AgentEvent,
    SystemPrompt,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnComplete,
    UserMessage,
)
from regact.config.schema import AgentConfig, AgentName
from regact.obs.errors import ErrorCategory
from regact.obs.transcript import event_from_json, event_to_json

_EVENTS: list[AgentEvent] = [
    TextDelta(text="hello"),
    ThinkingDelta(text="hmm"),
    ToolCall(id="1", name="Bash", input={"command": "ls"}),
    ToolResult(id="1", output="files", is_error=False),
    TurnComplete(final_text="done", usage={"output_tokens": 3}),
    AgentError(category=ErrorCategory.AGENT_API, message="boom"),
    SystemPrompt(text="brief"),
    UserMessage(text="go"),
]


@pytest.mark.parametrize("event", _EVENTS, ids=lambda e: type(e).__name__)
def test_event_json_round_trips(event: AgentEvent) -> None:
    """The child serializes with event_to_json and the parent rebuilds with
    event_from_json, so every member of the union must survive the trip intact."""
    assert event_from_json(dict(event_to_json(event))) == event


def test_event_from_json_ignores_what_it_cannot_decode() -> None:
    """A reader on a foreign or newer stream skips frames instead of raising."""
    assert event_from_json({"type": TURN_END}) is None  # a control frame, not an event
    assert event_from_json({"type": "Nope", "text": "x"}) is None  # unknown tag
    assert event_from_json({"type": "TextDelta", "wrong": 1}) is None  # bad payload
    assert event_from_json({"type": "AgentError", "category": "??", "message": "m"}) is None


def test_registry_builds_the_subprocess_backend() -> None:
    agent = build_agent(AgentConfig(name=AgentName.ALAN))
    assert isinstance(agent, AlanSubprocessAgent)


def test_capabilities_route_tools_through_the_control_channel() -> None:
    """Out-of-process, framework tools cannot be native Python objects: they must go
    over the workdir control CLI, and the loop must not also run them."""
    caps = AlanSubprocessAgent().capabilities()
    assert caps.control_actions == "client_cli"
    assert caps.executes_tools is False
    assert caps.system_prompt == "replace"


def test_fatal_frame_becomes_an_agent_error() -> None:
    """A backend fault in the child surfaces as a normalized error, not a lost turn."""
    event = AlanSubprocessAgent._to_event({"type": FATAL, "message": "no model"})
    assert isinstance(event, AgentError)
    assert event.category is ErrorCategory.AGENT_API and "no model" in event.message
    # A ready frame is control, not an event.
    assert AlanSubprocessAgent._to_event({"type": READY}) is None


async def test_send_before_start_reports_instead_of_crashing() -> None:
    agent = AlanSubprocessAgent()
    events = [event async for event in agent.send("hi")]
    assert len(events) == 1
    assert isinstance(events[0], AgentError) and "not started" in events[0].message


# Speaks the runner protocol (ready after the start frame), then dies mid-turn with a
# traceback on stderr - the shape of a real backend crash.
_DYING_CHILD = (
    "import json, sys\n"
    "sys.stdin.readline()\n"
    'print(json.dumps({"type": "_ready"}), flush=True)\n'
    "sys.stdin.readline()\n"
    'print("Traceback (most recent call last):", file=sys.stderr)\n'
    'print("ValueError: kaboom", file=sys.stderr, flush=True)\n'
    "sys.exit(7)\n"
)

# Closes its stdout mid-turn but stays alive (the other way a turn's stream can end).
# os.close(1), not sys.stdout.close(): CPython never closes the real std fds.
_STDOUT_CLOSER = (
    "import json, os, sys, time\n"
    "sys.stdin.readline()\n"
    'print(json.dumps({"type": "_ready"}), flush=True)\n'
    "sys.stdin.readline()\n"
    "os.close(1)\n"
    "time.sleep(30)\n"
)


async def _start_scripted_child(agent: AlanSubprocessAgent, script: str, cwd: str) -> None:
    """Boot the agent against a stand-in child (runtime_wrap swaps the runner argv)."""
    await agent.start(
        cwd=cwd,
        model=None,
        base_url=None,
        api_key=None,
        system_prompt=None,
        runtime_wrap=lambda argv: [sys.executable, "-c", script],
    )


async def test_child_death_reports_exit_code_and_stderr(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A dead child must be reported with its real exit code and its stderr tail -
    not the pre-reap ``code None`` that hides the cause."""
    agent = AlanSubprocessAgent()
    await _start_scripted_child(agent, _DYING_CHILD, str(tmp_path))
    try:
        events = [event async for event in agent.send("go")]
    finally:
        await agent.close()
    assert len(events) == 1 and isinstance(events[0], AgentError)
    assert "exited with code 7" in events[0].message
    assert "kaboom" in events[0].message


async def test_live_child_with_closed_stdout_is_distinguished(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Stdout EOF with the process still running is its own diagnosis, not a fake exit code."""
    monkeypatch.setattr(alan_subprocess, "_REAP_TIMEOUT_S", 0.3)
    agent = AlanSubprocessAgent()
    await _start_scripted_child(agent, _STDOUT_CLOSER, str(tmp_path))
    try:
        events = [event async for event in agent.send("go")]
    finally:
        await agent.close()
    assert len(events) == 1 and isinstance(events[0], AgentError)
    assert "still alive" in events[0].message


# Turn 1 errors (fatal + turn_end), turn 2 succeeds; exits when stdin closes.
_ERROR_THEN_OK_CHILD = (
    "import json, sys\n"
    "sys.stdin.readline()\n"
    'print(json.dumps({"type": "_ready"}), flush=True)\n'
    "sys.stdin.readline()\n"
    'print(json.dumps({"type": "_fatal", "message": "boom"}), flush=True)\n'
    'print(json.dumps({"type": "_turn_end"}), flush=True)\n'
    "sys.stdin.readline()\n"
    'print(json.dumps({"type": "TextDelta", "text": "ok"}), flush=True)\n'
    'print(json.dumps({"type": "_turn_end"}), flush=True)\n'
    "sys.stdin.readline()\n"
)


async def test_stale_turn_is_drained_before_the_next(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """When a turn's consumer stops early (the loop breaks on an error event), the
    leftover ``_turn_end`` must not truncate the NEXT turn to zero events."""
    agent = AlanSubprocessAgent()
    await _start_scripted_child(agent, _ERROR_THEN_OK_CHILD, str(tmp_path))
    try:
        stream = agent.send("first")
        async for event in stream:
            if isinstance(event, AgentError):
                break  # abandon the turn the way the loop does
        await stream.aclose()
        events = [event async for event in agent.send("second")]
    finally:
        await agent.close()
    assert [type(event).__name__ for event in events] == ["TextDelta"]


async def test_runner_reports_systemexit_as_fatal(capsys) -> None:  # type: ignore[no-untyped-def]
    """``except Exception`` used to let SystemExit kill the child silently; it must
    emit a ``_fatal`` frame (then still terminate)."""

    class _Dies:
        async def query_events_async(self, message):  # type: ignore[no-untyped-def]
            raise SystemExit(3)
            yield  # makes this an async generator, like the real backend

    with pytest.raises(SystemExit):
        await _run_turn(_Dies(), "go")
    frames = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert {"type": FATAL, "message": "SystemExit: 3"} in frames
    assert frames[-1] == {"type": TURN_END}


@pytest.mark.live
@pytest.mark.slow
async def test_runner_starts_and_reports_ready(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """End-to-end: spawn the real runner, have it build alancode, then shut it down.

    Marked live+slow: it needs ``alancode`` installed, and importing its litellm
    dependency can take minutes on a loaded machine. It needs no model endpoint —
    only construction is exercised, not a turn.

    NOTE the child must get an ABSOLUTE PYTHONPATH: it runs with cwd=<workdir>, so a
    relative entry would no longer resolve (this is what ``task.py`` passes).
    """
    import os
    import sys

    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src"
    agent = AlanSubprocessAgent({"permission_mode": "yolo", "memory": "off"})
    try:
        await agent.start(
            cwd=str(tmp_path),
            model="openai/fake",
            base_url="http://127.0.0.1:9/v1",
            api_key="dummy",
            system_prompt="You are a test.",
            tools=[],
            env={"PYTHONPATH": os.pathsep.join([src, *sys.path[:1]])},
            runtime_wrap=None,
        )
    finally:
        await agent.close()
