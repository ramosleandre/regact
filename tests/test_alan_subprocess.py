"""Unit tests: the out-of-process Alan backend and the wire format it rides on.

None of these need ``alancode`` — they cover the parts regact owns: the event
round-trip that the child/parent protocol depends on, the registry wiring, the
declared capabilities, and how a child frame becomes an event.
"""

from __future__ import annotations

import pytest

from regact.agent.alan_runner import FATAL, READY, TURN_END
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
