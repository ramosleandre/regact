"""Tests for the agnostic agent layer: events, capabilities, ScriptedAgent, build_agent."""

from typing import ClassVar

import pytest

from regact.agent.base import CodeAgent, build_agent
from regact.agent.capabilities import Capabilities
from regact.agent.events import (
    AgentError,
    TextDelta,
    ToolCall,
    ToolResult,
    TurnComplete,
)
from regact.agent.scripted_agent import ScriptedAgent
from regact.config.schema import AgentConfig, AgentName
from regact.obs.errors import ErrorCategory


async def _drain(agent: CodeAgent, message: str) -> list[object]:
    return [event async for event in agent.send(message)]


async def test_scripted_agent_replays_turns() -> None:
    turns = [
        [TextDelta("thinking out loud"), ToolCall("t1", "make_env", {}), TurnComplete("done")],
        [TextDelta("second turn"), TurnComplete("bye")],
    ]
    agent = ScriptedAgent(turns)
    await agent.start(
        cwd="/tmp/x",
        model=None,
        base_url=None,
        api_key=None,
        system_prompt="sys",
    )
    assert agent.started

    first = await _drain(agent, "go")
    assert first == turns[0]
    second = await _drain(agent, "again")
    assert second == turns[1]
    assert agent.sent == ["go", "again"]


async def test_scripted_agent_default_turn_when_exhausted() -> None:
    agent = ScriptedAgent([])
    events = await _drain(agent, "go")
    assert len(events) == 1 and isinstance(events[0], TurnComplete)


async def test_scripted_agent_records_inject_abort_close() -> None:
    agent = ScriptedAgent()
    await agent.inject("hint")
    await agent.abort()
    await agent.close()
    assert agent.injected == ["hint"]
    assert agent.aborted is True
    assert agent.closed is True


def test_scripted_agent_capabilities() -> None:
    caps = ScriptedAgent().capabilities()
    assert isinstance(caps, Capabilities)
    assert caps.system_prompt == "replace"
    assert caps.control_actions == "native_tools"


def test_build_agent_scripted() -> None:
    agent = build_agent(AgentConfig(name=AgentName.SCRIPTED))
    assert isinstance(agent, ScriptedAgent)


def test_build_agent_alan_constructs_without_alancode() -> None:
    # Construction must not import alancode (only start() does).
    from regact.agent.alan_adapter import AlanAgent

    agent = build_agent(AgentConfig(name=AgentName.ALAN))
    assert isinstance(agent, AlanAgent)


def test_build_agent_unknown_raises() -> None:
    with pytest.raises(ValueError, match=r"unknown"):
        build_agent(AgentConfig(name="bogus"))  # type: ignore[arg-type]


def test_agent_event_union_members() -> None:
    err = AgentError(category=ErrorCategory.AGENT_API, message="429")
    assert err.category is ErrorCategory.AGENT_API
    res = ToolResult(id="t1", output="ok")
    assert res.is_error is False


def test_alan_event_mapping() -> None:
    """AlanAgent._map translates native blocks to the normalized union by class name."""
    from regact.agent.alan_adapter import AlanAgent

    class TextBlock:
        text = "hi"

    class ToolUseBlock:
        id = "t1"
        name = "submit_solution"
        input: ClassVar[dict[str, str]] = {"path": "c.py"}

    class ToolResultBlock:
        tool_use_id = "t1"
        content = "scored"
        is_error = False

    class ResultMessage:
        result = "final"
        usage: ClassVar[dict[str, int]] = {"in": 10}

    class APIError:
        message = "rate limited"

    class Unknown:
        pass

    m = AlanAgent._map
    assert m(TextBlock()) == TextDelta("hi")
    assert m(ToolUseBlock()) == ToolCall("t1", "submit_solution", {"path": "c.py"})
    assert m(ToolResultBlock()) == ToolResult("t1", "scored", False)
    assert m(ResultMessage()) == TurnComplete("final", {"in": 10})
    assert m(APIError()) == AgentError(ErrorCategory.AGENT_API, "rate limited")
    assert m(Unknown()) is None


@pytest.mark.live
def test_alan_start_requires_alancode() -> None:
    """Runtime-gated: only meaningful where alancode is installed."""
    pytest.importorskip("alancode")
    from regact.agent.alan_adapter import AlanAgent

    agent = AlanAgent()
    assert agent.capabilities().writes_native_transcript is True
