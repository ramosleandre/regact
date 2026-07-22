"""Tests for the Alan adapter's event mapping.

The load-bearing invariant: ``query_events_async`` yields whole ``AssistantMessage``
objects (a ``.content`` list of blocks), NOT individual blocks. The adapter must UNPACK
each message into normalized events — otherwise every tool call is dropped, the loop sees
an empty stream, and the agent silently does nothing. These tests pin that unpacking with
stand-ins that mimic alancode's classes by name (the adapter dispatches on class name).
"""

from regact.agent.alan_adapter import AlanAgent
from regact.agent.events import AgentError, TextDelta, ThinkingDelta, ToolCall, ToolResult


class TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class ThinkingBlock:
    def __init__(self, thinking: str) -> None:
        self.thinking = thinking


class ToolUseBlock:
    def __init__(self, id: str, name: str, input: dict) -> None:
        self.id, self.name, self.input = id, name, input


class AssistantMessage:
    def __init__(self, content: list, *, is_api_error_message: bool = False,
                 error_details: str | None = None) -> None:
        self.content = content
        self.is_api_error_message = is_api_error_message
        self.error_details = error_details
        self.api_error = None


def test_assistant_message_unpacks_text_and_tool_call() -> None:
    msg = AssistantMessage([TextBlock("probing"), ToolUseBlock("i", "SubmitSolution", {"x": 1})])
    events = AlanAgent._map_all(msg)
    kinds = [type(e).__name__ for e in events]
    assert "TextDelta" in kinds and "ToolCall" in kinds
    tool = next(e for e in events if isinstance(e, ToolCall))
    assert tool.name == "SubmitSolution" and tool.input == {"x": 1}


def test_assistant_message_multiple_tool_calls() -> None:
    msg = AssistantMessage([ToolUseBlock("a", "RunPython", {}), ToolUseBlock("b", "ExitTask", {})])
    calls = [e for e in AlanAgent._map_all(msg) if isinstance(e, ToolCall)]
    assert [c.name for c in calls] == ["RunPython", "ExitTask"]


def test_assistant_api_error_becomes_agent_error() -> None:
    msg = AssistantMessage([], is_api_error_message=True, error_details="context window exceeded")
    events = AlanAgent._map_all(msg)
    assert len(events) == 1 and isinstance(events[0], AgentError)
    assert "context window" in events[0].message


def test_assistant_thinking_only() -> None:
    events = AlanAgent._map_all(AssistantMessage([ThinkingBlock("hmm")]))
    assert len(events) == 1 and isinstance(events[0], ThinkingDelta)


def test_empty_text_block_dropped() -> None:
    # An empty TextBlock should not produce a spurious empty TextDelta.
    events = AlanAgent._map_all(AssistantMessage([TextBlock("")]))
    assert events == []


def test_legacy_single_blocks_still_map() -> None:
    # Older alancode builds may stream individual blocks/messages; _map handles those.
    assert isinstance(AlanAgent._map_all(TextBlock("hi"))[0], TextDelta)
    assert isinstance(AlanAgent._map_all(ToolUseBlock("i", "X", {}))[0], ToolCall)


def test_unknown_item_dropped() -> None:
    class Weird:
        pass

    assert AlanAgent._map_all(Weird()) == []


def test_legacy_tool_result_maps() -> None:
    class ToolResultBlock:
        def __init__(self) -> None:
            self.tool_use_id, self.content, self.is_error = "i", "out", False

    (ev,) = AlanAgent._map_all(ToolResultBlock())
    assert isinstance(ev, ToolResult) and ev.output == "out"
