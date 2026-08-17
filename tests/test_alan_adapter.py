"""Tests for the Alan adapter's event mapping.

The load-bearing invariants: ``query_events_async`` yields whole messages, not blocks —
streaming display deltas (``AssistantMessage`` with ``hide_in_api=True``) must be dropped
so text appears exactly once; the assembled ``AssistantMessage`` must be unpacked into
its events, deriving ``TurnComplete`` when it ends the turn (no tool call); and tool
results, which come back wrapped in a ``UserMessage`` content list, must be unpacked into
``ToolResult`` events. These tests pin that mapping with stand-ins that mimic alancode's
classes by name (the adapter dispatches on class name).
"""

import dataclasses
import sys
import types

from regact.agent.alan_adapter import build_alan_agent, map_alan_events
from regact.agent.events import (
    AgentError,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnComplete,
)


class TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class ThinkingBlock:
    def __init__(self, thinking: str) -> None:
        self.thinking = thinking


class ToolUseBlock:
    def __init__(self, id: str, name: str, input: dict) -> None:
        self.id, self.name, self.input = id, name, input


class ToolResultBlock:
    def __init__(self, tool_use_id: str, content, is_error: bool = False) -> None:
        self.tool_use_id, self.content, self.is_error = tool_use_id, content, is_error


@dataclasses.dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


class AssistantMessage:
    def __init__(
        self,
        content: list,
        *,
        is_api_error_message: bool = False,
        error_details: str | None = None,
        hide_in_api: bool = False,
        usage=None,
    ) -> None:
        self.content = content
        self.is_api_error_message = is_api_error_message
        self.error_details = error_details
        self.api_error = None
        self.hide_in_api = hide_in_api
        self.usage = usage


class UserMessage:
    def __init__(self, content) -> None:
        self.content = content


def test_assistant_message_unpacks_text_and_tool_call() -> None:
    msg = AssistantMessage([TextBlock("probing"), ToolUseBlock("i", "SubmitSolution", {"x": 1})])
    events = map_alan_events(msg)
    kinds = [type(e).__name__ for e in events]
    assert "TextDelta" in kinds and "ToolCall" in kinds
    tool = next(e for e in events if isinstance(e, ToolCall))
    assert tool.name == "SubmitSolution" and tool.input == {"x": 1}


def test_assistant_message_multiple_tool_calls() -> None:
    msg = AssistantMessage([ToolUseBlock("a", "RunPython", {}), ToolUseBlock("b", "ExitTask", {})])
    calls = [e for e in map_alan_events(msg) if isinstance(e, ToolCall)]
    assert [c.name for c in calls] == ["RunPython", "ExitTask"]


def test_assistant_api_error_becomes_agent_error() -> None:
    msg = AssistantMessage([], is_api_error_message=True, error_details="context window exceeded")
    events = map_alan_events(msg)
    assert len(events) == 1 and isinstance(events[0], AgentError)
    assert "context window" in events[0].message


def test_hidden_streaming_message_dropped() -> None:
    # Streaming display deltas re-carry text the assembled message already holds.
    assert map_alan_events(AssistantMessage([TextBlock("chunk")], hide_in_api=True)) == []
    assert map_alan_events(AssistantMessage([ThinkingBlock("hmm")], hide_in_api=True)) == []


def test_terminal_message_derives_turn_complete() -> None:
    msg = AssistantMessage([TextBlock("done.")], usage=Usage(input_tokens=10, output_tokens=2))
    events = map_alan_events(msg)
    assert [type(e).__name__ for e in events] == ["TextDelta", "TurnComplete"]
    turn = events[-1]
    assert turn.final_text == "done."
    assert turn.usage == {"input_tokens": 10, "output_tokens": 2}


def test_message_with_tool_call_ends_with_turn_complete() -> None:
    # Every completion is a turn boundary (one completion = one viz turn), so a tool-call
    # message also closes with a TurnComplete carrying that completion's usage.
    msg = AssistantMessage([TextBlock("running"), ToolUseBlock("i", "Bash", {})])
    events = map_alan_events(msg)
    assert [type(e).__name__ for e in events] == ["TextDelta", "ToolCall", "TurnComplete"]


def test_assistant_thinking_only() -> None:
    events = map_alan_events(AssistantMessage([ThinkingBlock("hmm")]))
    assert isinstance(events[0], ThinkingDelta)
    assert isinstance(events[-1], TurnComplete)  # no tool call: the turn ends here


def test_empty_text_block_dropped() -> None:
    # An empty TextBlock should not produce a spurious empty TextDelta.
    events = map_alan_events(AssistantMessage([TextBlock("")]))
    assert not any(isinstance(e, TextDelta) for e in events)


def test_user_message_unpacks_tool_results() -> None:
    msg = UserMessage([ToolResultBlock("i", "file contents"), ToolResultBlock("j", "boom", True)])
    events = map_alan_events(msg)
    assert [type(e).__name__ for e in events] == ["ToolResult", "ToolResult"]
    assert events[0].id == "i" and events[0].output == "file contents" and not events[0].is_error
    assert events[1].id == "j" and events[1].is_error


def test_user_message_block_list_content_flattened() -> None:
    msg = UserMessage([ToolResultBlock("i", [TextBlock("a"), TextBlock("b")])])
    (ev,) = map_alan_events(msg)
    assert ev.output == "ab"


def test_plain_text_user_message_maps_to_nothing() -> None:
    # Interruptions/reminders are model-facing context, not agent output.
    assert map_alan_events(UserMessage("Request interrupted")) == []


def test_full_turn_event_order() -> None:
    # One turn as alancode streams it: hidden deltas, assembled message with a tool
    # call, the tool's result, then the terminal text-only message.
    stream = [
        AssistantMessage([TextBlock("I'll ")], hide_in_api=True),
        AssistantMessage([TextBlock("run X")], hide_in_api=True),
        AssistantMessage([TextBlock("I'll run X"), ToolUseBlock("t1", "Bash", {"command": "x"})]),
        UserMessage([ToolResultBlock("t1", "ok")]),
        AssistantMessage([TextBlock("Done")], hide_in_api=True),
        AssistantMessage([TextBlock("Done")], usage=Usage(output_tokens=5)),
    ]
    events = [e for item in stream for e in map_alan_events(item)]
    assert [type(e).__name__ for e in events] == [
        "TextDelta",
        "ToolCall",
        "TurnComplete",  # the tool-call completion closes its own turn
        "ToolResult",
        "TextDelta",
        "TurnComplete",
    ]
    assert events[0].text == "I'll run X"  # once, not once per delta plus once assembled
    assert events[3].id == "t1" and events[3].output == "ok"
    assert events[5].final_text == "Done"


def test_legacy_single_blocks_still_map() -> None:
    # Older alancode builds may stream individual blocks/messages; _map handles those.
    assert isinstance(map_alan_events(TextBlock("hi"))[0], TextDelta)
    assert isinstance(map_alan_events(ToolUseBlock("i", "X", {}))[0], ToolCall)


def test_unknown_item_dropped() -> None:
    class Weird:
        pass

    assert map_alan_events(Weird()) == []


def test_legacy_tool_result_maps() -> None:
    (ev,) = map_alan_events(ToolResultBlock("i", "out"))
    assert isinstance(ev, ToolResult) and ev.output == "out"


def _fake_alancode(monkeypatch, builtins):
    """Install a fake ``alancode`` package (agent + tools.registry) for the builder."""
    captured: dict = {}

    class FakeAgent:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def update_session_setting(self, key, value):
            captured.setdefault("settings", {})[key] = value
            return None  # error-or-None; None = accepted

    fake = types.ModuleType("alancode")
    fake.AlanCodeAgent = FakeAgent
    registry = types.ModuleType("alancode.tools.registry")
    registry.get_all_builtin_tools = lambda: builtins
    registry.find_tool_by_name = lambda tools, name: next(
        (t for t in tools if t.name == name), None
    )
    monkeypatch.setitem(sys.modules, "alancode", fake)
    monkeypatch.setitem(sys.modules, "alancode.tools", types.ModuleType("alancode.tools"))
    monkeypatch.setitem(sys.modules, "alancode.tools.registry", registry)
    return captured


def test_build_alan_agent_forwards_backend(monkeypatch) -> None:
    bash = types.SimpleNamespace(name="Bash")
    others = [types.SimpleNamespace(name=n) for n in ("Read", "Write", "Edit")]
    captured = _fake_alancode(monkeypatch, [bash, *others])
    build_alan_agent(
        cwd=".",
        model="remote",
        base_url=None,
        api_key=None,
        system_prompt=None,
        extra_tools=[],
        args={"backend": "scripted"},
    )
    assert captured["backend"] == "scripted" and captured["model"] == "remote"


def test_build_alan_agent_restricts_to_bash_only(monkeypatch) -> None:
    """The bash-only agent: only the Bash tool is passed, Read/Write/Edit/... dropped."""
    bash = types.SimpleNamespace(name="Bash")
    others = [types.SimpleNamespace(name=n) for n in ("Read", "Write", "Edit", "Glob", "Grep")]
    captured = _fake_alancode(monkeypatch, [bash, *others])
    build_alan_agent(
        cwd=".",
        model="m",
        base_url=None,
        api_key=None,
        system_prompt=None,
        extra_tools=[],
        args={},
    )
    assert captured["tools"] == [bash]  # only Bash reaches the agent


def test_build_alan_agent_sets_escalated_max_tokens(monkeypatch) -> None:
    """escalated_max_tokens rides the SETTINGS API (not a ctor kwarg, which would leak to the
    LLM transport). Default applies; agent.args overrides it (string-coerced)."""
    captured = _fake_alancode(monkeypatch, [types.SimpleNamespace(name="Bash")])
    build_alan_agent(
        cwd=".", model="m", base_url=None, api_key=None, system_prompt=None, extra_tools=[], args={}
    )
    assert captured["settings"]["escalated_max_tokens"] == 12000  # default
    assert "escalated_max_tokens" not in captured  # NOT a constructor kwarg

    captured2 = _fake_alancode(monkeypatch, [types.SimpleNamespace(name="Bash")])
    build_alan_agent(
        cwd=".",
        model="m",
        base_url=None,
        api_key=None,
        system_prompt=None,
        extra_tools=[],
        args={"escalated_max_tokens": "9999"},
    )
    assert captured2["settings"]["escalated_max_tokens"] == 9999  # override, coerced to int
