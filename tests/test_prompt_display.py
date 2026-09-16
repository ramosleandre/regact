"""New transcripts contain backend prompt text; old transcripts need no migration."""

from types import SimpleNamespace

import pytest

from regact.agent.alan_runner import READY, _assembled_prompt
from regact.agent.alan_subprocess import AlanSubprocessAgent
from regact.agent.claude_adapter import ClaudeAgent
from regact.agent.codex_adapter import CodexAgent
from regact.agent.events import SystemPrompt
from regact.obs.transcript import event_to_json
from regact.viz.reader import _group_turns


@pytest.mark.parametrize(
    "agent,marker,role",
    [
        (ClaudeAgent(), "[Claude Code system prompt", "[Appended system prompt]"),
        (CodexAgent(), "[Codex system prompt", "[Developer instructions]"),
    ],
)
def test_cli_markers_inside_existing_system_panel(agent, marker, role):
    text = agent.prompt_for_transcript("task instructions")
    item = _group_turns([event_to_json(SystemPrompt(text))])[0].items[0]
    assert item.kind == "system"
    assert item.text.startswith(marker)
    assert role not in item.text
    assert item.text.endswith("task instructions")


async def test_alan_builder_sections_survive_child_protocol_and_viewer():
    sections = ["custom prompt", "generated tool instructions\nBash(schema)"]
    text = _assembled_prompt(SimpleNamespace(build_system_prompt=lambda: (sections, 1)))
    agent = AlanSubprocessAgent()

    async def frames():
        yield {"type": READY, "system_prompt": text}

    agent._read_frames = frames
    await agent._await_ready()
    actual = agent.prompt_for_transcript("must not replace extracted text")
    item = _group_turns([event_to_json(SystemPrompt(actual))])[0].items[0]
    assert item.text == "\n\n".join(sections)


def test_unavailable_alan_builder_is_explicit():
    assert "unavailable" in _assembled_prompt(object())
    assert "unavailable" in AlanSubprocessAgent().prompt_for_transcript("prepared")


def test_old_system_prompt_is_unchanged():
    item = _group_turns([{"type": "SystemPrompt", "text": "old text"}])[0].items[0]
    assert item.text == "old text"
