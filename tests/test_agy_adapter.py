"""The Antigravity CLI adapter, on the event stream agy 1.4.7 writes (no agy needed)."""

import asyncio

from regact.agent.agy_adapter import AgyAgent
from regact.agent.events import AgentError, IterationComplete, TextDelta, ToolCall, ToolResult


def _step(index, step_type, state, **extra):
    return {
        "event": "step_update",
        "step_update": {"step_index": index, "step_type": step_type, "state": state, **extra},
    }


_USAGE = {"input_tokens": 100, "output_tokens": 10, "thinking_tokens": 5, "total_tokens": 115}
_STREAM = [
    {"event": "init", "conversation_id": "conv-1", "init": {"cwd": "/w", "tools": []}},
    _step(0, "user_input", "DONE"),
    _step(1, "agent_response", "DONE", usage=_USAGE),
    _step(
        2,
        "tool",
        "ACTIVE",
        tool_name="run_command",
        tool_info={"name": "run_command", "parameters": {"CommandLine": "ls"}},
    ),
    _step(
        2,
        "tool",
        "DONE",
        tool_name="run_command",
        tool_info={"name": "run_command", "parameters": {"CommandLine": "ls"}, "output": "a.txt"},
    ),
    _step(
        3,
        "tool",
        "DONE",
        tool_name="write_to_file",
        tool_info={"name": "write_to_file", "parameters": {"TargetFile": "/w/c.py"}},
    ),
    _step(4, "agent_response", "ACTIVE", text_delta="The files "),
    _step(4, "agent_response", "DONE", text_delta="are listed.", usage=_USAGE),
    {
        "event": "result",
        "result": {"status": "SUCCESS", "response": "The files are listed.", "usage": _USAGE},
    },
]


def _agent(tmp_path, **args):
    agent = AgyAgent({"agy_home": str(tmp_path / "home"), **args})
    workdir = tmp_path / "run" / "workdir"
    workdir.mkdir(parents=True)
    asyncio.run(
        agent.start(
            cwd=str(workdir),
            model="gemini-3.8-flash-high",
            base_url=None,
            api_key=None,
            system_prompt="Follow the rules.",
        )
    )
    return agent


def test_stream_becomes_normalized_events(tmp_path):
    agent = _agent(tmp_path)
    agent._processes = 1
    events = []
    for obj in _STREAM:
        agent._track_session(obj)
        events += agent._parse_events(obj)
    assert agent.session_id() == "conv-1"
    assert events[0] == ToolCall("p1:2", "run_command", {"command": "ls"})
    assert events[1] == ToolResult("p1:2", "a.txt", False)
    # A tool reported only once it finished still gets its call.
    assert events[2] == ToolCall("p1:3", "write_to_file", {"file_path": "/w/c.py"})
    assert isinstance(events[3], ToolResult)
    assert events[4] == TextDelta("The files are listed.")  # one message, not one per piece
    assert isinstance(events[5], IterationComplete)
    # Usage is counted per response (the final result's total is not added again).
    assert agent.usage() == {
        "source": "agy_stream",
        "models": {"gemini-3.8-flash-high": {"input": 200, "output": 20, "thinking": 10}},
    }
    asyncio.run(agent.close())


def test_the_agent_definition_allows_only_shell_and_file_tools(tmp_path):
    agent = _agent(tmp_path)
    home = agent._config_dir()
    definition = (tmp_path / "home" / "session").glob("*/.gemini/agents/regact.md")
    text = next(definition).read_text()
    assert "Follow the rules." in text and "run_command" in text
    for tool in ("search_web", "read_url_content", "browser", "view_file", "generate_image"):
        assert tool not in text
    assert all(path.startswith(home) for path in agent.host_rw_paths())
    argv, stdin = agent._command("go on")
    assert (
        argv[:3] == ["agy", "--agent", "regact"] and "--conversation" not in argv and stdin is None
    )
    agent._session_id = "conv-1"
    assert agent._command("go on")[0][-2:] == ["--conversation", "conv-1"]
    asyncio.run(agent.close())
    assert not (tmp_path / "home" / "session").exists() or not any(
        (tmp_path / "home" / "session").iterdir()
    )


def test_a_failed_login_says_why(tmp_path):
    agent = _agent(tmp_path)
    [error] = agent._parse_events(
        {
            "event": "result",
            "result": {"status": "ERROR", "error": "authentication failed or timed out"},
        }
    )
    assert isinstance(error, AgentError) and "sandbox=false" in error.message
    asyncio.run(agent.close())
