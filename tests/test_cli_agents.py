"""Unit tests for the CLI agent adapters (Claude + codex).

The meat is the stream-json → AgentEvent parsing and the command builder; both run
without the CLI installed. Actually spawning the CLI is a separate live concern.
"""

import os

from regact.agent.base import build_agent
from regact.agent.claude_adapter import ClaudeAgent
from regact.agent.codex_adapter import CodexAgent
from regact.agent.events import (
    AgentError,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnComplete,
)
from regact.config.schema import AgentConfig, AgentName


def test_build_agent_resolves_claude_and_codex() -> None:
    assert isinstance(build_agent(AgentConfig(name=AgentName.CLAUDE)), ClaudeAgent)
    assert isinstance(build_agent(AgentConfig(name=AgentName.CODEX)), CodexAgent)


def test_capabilities_mark_client_cli() -> None:
    assert ClaudeAgent().capabilities().control_actions == "client_cli"
    assert ClaudeAgent().capabilities().system_prompt == "append"
    assert CodexAgent().capabilities().control_actions == "client_cli"


def test_host_read_paths_are_per_agent() -> None:
    """Each backend declares only its OWN host dirs — never another backend's."""
    claude = ClaudeAgent().host_read_paths()
    codex = CodexAgent().host_read_paths()
    assert any(p.endswith("/.claude") for p in claude)
    assert any(p.endswith("/codex-home") for p in codex)  # codex's isolated CODEX_HOME
    assert not any("codex" in p for p in claude)  # no cross-contamination
    assert not any("/.claude" in p for p in codex)


def test_codex_uses_an_isolated_home(tmp_path) -> None:
    """codex runs against a generated home, not the user's ~/.codex, so no ambient
    config leaks into the session."""
    home = tmp_path / "codex-home"
    agent = CodexAgent({"codex_home": str(home)})
    real = os.path.realpath(str(home))
    assert agent.host_read_paths() == [real]

    agent._configure_workdir()  # what start() invokes to seed the home
    assert agent._env_overrides["CODEX_HOME"] == real
    assert agent._env_overrides["HOME"] == real  # also redirects ~/.agents
    assert (home / "skills").is_dir()
    assert (home / "config.toml").read_text().lstrip().startswith("#")


def test_host_egress_hosts_are_per_agent() -> None:
    assert ClaudeAgent().host_egress_hosts() == ["api.anthropic.com"]
    assert "api.openai.com" in CodexAgent().host_egress_hosts()
    assert not any("anthropic" in h for h in CodexAgent().host_egress_hosts())


# --- Claude stream-json parsing ------------------------------------------- #


def test_claude_tracks_session_id() -> None:
    agent = ClaudeAgent()
    agent._track_session({"type": "system", "subtype": "init", "session_id": "sess-1"})
    assert agent._session_id == "sess-1"


def test_claude_parses_assistant_text_and_tool_use() -> None:
    obj = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "I'll list files."},
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
            ]
        },
    }
    events = ClaudeAgent()._parse_events(obj)
    assert events == [
        TextDelta("I'll list files."),
        ToolCall("t1", "Bash", {"command": "ls"}),
    ]


def test_claude_parses_tool_result_and_result() -> None:
    agent = ClaudeAgent()
    user = {
        "type": "user",
        "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
    }
    assert agent._parse_events(user) == [ToolResult("t1", "ok", False)]

    done = {"type": "result", "subtype": "success", "result": "all done", "usage": {"in": 5}}
    assert agent._parse_events(done) == [TurnComplete("all done", {"in": 5})]


def test_claude_result_error_becomes_agent_error() -> None:
    obj = {"type": "result", "subtype": "error_max_turns", "is_error": True, "result": "too many"}
    [event] = ClaudeAgent()._parse_events(obj)
    assert isinstance(event, AgentError)
    assert event.message == "too many"


def test_claude_command_first_turn_then_resume() -> None:
    agent = ClaudeAgent()
    agent._system_prompt = "be good"
    argv, stdin = agent._command("go")
    assert stdin is None
    assert argv[:3] == ["claude", "-p", "go"]
    assert "--append-system-prompt" in argv and "be good" in argv

    agent._session_id = "sess-1"
    argv2, _ = agent._command("again")
    assert "--resume" in argv2 and "sess-1" in argv2
    assert "--append-system-prompt" not in argv2  # resume carries the prior context


# --- codex ndjson parsing (best-effort schema) ---------------------------- #


def test_codex_tracks_thread_id() -> None:
    agent = CodexAgent()
    agent._track_session({"type": "thread.started", "thread_id": "th-1"})
    assert agent._session_id == "th-1"


def test_codex_parses_message_reasoning_command_and_completion() -> None:
    agent = CodexAgent()
    assert agent._parse_events({"type": "item.completed", "item": {"text": "hello"}}) == [
        TextDelta("hello")
    ]
    assert agent._parse_events(
        {"type": "item.completed", "item": {"type": "reasoning", "text": "hmm"}}
    ) == [ThinkingDelta("hmm")]
    # a command: clean ToolCall on start, ToolResult (paired by id) on completion
    [call] = agent._parse_events(
        {
            "type": "item.started",
            "item": {"type": "command_execution", "command": "ls", "id": "c1"},
        }
    )
    assert isinstance(call, ToolCall) and call.name == "shell" and call.input == {"command": "ls"}
    done = {"type": "command_execution", "id": "c1", "aggregated_output": "x", "exit_code": 0}
    [res] = agent._parse_events({"type": "item.completed", "item": done})
    assert isinstance(res, ToolResult) and res.id == "c1" and res.output == "x"
    assert not res.is_error
    # an intermediate update of the same item is dropped (no duplicate ToolCall)
    assert (
        agent._parse_events(
            {"type": "item.updated", "item": {"type": "command_execution", "id": "c1"}}
        )
        == []
    )
    assert agent._parse_events({"type": "turn.completed", "item": {"text": "fin"}}) == [
        TurnComplete("fin")
    ]


def test_codex_command_pipes_prompt_on_stdin() -> None:
    agent = CodexAgent()
    agent._cwd = "/tmp/wd"
    argv, stdin = agent._command("solve it")
    assert stdin == "solve it"  # codex reads the prompt from stdin
    assert "exec" in argv and "--json" in argv and "--cd" in argv
    assert os.path.isabs(argv[argv.index("--cd") + 1])  # absolute, else codex re-nests it


def test_codex_resume_puts_exec_flags_before_the_subcommand() -> None:
    """--cd/--json are exec options; `exec resume` rejects them, so they must precede it."""
    agent = CodexAgent()
    agent._cwd = "/tmp/wd"
    agent._session_id = "th-1"
    argv, _ = agent._command("again")
    assert argv.index("--cd") < argv.index("resume")
    assert argv.index("--json") < argv.index("resume")
    assert argv[argv.index("resume") + 1] == "th-1"
