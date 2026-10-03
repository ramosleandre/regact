"""Unit tests for the CLI agent adapters (Claude + codex).

The meat is the stream-json → AgentEvent parsing and the command builder; both run
without the CLI installed. Actually spawning the CLI is a separate live concern.
"""

import asyncio
import json
import os
import sys
import tomllib
from pathlib import Path

import pytest

from regact.agent.base import build_agent
from regact.agent.claude_adapter import ClaudeAgent
from regact.agent.codex_adapter import CodexAgent
from regact.agent.events import (
    AgentError,
    IterationComplete,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
)
from regact.config.schema import AgentConfig, AgentName


def test_build_agent_resolves_claude_and_codex() -> None:
    assert isinstance(build_agent(AgentConfig(name=AgentName.CLAUDE)), ClaudeAgent)
    assert isinstance(build_agent(AgentConfig(name=AgentName.CODEX)), CodexAgent)


def test_capabilities_mark_client_cli() -> None:
    assert ClaudeAgent().capabilities().tool_protocol == "client_cli"
    assert ClaudeAgent().capabilities().system_prompt == "append"
    assert CodexAgent().capabilities().tool_protocol == "client_cli"


def test_host_paths_are_per_agent_and_ambient_config_stays_out() -> None:
    """Each backend declares only its OWN dirs, and never the user-level config roots
    (~/.claude read-only, ~/.codex) — those hold other sessions' transcripts and history."""
    home = os.path.expanduser("~")
    claude_ro = ClaudeAgent().host_read_paths()
    codex_rw = CodexAgent().host_rw_paths()
    assert os.path.join(home, ".claude") not in claude_ro  # never in the read-only set
    # .claude.json IS exposed read-only: the CLI will not start a session without it.
    assert os.path.join(home, ".claude.json") in claude_ro
    assert any("codex-home" in p for p in codex_rw)  # under codex's isolated CODEX_HOME root
    assert not any("claude" in p for p in codex_rw)  # no cross-contamination


def test_claude_config_dir_isolates_when_forced(tmp_path) -> None:
    """Forcing ``claude_home`` relocates the config dir to a FRESH per-task home UNDER it (not the
    shared root, so nothing accumulates); host_rw_paths binds that dir, never codex's."""
    isolated = str(tmp_path / "claude-home")
    agent = ClaudeAgent({"claude_home": isolated})
    rw = agent.host_rw_paths()
    root = os.path.realpath(isolated)
    assert len(rw) == 1 and rw[0].startswith(root + os.sep) and rw[0] != root
    assert not any("codex" in p for p in rw)


def test_claude_config_dir_falls_back_to_real_home_for_keychain_auth(
    monkeypatch,
) -> None:
    """With no forced home and no copyable .credentials.json (macOS Keychain auth),
    relocating would strand the CLI as 'Not logged in', so it uses the real ~/.claude."""
    monkeypatch.setattr(os.path, "exists", lambda p: False)  # no .credentials.json anywhere
    agent = ClaudeAgent()  # no claude_home forced
    assert agent._config_dir() == os.path.join(os.path.expanduser("~"), ".claude")


def test_codex_uses_an_isolated_home(tmp_path) -> None:
    """codex runs against a FRESH per-task home under the root (not the user's ~/.codex, and not the
    shared root), so no ambient config or a prior task's session store leaks in."""
    home = tmp_path / "codex-home"
    agent = CodexAgent({"codex_home": str(home)})
    root = os.path.realpath(str(home))
    rw = agent.host_rw_paths()
    assert len(rw) == 1 and rw[0].startswith(root + os.sep) and rw[0] != root
    # Only the CLI's install dirs are readable — never the ambient config root.
    assert os.path.realpath(os.path.expanduser("~/.codex")) not in agent.host_read_paths()

    agent._configure_workdir()  # what start() invokes to seed the home
    cfg = rw[0]
    assert agent._env_overrides["CODEX_HOME"] == cfg
    assert agent._env_overrides["HOME"] == cfg  # also redirects ~/.agents
    assert os.path.isdir(os.path.join(cfg, "skills"))
    assert Path(cfg, "config.toml").read_text().lstrip().startswith("#")


def test_alan_runner_regact_closure_is_bound_but_scoring_is_not() -> None:
    """Alan's confined child runs `python -m regact.agent.alan_runner`; F1's default agent bind is
    only regact.envclient + netbridge, so alan must add the runner's regact closure (else it dies
    with ModuleNotFoundError, like the netbridge regression). The closure files must exist and must
    NOT include the scoring / jail source (regact.controllers / regact.security.runtime)."""
    from regact.agent.alan_subprocess import _runner_regact_paths

    paths = _runner_regact_paths()
    assert paths and all(os.path.exists(p) for p in paths)
    assert any(p.endswith(os.path.join("agent", "alan_runner.py")) for p in paths)
    assert any(p.endswith(os.path.join("agent", "alan_adapter.py")) for p in paths)
    assert not any("controllers" in p or "runtime.py" in p for p in paths)


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
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Bash",
                    "input": {"command": "ls"},
                },
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

    done = {
        "type": "result",
        "subtype": "success",
        "result": "all done",
        "usage": {"in": 5},
    }
    assert agent._parse_events(done) == [IterationComplete("all done", {"in": 5})]


def test_claude_result_error_becomes_agent_error() -> None:
    obj = {
        "type": "result",
        "subtype": "error_max_turns",
        "is_error": True,
        "result": "too many",
    }
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
    done = {
        "type": "command_execution",
        "id": "c1",
        "aggregated_output": "x",
        "exit_code": 0,
    }
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
        IterationComplete("fin")
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


def test_executable_paths_cover_the_symlink_dir_and_the_real_install_dir(
    tmp_path, monkeypatch
) -> None:
    """Installers put a symlink on PATH and the real binary in a versioned tree; the
    sandbox must see BOTH dirs or execvp dies on the dangling link."""
    from regact.agent.base import executable_paths

    install = tmp_path / "share" / "tool" / "versions"
    install.mkdir(parents=True)
    real = install / "tool-1.0"
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "tool").symlink_to(real)

    monkeypatch.setenv("PATH", str(bin_dir))
    paths = executable_paths("tool")
    assert os.path.realpath(str(bin_dir)) in paths
    assert os.path.realpath(str(install)) in paths

    assert executable_paths("definitely-absent-tool-xyz") == []


def test_claude_uses_an_isolated_config_dir(tmp_path, monkeypatch) -> None:
    """claude runs against a generated CLAUDE_CONFIG_DIR, not ~/.claude: other
    sessions' transcripts and the prompt history stay invisible; only auth is seeded."""
    user_home = tmp_path / "userhome"
    (user_home / ".claude").mkdir(parents=True)
    (user_home / ".claude" / ".credentials.json").write_text("{}")
    (user_home / ".claude" / "history.jsonl").write_text("secret past prompt\n")
    monkeypatch.setenv("HOME", str(user_home))

    home = tmp_path / "claude-home"
    agent = ClaudeAgent({"claude_home": str(home)})
    agent._cwd = str(tmp_path / "wd")
    os.makedirs(agent._cwd, exist_ok=True)
    agent._configure_workdir()

    root = os.path.realpath(str(home))
    cfg = agent._env_overrides["CLAUDE_CONFIG_DIR"]
    assert cfg.startswith(root + os.sep) and cfg != root  # a fresh per-task home under the root
    assert Path(cfg, ".credentials.json").read_text() == "{}"  # auth seeded...
    assert not os.path.exists(os.path.join(cfg, "history.jsonl"))  # ...and nothing else
    assert agent.host_rw_paths() == [cfg]
    assert str(user_home / ".claude") not in agent.host_read_paths()


async def test_claude_config_home_is_per_task_and_cleaned(tmp_path) -> None:
    """Each task gets its OWN empty config home seeded with only auth (so no memory/history from a
    prior task leaks in), and the home is removed on close so nothing accumulates."""
    home = tmp_path / "claude-home"
    home.mkdir()
    (home / ".credentials.json").write_text("{}")  # the persistent login at the root
    a1 = ClaudeAgent({"claude_home": str(home)})
    a2 = ClaudeAgent({"claude_home": str(home)})
    d1, d2 = a1._config_dir(), a2._config_dir()
    assert d1 != d2  # different tasks -> different homes (no shared, accumulating dir)
    assert Path(d1, ".credentials.json").read_text() == "{}"  # auth seeded
    assert not os.path.exists(os.path.join(d1, "projects"))  # no memory dir
    await a1.close()
    assert not os.path.exists(d1)  # cleaned on teardown


@pytest.mark.parametrize("prompt", [None, "", 'Quotes " and \\ paths\nLéandre 😀', "x" * 200_000])
async def test_codex_task_prompt_available_on_initial_and_resumed_launch(
    tmp_path, monkeypatch, prompt
):
    agent = CodexAgent({"codex_home": str(tmp_path / "home")})
    monkeypatch.setattr(agent, "_freshest_auth", lambda: None)
    cwd = tmp_path / "workdir"
    cwd.mkdir()
    await agent.start(cwd=str(cwd), model=None, base_url=None, api_key=None, system_prompt=prompt)
    home = Path(agent._env_overrides["CODEX_HOME"])
    try:
        assert agent.capabilities().system_prompt == "append"
        for session in (None, "test-session"):
            agent._session_id = session
            argv, stdin = agent._command("Begin working.")
            config = tomllib.loads((home / "config.toml").read_text())
            assert config.get("developer_instructions") == prompt
            assert "model_instructions_file" not in config
            assert stdin == "Begin working."
            assert ("resume" in argv) == (session is not None)
            if prompt:
                assert prompt not in argv
    finally:
        await agent.close()
    assert not home.exists()


@pytest.mark.parametrize(
    "item",
    [
        {"type": "command_execution", "command": "echo hello", "exit_code": 0},
        {
            "type": "file_change",
            "changes": [
                {"path": "a.py", "kind": "update"},
                {"path": "b.py", "kind": "add"},
            ],
            "status": "completed",
        },
        {
            "type": "web_search",
            "query": "example",
            "action": {"type": "search", "query": "example"},
        },
        {
            "type": "mcp_tool_call",
            "server": "local",
            "tool": "read",
            "arguments": {},
            "result": {"content": [{"type": "text", "text": "ok"}]},
        },
        {
            "type": "collab_tool_call",
            "tool": "spawn_agent",
            "sender_thread_id": "main",
            "receiver_thread_ids": ["child"],
            "status": "completed",
        },
    ],
)
@pytest.mark.parametrize("with_start", [True, False])
def test_codex_counts_each_tool_once(item, with_start) -> None:
    agent = CodexAgent()
    item = {"id": "item_0", **item}
    events = []
    if with_start:
        started = {"type": "item.started", "item": {**item, "status": "in_progress"}}
        events += agent._parse_events(started)
        assert agent._parse_events(started) == []
    # Even a progress event with terminal-looking fields is not a completion.
    assert agent._parse_events({"type": "item.updated", "item": item}) == []
    done = {"type": "item.completed", "item": item}
    events += agent._parse_events(done)
    assert len(events) == 2
    assert isinstance(events[0], ToolCall)
    assert isinstance(events[1], ToolResult)
    assert events[0].id == events[1].id
    assert not events[1].is_error
    assert agent._parse_events(done) == []
    if item["type"] == "file_change":
        assert events[0].name == "apply_patch"
        assert len(events[0].input["changes"]) == 2  # one patch, not two calls
    if item["type"] == "mcp_tool_call":
        assert '"ok"' in events[1].output


@pytest.mark.parametrize(
    "item",
    [
        {"type": "file_change", "status": "failed", "changes": []},
        {"type": "command_execution", "status": "declined", "command": "false"},
        {"type": "command_execution", "exit_code": 1, "command": "false"},
        {
            "type": "mcp_tool_call",
            "status": "failed",
            "error": {"message": "unavailable"},
        },
        {"type": "collab_tool_call", "status": "failed", "tool": "spawn_agent"},
    ],
)
def test_codex_counts_failed_tools_and_records_error(item) -> None:
    call, result = CodexAgent()._parse_events(
        {"type": "item.completed", "item": {"id": "x", **item}}
    )
    assert isinstance(call, ToolCall)
    assert isinstance(result, ToolResult) and result.is_error


def test_codex_plan_updates_are_completed_calls_not_a_turn_long_pending_tool() -> None:
    agent = CodexAgent()
    item = {
        "id": "plan",
        "type": "todo_list",
        "items": [{"text": "test", "completed": False}],
    }
    events = []
    for kind in ("item.started", "item.updated", "item.completed"):
        events += agent._parse_events({"type": kind, "item": item})
    assert [type(e) for e in events] == [ToolCall, ToolResult, ToolCall, ToolResult]
    assert events[0].id != events[2].id


def test_codex_item_ids_restart_on_resumed_process() -> None:
    agent = CodexAgent()
    done = {
        "type": "item.completed",
        "item": {
            "id": "item_0",
            "type": "file_change",
            "changes": [],
            "status": "completed",
        },
    }
    assert len(agent._parse_events(done)) == 2
    agent._session_id = "existing-thread"
    agent._command("continue")
    assert len(agent._parse_events(done)) == 2


@pytest.mark.parametrize("enabled", [True, False, None])
def test_codex_subagent_control_is_explicit_and_survives_resume(enabled) -> None:
    agent = CodexAgent({"subagents_enabled": enabled})
    for session in (None, "existing-thread"):
        agent._session_id = session
        argv, _ = agent._command("go")
        if enabled is None:
            assert not any("agents.enabled=" in arg for arg in argv)
        else:
            assert f"agents.enabled={str(enabled).lower()}" in argv
            assert f"features.multi_agent={str(enabled).lower()}" in argv
            assert "features.multi_agent_v2=false" in argv


def test_codex_rejects_string_boolean_for_subagents() -> None:
    with pytest.raises(ValueError, match="subagents_enabled"):
        CodexAgent({"subagents_enabled": "false"})


@pytest.mark.parametrize("session_id", [None, "existing-session"])
@pytest.mark.parametrize("sandbox", [None, "workspace-write"])
def test_codex_disables_web_search_on_every_launch(session_id, sandbox) -> None:
    agent = CodexAgent({"sandbox": sandbox})
    agent._session_id = session_id
    argv, stdin = agent._command("continue")
    overrides = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-c"]
    assert [value for value in overrides if value.startswith("web_search=")] == [
        'web_search="disabled"'
    ]
    assert argv.index('web_search="disabled"') < argv.index("exec")
    assert ("resume" in argv) == (session_id is not None)
    assert stdin == "continue"


async def test_claude_base_url_points_at_a_local_anthropic_server(tmp_path) -> None:
    """agent.base_url (already recorded in config.json) drives Claude Code at a self-hosted
    Anthropic-compatible server, with no egress needed for a loopback one."""
    root = tmp_path / "claude-home"
    root.mkdir()
    (root / ".credentials.json").write_text("{}")
    agent = ClaudeAgent({"claude_home": str(root), "context_window": "131072"})
    cwd = tmp_path / "workdir"
    cwd.mkdir()
    await agent.start(
        cwd=str(cwd),
        model="DeepSeek-V4-Flash",
        base_url="http://127.0.0.1:8080",
        api_key=None,
        system_prompt=None,
    )
    try:
        env = agent._env_overrides
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "local" and env["ANTHROPIC_API_KEY"] == ""
        assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"
        assert agent.host_egress_hosts() == []
    finally:
        await agent.close()
    plain = ClaudeAgent({"claude_home": str(root)})
    assert plain.host_egress_hosts() == ["api.anthropic.com"]


async def test_codex_base_url_writes_a_local_responses_provider(tmp_path, monkeypatch) -> None:
    agent = CodexAgent(
        {"codex_home": str(tmp_path / "home"), "context_window": 120000, "max_output_tokens": 8192}
    )
    monkeypatch.setattr(agent, "_freshest_auth", lambda: None)
    cwd = tmp_path / "workdir"
    cwd.mkdir()
    await agent.start(
        cwd=str(cwd),
        model="DeepSeek-V4-Flash",
        base_url="http://10.0.0.5:8080",
        api_key=None,
        system_prompt="Solve the game.",
    )
    try:
        config = tomllib.loads(Path(agent._env_overrides["CODEX_HOME"], "config.toml").read_text())
        assert config["developer_instructions"] == "Solve the game."
        assert config["model_provider"] == "local"
        assert config["model_context_window"] == 120000
        assert config["model_max_output_tokens"] == 8192
        provider = config["model_providers"]["local"]
        assert provider["base_url"] == "http://10.0.0.5:8080/v1"
        assert provider["wire_api"] == "responses"
        assert agent.host_egress_hosts() == ["10.0.0.5"]  # a remote endpoint still needs egress
    finally:
        await agent.close()


@pytest.mark.parametrize("vision", [False, True])
async def test_agent_vision_gates_the_cli_image_tools(tmp_path, monkeypatch, vision) -> None:
    """A text-only serve returns HTTP 500 for any request carrying an image (Codex hit it by
    viewing its own renders), so without agent.vision neither CLI may send one."""
    root = tmp_path / "claude-home"
    root.mkdir()
    (root / ".credentials.json").write_text("{}")
    claude = ClaudeAgent({"claude_home": str(root)}, vision=vision)
    cwd = tmp_path / "claude-work"
    cwd.mkdir()
    await claude.start(cwd=str(cwd), model="m", base_url=None, api_key=None, system_prompt=None)
    try:
        deny = json.loads((cwd / ".claude" / "settings.json").read_text())["permissions"]["deny"]
        assert ("Read(**/*.png)" in deny) is not vision
    finally:
        await claude.close()

    codex = CodexAgent({"codex_home": str(tmp_path / "codex-home")}, vision=vision)
    monkeypatch.setattr(codex, "_freshest_auth", lambda: None)
    work = tmp_path / "codex-work"
    work.mkdir()
    await codex.start(cwd=str(work), model="m", base_url=None, api_key=None, system_prompt=None)
    try:
        argv, _ = codex._command("go")
        assert ("features.view_image=false" in argv) is not vision
    finally:
        await codex.close()


class _SleepyCli(CodexAgent):
    """A CLI agent whose 'CLI' is a process that just sleeps, so abort() can kill it mid-turn."""

    def _command(self, message: str) -> tuple[list[str], str | None]:
        return [sys.executable, "-c", "import time; time.sleep(30)"], None


async def test_a_cli_we_killed_is_not_reported_as_an_agent_error(tmp_path) -> None:
    """The walltime watchdog SIGKILLs the CLI; that exit is ours, not an API failure. Before the
    fix every capped run logged 'CLI exited with code -9' next to its walltime_limit verdict."""
    agent = _SleepyCli({"codex_home": str(tmp_path / "home")})
    cwd = tmp_path / "w"
    cwd.mkdir()
    await agent.start(cwd=str(cwd), model=None, base_url=None, api_key=None, system_prompt=None)

    async def kill_soon() -> None:
        await asyncio.sleep(0.3)
        await agent.abort()

    killer = asyncio.create_task(kill_soon())
    events = [event async for event in agent.send("go")]
    await killer
    await agent.close()
    assert not any(isinstance(e, AgentError) for e in events)


async def test_a_cli_that_dies_on_its_own_is_still_an_agent_error(tmp_path) -> None:
    class _Crashing(CodexAgent):
        def _command(self, message: str) -> tuple[list[str], str | None]:
            return [sys.executable, "-c", "raise SystemExit(3)"], None

    agent = _Crashing({"codex_home": str(tmp_path / "home")})
    cwd = tmp_path / "w"
    cwd.mkdir()
    await agent.start(cwd=str(cwd), model=None, base_url=None, api_key=None, system_prompt=None)
    events = [event async for event in agent.send("go")]
    await agent.close()
    assert any(isinstance(e, AgentError) and "code 3" in e.message for e in events)
