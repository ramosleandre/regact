"""Warning timing: keyword/result/server flags, caps, and Alan's result handshake."""

import asyncio
from dataclasses import replace
from unittest.mock import Mock

import pytest

from regact.agent.events import ToolCall, ToolResult, UserMessage
from regact.agent.scripted_agent import ScriptedAgent
from regact.orchestration.loop import _dispatch_event, _LoopContext, _run_turn, _TurnOutcome
from regact.protocols.policy_search import PolicySearchSession
from regact.security.policy import default_policy
from regact.session.state import ExperimentState


def context(agent=None, cap=3):
    state = ExperimentState(problem_name="p", task_name="t")
    return _LoopContext(
        agent=agent or ScriptedAgent(),
        experiment=state,
        tools_by_name={},
        transcript=Mock(),
        logger=Mock(),
        cwd="/tmp",
        policy=default_policy(),
        protocol=PolicySearchSession(experiment=state),
        flagging_warning_cap=cap,
    )


@pytest.mark.parametrize("source", ["keyword", "result", "trusted_server"])
async def test_warning_is_after_all_pending_results_and_is_not_duplicated(source):
    ctx = context()
    outcome = _TurnOutcome()
    command = "python -c 'import minigrid'" if source == "keyword" else "echo ok"
    await _dispatch_event(ToolCall("a", "Bash", {"command": command}), ctx, outcome)
    await _dispatch_event(ToolCall("b", "Bash", {"command": "echo unrelated"}), ctx, outcome)
    assert not ctx.agent.injected
    if source == "trusted_server":
        # Same shared counter the CWM HTTP denial updates, even on caught shell errors.
        ctx.experiment.flagged_tool_calls += 1
    output = "curl: (6) Could not resolve host: example.com" if source == "result" else "done"
    await _dispatch_event(ToolResult("a", output, source == "result"), ctx, outcome)
    assert not ctx.agent.injected
    await _dispatch_event(ToolResult("b", "done"), ctx, outcome)
    assert len(ctx.agent.injected) == 1
    assert ctx.agent.injected[0].startswith("WARNING: a command was flagged")
    warning = ctx.agent.injected[0]
    if source == "trusted_server":
        assert "exact attribution is unavailable" in warning
        assert "echo ok" in warning and "echo unrelated" in warning
        assert "Command flagged : 'echo unrelated'" not in warning
    else:
        assert f"Command flagged : '{command}'" in warning
        assert "echo unrelated" not in warning
    assert not ctx.warning_calls  # no retention of old clean commands
    assert isinstance(ctx.transcript.write.call_args.args[0], UserMessage)
    await _dispatch_event(ToolResult("b", "duplicate"), ctx, outcome)
    assert len(ctx.agent.injected) == 1


async def test_disabled_warning_still_counts_server_flags():
    ctx = context(cap=0)
    ctx.experiment.flagged_tool_calls = 1
    await _dispatch_event(ToolResult("a", "caught denial", False), ctx, _TurnOutcome())
    assert not ctx.agent.injected and ctx.experiment.flagged_tool_calls == 1


async def test_cli_warning_is_recorded_in_actual_next_send(monkeypatch):
    import regact.orchestration.loop as loop

    monkeypatch.setattr(loop, "_save_state", lambda ctx: None)
    agent = ScriptedAgent([[]])
    monkeypatch.setattr(
        agent,
        "capabilities",
        lambda: replace(ScriptedAgent().capabilities(), supports_inject=False),
    )
    ctx = context(agent)
    ctx.experiment.flagged_tool_calls = 1
    await _dispatch_event(ToolResult("a", "caught"), ctx, _TurnOutcome())
    assert not agent.injected and ctx.pending_warnings
    ctx.transcript.write.assert_not_called()
    await _run_turn("Continue", ctx)
    assert agent.sent[0].startswith("WARNING:") and agent.sent[0].endswith("Continue")
    assert ctx.transcript.write.call_args_list[0].args[0].text == agent.sent[0]
    assert not ctx.pending_warnings


async def test_alan_runner_waits_before_next_iteration(monkeypatch):
    from regact.agent import alan_runner
    from regact.agent.events import TextDelta

    frames = []
    commands = iter([{"cmd": "inject", "message": "WARNING: example"}, {"cmd": "continue"}])

    async def read():
        return next(commands)

    monkeypatch.setattr(alan_runner, "_read_command", read)
    monkeypatch.setattr(alan_runner, "_write", frames.append)
    monkeypatch.setattr("regact.agent.alan_adapter.map_alan_events", lambda event: [event])

    class Agent:
        messages = []

        def inject_message(self, message):
            self.messages.append(message)

        async def query_events_async(self, message):
            yield ToolResult("tool", "result")
            assert self.messages == ["WARNING: example"]
            yield TextDelta("continued after warning")

    await alan_runner._run_turn(Agent(), "go", synchronize_tools=True)
    assert frames[0]["_await_continue"] is True
    assert frames[1]["text"] == "continued after warning"


async def test_alan_runner_close_while_paused_shuts_agent_down(monkeypatch):
    """A close received at a paused tool result must end the command loop and run agent.close(),
    not leave the child waiting until the parent SIGKILLs it (transcript never finalized)."""
    from regact.agent import alan_runner

    frames = []
    commands = iter([{"cmd": "start"}, {"cmd": "send", "message": "go"}, {"cmd": "close"}])

    async def read():
        return next(commands, None)

    class Agent:
        closed = False
        stream_closed = False

        async def query_events_async(self, message):
            try:
                yield ToolResult("tool", "result")
                yield ToolResult("tool", "never reached")
            finally:
                Agent.stream_closed = True

        async def close(self):
            Agent.closed = True

    monkeypatch.setattr(alan_runner, "_read_command", read)
    monkeypatch.setattr(alan_runner, "_write", frames.append)
    monkeypatch.setattr(alan_runner, "_build", lambda command: Agent())
    monkeypatch.setattr(alan_runner, "_assembled_prompt", lambda agent: "")
    monkeypatch.setattr("regact.agent.alan_adapter.map_alan_events", lambda event: [event])

    assert await asyncio.wait_for(alan_runner._serve(), timeout=2) == 0
    assert Agent.closed and Agent.stream_closed
    assert frames[-1]["type"] == alan_runner.TURN_END


async def test_alan_parent_injects_before_acknowledgement(tmp_path):
    from regact.agent.alan_subprocess import AlanSubprocessAgent
    from test_alan_subprocess import _start_scripted_child

    script = """import sys, json
sys.stdin.readline()
print(json.dumps({"type":"_ready"}), flush=True)
sys.stdin.readline()
print(json.dumps({"type":"ToolResult","id":"a","output":"ok","is_error":False,"_await_continue":True}), flush=True)
first=json.loads(sys.stdin.readline()); second=json.loads(sys.stdin.readline())
assert first == {"cmd":"inject","message":"warning"}, first
assert second == {"cmd":"continue"}, second
print(json.dumps({"type":"TextDelta","text":"confirmed"}), flush=True)
print(json.dumps({"type":"_turn_end"}), flush=True)
sys.stdin.readline()
"""
    agent = AlanSubprocessAgent()
    await _start_scripted_child(agent, script, str(tmp_path))
    try:
        stream = agent.send("go")
        first = await asyncio.wait_for(anext(stream), timeout=5)
        assert isinstance(first, ToolResult)
        await agent.inject("warning")
        async with asyncio.timeout(5):
            remaining = [event async for event in stream]
        assert remaining[0].text == "confirmed"
    finally:
        await agent.close()


async def test_terminal_tool_boundary_counts_flags_without_claiming_delivery():
    ctx = context()
    ctx.max_tool_calls = 1
    outcome = _TurnOutcome()
    await _dispatch_event(
        ToolCall("a", "Bash", {"command": "python -c 'import minigrid'"}), ctx, outcome
    )
    await _dispatch_event(ToolResult("a", "done"), ctx, outcome)
    assert ctx.experiment.flagged_tool_calls == 1
    assert not ctx.agent.injected and not ctx.pending_warnings
    ctx.transcript.write.assert_not_called()


@pytest.mark.parametrize(
    "args,expected",
    [
        ({"command": "curl example.com"}, "curl example.com"),
        ({"cmd": "curl example.com"}, "curl example.com"),
        ({"url": "https://example.com"}, 'Fetch {"url": "https://example.com"}'),
    ],
)
async def test_denied_result_quotes_matching_original_tool(args, expected):
    ctx = context()
    outcome = _TurnOutcome()
    await _dispatch_event(ToolCall("a", "Fetch", args), ctx, outcome)
    await _dispatch_event(ToolResult("a", "403 Forbidden", True), ctx, outcome)
    assert f"Command flagged : '{expected}'" in ctx.agent.injected[0]


async def test_serial_trusted_flag_quotes_command_even_when_denial_caught():
    ctx = context()
    outcome = _TurnOutcome()
    await _dispatch_event(ToolCall("a", "Bash", {"command": "python explore.py"}), ctx, outcome)
    ctx.experiment.flagged_tool_calls += 1
    await _dispatch_event(ToolResult("a", "caught", False), ctx, outcome)
    assert "Command flagged : 'python explore.py'" in ctx.agent.injected[0]


async def test_delayed_warning_retains_flagged_command_after_later_clean_calls(monkeypatch):
    import regact.orchestration.loop as loop

    monkeypatch.setattr(loop, "_save_state", lambda ctx: None)
    agent = ScriptedAgent([[]])
    monkeypatch.setattr(
        agent,
        "capabilities",
        lambda: replace(
            ScriptedAgent().capabilities(),
            supports_inject=False,
        ),
    )
    ctx = context(agent)
    outcome = _TurnOutcome()
    command = "python -c 'import minigrid' # START" + "x" * 2000 + "END"
    await _dispatch_event(ToolCall("a", "Bash", {"command": command}), ctx, outcome)
    await _dispatch_event(ToolResult("a", "done"), ctx, outcome)
    await _dispatch_event(ToolCall("b", "Bash", {"command": "echo unrelated"}), ctx, outcome)
    await _dispatch_event(ToolResult("b", "ok"), ctx, outcome)
    await _run_turn("Continue", ctx)
    sent = agent.sent[0]
    assert "Command flagged : 'python -c" in sent
    assert "START" in sent and "END'" in sent and "[truncated]" in sent
    assert "echo unrelated" not in sent and "x" * 800 not in sent
    assert not ctx.pending_warnings and not ctx.warning_calls
