"""One framework command of each name per agent tool call (protocol.one_command_per_tool_call)."""

import asyncio
import json
from unittest.mock import Mock

import pytest

from regact.agent.events import ToolCall, ToolResult
from regact.agent.scripted_agent import ScriptedAgent
from regact.orchestration.loop import _dispatch_event, _LoopContext, _TurnOutcome
from regact.protocols.managed.grants import CommandGrants
from regact.protocols.managed.session import ManagedSession, ManagedTool
from regact.security.policy import default_policy
from regact.session.state import ExperimentState
from regact.tools.base import ToolContext
from test_cwm_protocol import exploration
from test_managed_protocol import make_rig  # noqa: F401  (fixture)

_WAIT = 0.15


async def _claim(grants, name, request_id=None):
    return (await grants.claim(name, request_id))[0]


async def test_a_call_grants_one_command_of_each_name():
    grants = CommandGrants(_WAIT)
    grants.call_started("a")
    assert await _claim(grants, "RunController") == "granted"
    assert await _claim(grants, "ResetLevel") == "granted"
    assert await _claim(grants, "UpdateCodeWorldModel") == "granted"
    assert await _claim(grants, "RunController") == "refused"
    assert await _claim(grants, "ResetLevel") == "refused"
    grants.call_finished("a")
    grants.call_started("b")
    assert await _claim(grants, "RunController") == "granted"


async def test_commands_run_when_no_tool_call_was_ever_seen():
    grants = CommandGrants(_WAIT)
    for _ in range(3):
        assert await _claim(grants, "RunController") == "unattributed"


async def test_a_command_ahead_of_its_tool_call_start_waits_for_it():
    grants = CommandGrants(2.0)
    grants.call_started("a")
    assert await _claim(grants, "RunController") == "granted"
    grants.call_finished("a")
    pending = asyncio.create_task(grants.claim("RunController"))
    await asyncio.sleep(0.1)
    assert not pending.done()
    grants.call_started("b")
    outcome, waited = await pending
    assert outcome == "granted" and 0.1 <= waited < 2.0


async def test_a_background_command_uses_the_grant_of_the_call_that_just_returned():
    grants = CommandGrants(_WAIT)
    grants.call_started("a")
    grants.call_finished("a")
    assert await _claim(grants, "RunController") == "granted"
    assert await _claim(grants, "RunController") == "refused"  # a background loop
    grants.call_started("b")
    grants.call_finished("b")
    assert await _claim(grants, "RunController") == "granted"


async def test_parallel_calls_each_hold_a_grant():
    grants = CommandGrants(_WAIT)
    grants.call_started("a")
    grants.call_started("b")
    assert await _claim(grants, "RunController") == "granted"
    assert await _claim(grants, "RunController") == "granted"
    assert await _claim(grants, "RunController") == "refused"


async def test_a_retried_request_is_not_a_second_command():
    grants = CommandGrants(_WAIT)
    grants.call_started("a")
    assert await _claim(grants, "RunController", "r1") == "granted"
    assert await _claim(grants, "RunController", "r1") == "granted"
    assert await _claim(grants, "RunController", "r2") == "refused"


async def test_the_loop_reports_every_tool_call_to_the_protocol():
    state = ExperimentState(problem_name="p", task_name="t")
    grants = CommandGrants(_WAIT)
    session = ManagedSession(coordinator=Mock(grants=grants))
    ctx = _LoopContext(
        agent=ScriptedAgent(),
        experiment=state,
        tools_by_name={},
        transcript=Mock(),
        logger=Mock(),
        cwd="/tmp",
        policy=default_policy(),
        protocol=session,
    )
    outcome = _TurnOutcome()
    await _dispatch_event(ToolCall("a", "Bash", {"command": "echo"}), ctx, outcome)
    assert await _claim(grants, "RunController") == "granted"
    assert await _claim(grants, "RunController") == "refused"
    await _dispatch_event(ToolResult("a", "done"), ctx, outcome)
    await _dispatch_event(ToolCall("b", "Read", {"file_path": "x"}), ctx, outcome)
    assert await _claim(grants, "RunController") == "granted"


_CONTROL = {"channel": "control"}


@pytest.mark.integration
async def test_a_second_command_in_one_call_is_refused_and_not_executed(make_rig):  # noqa: F811
    c = make_rig()
    c.grants.wait_seconds = _WAIT
    exploration(c, (1,))
    run = ManagedTool("RunController", c)
    reset = ManagedTool("ResetEnvironment", c)
    context = ToolContext(cwd=str(c.workdir), detail=_CONTROL)
    c.grants.call_started("a")
    first = await run.call({}, context)
    assert not first.is_error and json.loads(first.data)["real_actions"] == 1
    assert not (await reset.call({}, context)).is_error  # another name: its own grant
    before = c.store.summary()
    second = await run.call({}, context)
    value = json.loads(second.data)
    assert second.is_error and value["status"] == "Refused"
    assert "one RunController" in value["message"] and "new tool call" in value["message"]
    assert c.store.summary() == before
    refused = c.store.db.execute("SELECT COUNT(*) FROM events WHERE kind='command_refused'")
    assert refused.fetchone()[0] == 1
    c.grants.call_finished("a")
    c.grants.call_started("b")
    assert not (await run.call({}, context)).is_error


@pytest.mark.integration
@pytest.mark.parametrize("case", ["loop_executed", "option_off"])
async def test_commands_outside_the_rule_are_never_refused(make_rig, case):  # noqa: F811
    c = make_rig(**({"one_command_per_tool_call": False} if case == "option_off" else {}))
    c.grants.wait_seconds = _WAIT
    exploration(c, (1,))
    detail = _CONTROL if case == "option_off" else {}
    c.grants.call_started("a")
    for _ in range(3):
        result = await ManagedTool("RunController", c).call(
            {}, ToolContext(cwd=str(c.workdir), detail=detail)
        )
        assert not result.is_error
