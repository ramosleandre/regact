"""Costly smoke: does a REAL agent authenticate, start, and take a real tool call?

HARD-gated behind ``--run-costly`` (see conftest) - it calls a real CLI agent, so it spends
credits / needs a served model, and must NEVER run in CI or a plain ``pytest``. It is the ONLY
guard for the auth / real-CLI class that no free test can reach (e.g. the "OAuth token revoked"
regression). Deliberately a SMOKE, not a solve test: 1 easy task, hard caps (<=60s, <=5 turns),
asserting the run reached the agent + took a tool call with no agent-side API error.

Run one backend:  pytest tests/test_costly.py -k claude --run-costly
Run all:          pytest tests/test_costly.py --run-costly
"""

import json
import shutil
from pathlib import Path

import pytest

from regact.config.schema import (
    AgentConfig,
    AgentName,
    ControllerConfig,
    LimitsConfig,
    ProblemConfig,
    RunConfig,
)
from regact.orchestration.experiment import run_experiment
from regact.security.runtime import SandboxRuntime, detect

pytestmark = pytest.mark.costly

# Hard caps baked in so a costly smoke can NEVER balloon into a real benchmark.
_MAX_SECONDS = 60
_MAX_TURNS = 5

_CLAUDE_ARGS = {"effort": "low", "permission_mode": "bypassPermissions"}
_BACKENDS = [
    pytest.param(AgentName.CLAUDE, "sonnet", _CLAUDE_ARGS, id="claude"),
    pytest.param(AgentName.CODEX, "gpt-5.5", {"reasoning_effort": "low"}, id="codex"),
]


@pytest.mark.parametrize(("agent_name", "model", "args"), _BACKENDS)
async def test_real_agent_smoke(
    tmp_path: Path, agent_name: AgentName, model: str, args: dict[str, object]
) -> None:
    if shutil.which(agent_name.value) is None:
        pytest.skip(f"{agent_name.value} CLI not on PATH")
    if detect() is SandboxRuntime.NONE:
        pytest.skip("no sandbox backend on this host (the smoke tests the real sandboxed path)")

    config = RunConfig(
        agent=AgentConfig(name=agent_name, model=model, args=dict(args)),
        problem=ProblemConfig(
            name="minigrid", tasks=["MiniGrid-Empty-5x5-v0"], kwargs={"fully_obs": True}
        ),
        controller=ControllerConfig(n_episodes=1, max_moves=50, n_videos=0, shadow_replay=True),
        limits=LimitsConfig(max_turns=_MAX_TURNS, max_seconds_per_task=_MAX_SECONDS),
        sandbox=True,
    )
    # The caps are the whole safety story - assert them before spending anything.
    assert config.limits.max_seconds_per_task is not None
    assert config.limits.max_seconds_per_task <= _MAX_SECONDS
    assert config.limits.max_turns <= _MAX_TURNS

    reasons = await run_experiment(config, output_root=str(tmp_path))
    assert set(reasons) == {"MiniGrid-Empty-5x5-v0"}  # the task ran

    state = json.loads(
        (tmp_path / "MiniGrid-Empty-5x5-v0" / "logs" / "experiment_state.json").read_text()
    )
    # It authenticated and ran (no agent-side API/auth error - what M's "revoked" bug produced).
    assert state["last_error_category"] != "agent_api", (
        f"agent-side API/auth failure: {state.get('last_submission_results')}"
    )
    # It stopped for a normal reason, not a crash.
    assert state["exit_reason"] in {"agent_exit", "walltime_limit", "loop_limit"}

    # It actually took a real tool call (a spinning/failed agent would have none).
    transcript = (tmp_path / "MiniGrid-Empty-5x5-v0" / "logs" / "transcript.jsonl").read_text()
    tool_calls = sum(1 for line in transcript.splitlines() if '"type": "ToolCall"' in line)
    assert tool_calls >= 1, "the agent never took a tool call (did it authenticate / start?)"
