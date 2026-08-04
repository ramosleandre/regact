"""Unit tests: the agent launch diagnostic.

No CLI is required — these cover the parts regact owns: which backends are swept,
how a result renders, and that an in-process backend is correctly reported as having
nothing to launch.
"""

from __future__ import annotations

from pathlib import Path

from regact.agentcheck import FAIL, MISSING, OK, LaunchResult, check_agent, format_report
from regact.config.schema import AgentName


def test_in_process_backends_have_nothing_to_launch(tmp_path: Path) -> None:
    """scripted runs inside the orchestrator, so there is no argv to wrap — the
    diagnostic must skip it rather than invent a check."""
    assert check_agent(AgentName.SCRIPTED, workdir=str(tmp_path)) == []


def test_subprocess_backends_declare_a_launch_probe() -> None:
    """Every out-of-process backend must say how to prove it can start, or the
    diagnostic silently covers nothing for it."""
    from regact.agent.base import build_agent
    from regact.config.schema import AgentConfig

    for name in (AgentName.CLAUDE, AgentName.CODEX, AgentName.ALAN):
        argv = build_agent(AgentConfig(name=name)).launch_probe_argv()
        assert argv, f"{name.value} declares no launch probe"


def test_report_marks_failures_and_hides_argv_unless_verbose() -> None:
    results = [
        LaunchResult(
            "claude", "sandbox", FAIL, "not found once wrapped", ["claude", "--version"], "boom"
        ),
        LaunchResult("codex", "bare", OK, "codex-cli 1.0", ["codex", "--version"]),
    ]
    plain = format_report(results, verbose=False)
    assert "FAIL" in plain and "not found once wrapped" in plain
    assert "1 FAILURE(S)" in plain
    assert "--version" not in plain  # argv is verbose-only

    loud = format_report(results, verbose=True)
    assert "argv: claude --version" in loud
    assert "stderr: boom" in loud


def test_report_is_green_when_everything_launches() -> None:
    results = [LaunchResult("codex", "sandbox", OK, "ok", ["codex"])]
    assert "ALL LAUNCHED" in format_report(results, verbose=False)


def test_absent_backend_is_not_a_failure() -> None:
    """A CLI that is simply not installed says nothing about the sandbox, so it must not
    turn the diagnostic red on a host that only runs the other agents."""
    results = [LaunchResult("claude", "bare", MISSING, "'claude' is not installed here", [])]
    out = format_report(results, verbose=False)
    assert "ALL LAUNCHED" in out  # no FAIL rows
    assert "not installed here (not a failure): claude" in out
