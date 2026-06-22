"""Unit tests for the anti-cheat layer: path confinement, the tool-call camera,
and the Claude deny-list. Confinement is enforced by the OS sandbox; these cover
the cheap, non-blocking detection signals that ride alongside it.
"""

import json
from pathlib import Path

from regact.agent.claude_adapter import claude_deny_settings
from regact.security.detection import flag_os_denial, flag_tool_call
from regact.security.paths import path_within
from regact.security.policy import default_policy


def test_path_within_confines_to_workdir(tmp_path: Path) -> None:
    root = str(tmp_path / "wd")
    assert path_within(str(tmp_path / "wd" / "solution.py"), root)
    assert not path_within(str(tmp_path / "environnement" / "ls20.py"), root)
    assert not path_within("../secret.py", root)


def test_camera_flags_forbidden_paths_and_imports() -> None:
    """Flags (for logging) the obvious attempts; it never blocks."""
    policy = default_policy()
    assert flag_tool_call("Bash", {"command": "cat ../environnement/ls20/x.py"}, policy)
    assert flag_tool_call("Bash", {"command": "python -c 'import arc_agi'"}, policy)
    assert flag_tool_call("Bash", {"command": "ls code_library"}, policy) == []


def test_os_denial_recognizes_blocked_egress_only() -> None:
    """Egress denials are flagged; benign sandbox friction (ps/kill/tmp) is NOT."""
    assert flag_os_denial("curl: (6) Could not resolve host: arcprize.org")
    assert flag_os_denial("HTTP/1.1 403 Forbidden")  # egress proxy refused the host
    # file/process friction reports "operation not permitted" too — not a cheat signal
    assert not flag_os_denial("ps: operation not permitted")
    assert not flag_os_denial("levels_completed=2\nstate=WIN")


def test_egress_camera_counts_blocked_curls_not_friction() -> None:
    """A blocked external fetch is counted; an errored, non-egress result is not."""
    import tempfile

    from regact.agent.events import ToolResult
    from regact.obs.logger import RunLogger
    from regact.orchestration.loop import _flag_blocked_result, _LoopContext
    from regact.session.state import ExperimentState

    with tempfile.TemporaryDirectory() as logs, RunLogger(logs, task="t") as logger:
        exp = ExperimentState(problem_name="p", task_name="t", n_eval_episodes=1, n_videos=0)
        ctx = _LoopContext(
            agent=None,  # type: ignore[arg-type]
            experiment=exp,
            tools_by_name={},
            transcript=None,  # type: ignore[arg-type]
            logger=logger,
            cwd="",
            policy=default_policy(),
        )
        curl = ToolResult("1", "curl: (6) Could not resolve host", is_error=True)
        _flag_blocked_result(curl, ctx)
        assert exp.cheat_attempts == 1
        # sandbox friction (ps/kill) and ordinary errors are not egress → not counted
        _flag_blocked_result(ToolResult("2", "ps: operation not permitted", is_error=True), ctx)
        _flag_blocked_result(ToolResult("3", "Traceback: KeyError", is_error=True), ctx)
        assert exp.cheat_attempts == 1


def test_claude_deny_settings_blocks_game_data_not_the_workdir() -> None:
    deny = claude_deny_settings("/tmp/wd")["permissions"]["deny"]
    # The game data is denied wherever it lives...
    assert "Read(**/environnement/**)" in deny
    # ...but reads are NOT blanket-denied (that would cripple the agent's own workdir).
    assert "Read(/**)" not in deny
    assert all("/**)" in rule and rule != "Read(/**)" for rule in deny)


def test_loop_flags_and_counts_cheat_attempts(tmp_path: Path) -> None:
    """A tool call reaching for the game data is counted + logged, never blocked."""
    from regact.agent.events import ToolCall
    from regact.obs.logger import RunLogger
    from regact.orchestration.loop import _flag_suspicious_call, _LoopContext
    from regact.session.state import ExperimentState

    logs = tmp_path / "logs"
    logs.mkdir()
    exp = ExperimentState(problem_name="p", task_name="t", n_eval_episodes=1, n_videos=0)
    with RunLogger(str(logs), task="t") as logger:
        ctx = _LoopContext(
            agent=None,  # type: ignore[arg-type]
            experiment=exp,
            tools_by_name={},
            transcript=None,  # type: ignore[arg-type]
            logger=logger,
            cwd="",
            policy=default_policy(),
        )
        _flag_suspicious_call(ToolCall("1", "Bash", {"command": "ls code_library"}), ctx)
        assert exp.cheat_attempts == 0  # a benign call is not flagged
        _flag_suspicious_call(ToolCall("2", "Bash", {"command": "cat ../environnement/x.py"}), ctx)
        assert exp.cheat_attempts >= 1  # reaching for the game data is counted

    events = [
        json.loads(line)
        for line in (logs / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert any(e["event"] == "cheat_attempt" for e in events)
