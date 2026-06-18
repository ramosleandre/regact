"""Unit tests for the anti-cheat layer: AST scan, path confinement, tool guard,
and the Claude deny-list. Submit-time blocking is covered against the executor.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from regact.config.schema import Lifecycle
from regact.controllers.executor import ControllerExecutor
from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.envclient.client import EnvClient
from regact.isolation.harness import AntiCheatHarness, claude_deny_settings
from regact.isolation.paths import path_within
from regact.isolation.policy import default_policy
from regact.isolation.scan import scan_source
from regact.isolation.tool_guard import guard_tool_call
from regact.testing.fakes import FakeNativeEnv


def test_scan_flags_forbidden_imports_and_calls() -> None:
    policy = default_policy()
    assert scan_source("import arc_agi", policy)
    assert scan_source("from arcengine import GameAction", policy)
    assert scan_source("import inspect", policy)
    assert scan_source("importlib.import_module('arc_agi')", policy)
    assert scan_source("y = __import__('os')", policy)  # a forbidden call


def test_scan_passes_a_clean_controller() -> None:
    clean = "class Controller:\n    def act(self, obs):\n        return obs.available_actions[0]\n"
    assert scan_source(clean, default_policy()) == []


def test_path_within_confines_to_workdir(tmp_path: Path) -> None:
    root = str(tmp_path / "wd")
    assert path_within(str(tmp_path / "wd" / "solution.py"), root)
    assert not path_within(str(tmp_path / "environnement" / "ls20.py"), root)
    assert not path_within("../secret.py", root)


def test_tool_guard_flags_forbidden_paths_and_imports() -> None:
    policy = default_policy()
    assert guard_tool_call("Bash", {"command": "cat ../environnement/ls20/x.py"}, policy)
    assert guard_tool_call("Bash", {"command": "python -c 'import arc_agi'"}, policy)
    assert guard_tool_call("Bash", {"command": "ls code_library"}, policy) == []


def test_harness_scan_file(tmp_path: Path) -> None:
    sol = tmp_path / "solution.py"
    sol.write_text("import arc_agi\n")
    assert AntiCheatHarness().scan_file(str(sol))


def test_claude_deny_settings_confines_reads() -> None:
    settings = claude_deny_settings("/tmp/wd")
    deny = settings["permissions"]["deny"]
    assert any("environnement" in rule for rule in deny)
    assert "Read(/**)" in deny


def test_executor_blocks_a_cheating_solution(tmp_path: Path) -> None:
    """A solution importing the game lib is rejected before it ever runs."""
    server = EnvServer()
    server.register(
        "g",
        EnvSession(
            make_native=lambda: FakeNativeEnv(goal=3),
            key="g",
            renderer=RawRenderer(),
            lifecycle=MultiInstancePolicy(),
        ),
    )
    client = EnvClient(TestClient(server.app), "g")
    sol = tmp_path / "solution.py"
    sol.write_text("import arc_agi\n\ndef get_controller():\n    return None\n")
    out = tmp_path / "results.json"

    result = ControllerExecutor(client).run(
        task_name="g",
        solution_path=str(sol),
        output_path=str(out),
        lifecycle=Lifecycle.MULTI_INSTANCE,
        n_episodes=1,
        max_moves=10,
    )
    assert result.error is not None and "anti-cheat" in result.error
    assert result.episodes == []  # never executed the cheating module
    assert "anti-cheat" in json.loads(out.read_text())["error"]
