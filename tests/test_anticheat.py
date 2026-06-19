"""Unit tests for the anti-cheat layer: AST scan, path confinement, tool guard,
and the Claude deny-list. Submit-time blocking is covered against the executor.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from regact.agent.claude_adapter import claude_deny_settings
from regact.config.schema import Lifecycle
from regact.controllers.executor import ControllerExecutor
from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.envclient.client import EnvClient
from regact.security.detection import flag_tool_call
from regact.security.paths import path_within
from regact.security.policy import default_policy
from regact.security.scan import scan_file, scan_source
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


def test_camera_flags_forbidden_paths_and_imports() -> None:
    """Flags (for logging) the obvious attempts; it never blocks."""
    policy = default_policy()
    assert flag_tool_call("Bash", {"command": "cat ../environnement/ls20/x.py"}, policy)
    assert flag_tool_call("Bash", {"command": "python -c 'import arc_agi'"}, policy)
    assert flag_tool_call("Bash", {"command": "ls code_library"}, policy) == []


def test_scan_file_flags_cheating_solution(tmp_path: Path) -> None:
    sol = tmp_path / "solution.py"
    sol.write_text("import arc_agi\n")
    assert scan_file(str(sol), default_policy())


def test_claude_deny_settings_blocks_game_data_not_the_workdir() -> None:
    deny = claude_deny_settings("/tmp/wd")["permissions"]["deny"]
    # The game data is denied wherever it lives...
    assert "Read(**/environnement/**)" in deny
    # ...but reads are NOT blanket-denied (that would cripple the agent's own workdir).
    assert "Read(/**)" not in deny
    assert all("/**)" in rule and rule != "Read(/**)" for rule in deny)


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
