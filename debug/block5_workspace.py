"""Block 5 manual smoke: bootstrap a workdir and render a prompt.

Run:  python debug/block5_workspace.py
"""

import tempfile
from pathlib import Path

from regact.config.schema import Lifecycle
from regact.prompt.builder import PromptBuilder
from regact.workspace.bootstrap import Workspace


class _DemoProblem:
    name = "minigrid-empty"

    def build_prompt(self, task_name: str, *, info_mode: object) -> str:
        return f"Reach the goal tile in {task_name}. Actions: turn/forward."


def main() -> None:
    root = tempfile.mkdtemp(prefix="regact-wd-")
    ws = Workspace(root)
    ws.bootstrap(
        [],
        problem_name="minigrid-empty",
        task_name="MiniGrid-Empty-5x5",
        env_base_url="http://127.0.0.1:8000",
        game_id="MiniGrid-Empty-5x5",
        lifecycle=Lifecycle.MULTI_INSTANCE,
    )
    print("workdir laid out at (agnostic base — no solution.py yet):", root)
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            print("  ", path.relative_to(root))

    builder = PromptBuilder()
    print("\n--- first message ---")
    print(builder.build_first_message(_DemoProblem(), "MiniGrid-Empty-5x5", []))


if __name__ == "__main__":
    main()
