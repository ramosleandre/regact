"""Block 6 (controller feature) manual smoke.

Shows everything ControllerFeature produces, so you can eyeball what the agent
actually sees and edits:
  1. the prompt_fragment (the controller contract, in isolation),
  2. the 3 templates in full (base_controller / example / solution stub),
  3. the tools wired with a fake RunDeps,
  4. the full first message + bootstrapped workdir.

Run:  python debug/block6_controller_feature.py
"""

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from regact.config.schema import Lifecycle
from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.envclient.client import EnvClient
from regact.features.base import FeatureContext, RunDeps, build_features
from regact.features.controller import ControllerFeature
from regact.prompt.builder import PromptBuilder
from regact.session.state import ExperimentState
from regact.testing.fakes import FakeNativeEnv
from regact.workspace.bootstrap import Workspace


class _DemoProblem:
    name = "minigrid-empty"

    def build_prompt(self, task_name: str, *, info_mode: object) -> str:
        return f"Reach the goal tile in {task_name}."


def _env_client() -> EnvClient:
    server = EnvServer()
    server.register(
        "demo",
        EnvSession(
            make_native=lambda: FakeNativeEnv(goal=3),
            key="demo",
            renderer=RawRenderer(),
            lifecycle=MultiInstancePolicy(),
        ),
    )
    return EnvClient(TestClient(server.app), "demo")


def _rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    feature = ControllerFeature()
    ctx = FeatureContext("minigrid-empty", "MiniGrid-Empty-5x5", "/tmp/demo-wd")

    # 1. The prompt fragment, on its own.
    _rule("1. ControllerFeature.prompt_fragment(ctx)")
    print(feature.prompt_fragment(ctx))

    # 2. Every template the feature scaffolds, in full.
    _rule("2. ControllerFeature.templates(ctx)")
    for tmpl in feature.templates(ctx):
        print(f"\n----- {tmpl.relpath} -----")
        print(tmpl.content)

    # 3. The tools, wired with fake run dependencies.
    _rule("3. ControllerFeature.tools(deps)")
    deps = RunDeps(
        experiment=ExperimentState(
            problem_name="minigrid-empty",
            task_name="MiniGrid-Empty-5x5",
            n_eval_episodes=3,
            n_videos=0,
        ),
        env_client=_env_client(),
        lifecycle=Lifecycle.MULTI_INSTANCE,
        solution_path="/tmp/demo-wd/solution.py",
        submissions_dir="/tmp/demo-wd/submissions",
        n_episodes=3,
        max_moves=100,
    )
    for tool in feature.tools(deps):
        print(f"  - {tool.name}: {tool.description}")
    print("  hooks:", [(type(h).__name__, h.phase.value) for h in feature.hooks(deps)])

    # 4. The full first message + bootstrapped workdir.
    _rule("4. bootstrap + full first message")
    features = build_features(["controller"])
    root = tempfile.mkdtemp(prefix="regact-wd-")
    Workspace(root).bootstrap(
        features,
        problem_name="minigrid-empty",
        task_name="MiniGrid-Empty-5x5",
        env_base_url="http://127.0.0.1:8000",
        game_id="MiniGrid-Empty-5x5",
        lifecycle=Lifecycle.MULTI_INSTANCE,
    )
    print("workdir:", root)
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            print("  ", path.relative_to(root))
    print("\n--- first message ---")
    print(PromptBuilder().build_first_message(_DemoProblem(), "MiniGrid-Empty-5x5", features))


if __name__ == "__main__":
    main()
