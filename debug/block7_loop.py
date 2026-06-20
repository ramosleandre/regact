"""Block 7 manual smoke: run a full scripted session end-to-end, no LLM.

A ScriptedAgent submits a forward-stepping controller, then exits. Prints the
exit reason and the artifacts the loop wrote to disk.

Run:  python debug/block7_loop.py
"""

import asyncio
import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from regact.agent.events import TextDelta, ToolCall, TurnComplete
from regact.agent.scripted_agent import ScriptedAgent
from regact.config.schema import Lifecycle, LimitsConfig
from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.envclient.client import EnvClient
from regact.features.base import RunDeps
from regact.features.controller import ControllerFeature
from regact.obs.logger import RunLogger
from regact.obs.transcript import TranscriptWriter
from regact.orchestration.loop import run_session
from regact.session.state import ExperimentState
from regact.testing.fakes import FakeNativeEnv

_FORWARD = """\
class Controller:
    def act(self, obs):
        return 1

def get_controller():
    return Controller()
"""


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="regact-run-"))
    workdir = root / "wd"
    workdir.mkdir()
    (workdir / "solution.py").write_text(_FORWARD)
    logs = root / "logs"
    logs.mkdir()

    server = EnvServer()
    server.register(
        "corridor",
        EnvSession(
            make_native=lambda: FakeNativeEnv(goal=3),
            key="corridor",
            renderer=RawRenderer(),
            lifecycle=MultiInstancePolicy(),
        ),
    )
    client = EnvClient(TestClient(server.app), "corridor")
    experiment = ExperimentState(
        problem_name="demo", task_name="corridor", n_eval_episodes=2, n_videos=0
    )
    deps = RunDeps(
        experiment=experiment,
        env_client=client,
        lifecycle=Lifecycle.MULTI_INSTANCE,
        solution_path=str(workdir / "solution.py"),
        submissions_dir=str(workdir / "submissions"),
        n_episodes=2,
        max_moves=10,
    )
    feature = ControllerFeature()
    tools = feature.tools(deps)
    hooks = feature.hooks(deps)

    agent = ScriptedAgent(
        [
            [
                TextDelta("I'll submit my controller."),
                ToolCall("c1", "SubmitSolution", {}),
                TurnComplete(),
            ],
            [TextDelta("Looks good, exiting."), ToolCall("c2", "ExitTask", {}), TurnComplete()],
        ]
    )

    with (
        TranscriptWriter(str(logs / "transcript.jsonl")) as transcript,
        RunLogger(str(logs), task="corridor") as logger,
    ):
        reason = await run_session(
            agent,
            first_message="Study the game and write a controller.",
            experiment=experiment,
            tools=tools,
            transcript=transcript,
            logger=logger,
            limits=LimitsConfig(keep_alive=10, max_moves=10),
            state_path=str(logs / "experiment_state.json"),
            cwd=str(workdir),
            hooks=hooks,
        )

    print("exit reason:", reason)
    print(
        "submission_count:",
        experiment.submission_count,
        "| exit_requested:",
        experiment.exit_requested,
    )
    print("\nartifacts:")
    for path in sorted(root.rglob("*")):
        if path.is_file():
            print("  ", path.relative_to(root))
    print("\ntranscript.jsonl:")
    for line in (logs / "transcript.jsonl").read_text().splitlines():
        print("  ", json.loads(line)["type"])
    results = json.loads((workdir / "submissions" / "0" / "results.json").read_text())
    print("\nresults.json aggregate:", results["aggregate"])


if __name__ == "__main__":
    asyncio.run(main())
