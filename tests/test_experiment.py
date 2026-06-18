"""Integration: run_experiment iterates a problem's tasks via the Scheduler.

No LLM, no real game: a registered fake problem (FakeNativeEnv) + the scripted
backend, driven through run_experiment end-to-end.
"""

from pathlib import Path
from typing import Any

import pytest

from regact.config.schema import AgentConfig, AgentName, LimitsConfig, ProblemConfig, RunConfig
from regact.env.renderer import RawRenderer
from regact.envclient.obs import Obs
from regact.orchestration.experiment import run_experiment
from regact.problems.base import BaseProblem, register_problem
from regact.testing.fakes import FakeNativeEnv

pytestmark = pytest.mark.integration


class _TwoGameProblem(BaseProblem):
    name = "fake_exp"

    def make_env(self, task_name: str) -> Any:
        return FakeNativeEnv(goal=3)

    def get_task_names(self) -> list[str]:
        return ["g1", "g2"]

    def obs_renderer(self, task_name: str, *, mode: Any) -> RawRenderer:
        return RawRenderer()

    def compute_episode_metrics(self, final_obs: Obs, *, steps: int) -> dict[str, Any]:
        return {"success": final_obs.is_done, "steps": steps}

    def aggregate_episode_metrics(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        return {"n_episodes": len(episodes)}

    def build_prompt(self, task_name: str, *, info_mode: Any) -> str:
        return f"# fake ({task_name})"

    def config_kwargs(self) -> dict[str, Any]:
        return {}


register_problem("fake_exp", lambda kwargs: _TwoGameProblem())


async def test_run_experiment_runs_all_tasks(tmp_path: Path) -> None:
    config = RunConfig(
        agent=AgentConfig(name=AgentName.SCRIPTED),
        problem=ProblemConfig(name="fake_exp"),
        limits=LimitsConfig(keep_alive=1, max_moves=10),
    )
    reasons = await run_experiment(config, output_root=str(tmp_path))

    # Both games ran; the default scripted agent never submits/exits -> loop_limit.
    assert set(reasons) == {"g1", "g2"}
    assert all(r == "loop_limit" for r in reasons.values())
    # Per-task output dirs were created.
    assert (tmp_path / "g1" / "logs" / "experiment_state.json").exists()
    assert (tmp_path / "g2" / "logs" / "experiment_state.json").exists()
