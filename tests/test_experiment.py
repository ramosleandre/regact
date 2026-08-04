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
from regact.obs.errors import RegactError
from regact.orchestration.experiment import _resolve_task_names, run_experiment
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


def test_problem_tasks_are_validated_against_catalogue() -> None:
    config = RunConfig(
        agent=AgentConfig(name=AgentName.SCRIPTED),
        problem=ProblemConfig(name="fake_exp", tasks=["missing"]),
    )
    with pytest.raises(RegactError, match="not exposed"):
        _resolve_task_names(config, ["g1", "g2"])

    config.problem.tasks = ["g1", "g1"]
    with pytest.raises(RegactError, match="duplicate"):
        _resolve_task_names(config, ["g1", "g2"])


async def test_run_experiment_runs_all_tasks(tmp_path: Path) -> None:
    config = RunConfig(
        agent=AgentConfig(name=AgentName.SCRIPTED),
        problem=ProblemConfig(name="fake_exp"),
        limits=LimitsConfig(max_turns=1),
    )
    reasons = await run_experiment(config, output_root=str(tmp_path))

    # Both games ran; the default scripted agent never submits/exits -> loop_limit.
    assert set(reasons) == {"g1", "g2"}
    assert all(r == "loop_limit" for r in reasons.values())
    # Per-task output dirs were created.
    assert (tmp_path / "g1" / "logs" / "experiment_state.json").exists()
    assert (tmp_path / "g2" / "logs" / "experiment_state.json").exists()


async def test_problem_tasks_selects_experiment_subset(tmp_path: Path) -> None:
    config = RunConfig(
        agent=AgentConfig(name=AgentName.SCRIPTED),
        problem=ProblemConfig(name="fake_exp", tasks=["g2"]),
        limits=LimitsConfig(max_turns=1),
    )
    reasons = await run_experiment(config, output_root=str(tmp_path))

    assert reasons == {"g2": "loop_limit"}
    assert not (tmp_path / "g1").exists()
    assert (tmp_path / "g2").exists()


def test_each_run_gets_its_own_timestamped_dir(tmp_path: Path) -> None:
    """Two runs of the same experiment name must not share a directory, or the second
    would overwrite the first's logs/scaffold and interleave its submissions."""
    from regact.orchestration.experiment import resolve_run_dir

    config = RunConfig(
        agent=AgentConfig(name=AgentName.SCRIPTED),
        problem=ProblemConfig(name="fake_exp"),
        experiment_name="same_name",
        output_root=str(tmp_path),
    )
    first = resolve_run_dir(config)
    assert Path(first).parent == tmp_path / "same_name"  # runs group under the name
    assert Path(first).name != "same_name"  # ...in a timestamped subdir

    # An explicit output_root still wins verbatim (tests and run_kaggle rely on it).
    assert resolve_run_dir(config, output_root=str(tmp_path / "fixed")) == str(tmp_path / "fixed")
