"""Tests for run_controller + ControllerExecutor: drive a controller over HTTP, score it.

No LLM: a trivial controller is written to a temp solution.py and evaluated against
FakeNativeEnv behind the in-process TestClient.
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from regact.config.schema import Lifecycle
from regact.controllers.executor import ControllerExecutor, _default_episode_metrics
from regact.controllers.runner import run_controller
from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.envclient.client import EnvClient
from regact.envclient.obs import Obs
from regact.obs.errors import ErrorCategory
from regact.testing.fakes import FakeNativeEnv

# A controller that always steps forward (action 1) reaches the goal in 3 moves.
_FORWARD_SOLUTION = """\
class Controller:
    def act(self, obs):
        return 1

def get_controller():
    return Controller()
"""

# A controller that raises inside act().
_BROKEN_SOLUTION = """\
class Controller:
    def act(self, obs):
        raise RuntimeError("boom")

def get_controller():
    return Controller()
"""

# A solution file with no get_controller (import-time contract violation).
_NO_FACTORY_SOLUTION = "x = 1\n"


def _client() -> EnvClient:
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
    return EnvClient(TestClient(server.app), "g")


def test_run_controller_reaches_goal() -> None:
    client = _client()
    client.reset()

    class Forward:
        def act(self, obs: object) -> int:
            return 1

    summary = run_controller(client, Forward(), max_steps=10)
    assert summary.stop_kind == "env_done"
    assert summary.total_steps == 3


def test_run_controller_hits_max_steps() -> None:
    client = _client()
    client.reset()

    class Stay:
        def act(self, obs: object) -> int:
            return 0  # never advances

    summary = run_controller(client, Stay(), max_steps=5)
    assert summary.stop_kind == "max_steps"
    assert summary.total_steps == 5


def _write_solution(tmp_path: Path, body: str) -> str:
    path = tmp_path / "solution.py"
    path.write_text(body)
    return str(path)


def test_executor_scores_a_solution(tmp_path: Path) -> None:
    executor = ControllerExecutor(_client())
    out = str(tmp_path / "results.json")
    result = executor.run(
        task_name="corridor",
        solution_path=_write_solution(tmp_path, _FORWARD_SOLUTION),
        output_path=out,
        lifecycle=Lifecycle.MULTI_INSTANCE,
        n_episodes=2,
        max_moves=10,
    )
    assert result.aggregate["n_episodes"] == 2
    assert result.aggregate["success_rate"] == 1.0
    assert result.aggregate["mean_steps"] == 3.0
    # results.json written and matches.
    assert json.loads(Path(out).read_text())["aggregate"]["success_rate"] == 1.0


def test_executor_imports_sibling_workdir_packages(tmp_path: Path) -> None:
    """solution.py can import its sibling ``code_library/`` — the workdir is on sys.path.

    Mirrors the real bootstrap layout (a namespace package, no ``__init__.py``); a
    regression against the eval failing with ``ModuleNotFoundError: 'code_library'``.
    """
    lib = tmp_path / "code_library"
    lib.mkdir()
    (lib / "base_controller.py").write_text(
        "class BaseController:\n    def act(self, obs):\n        raise NotImplementedError\n"
    )
    body = (
        "from code_library.base_controller import BaseController\n"
        "class Controller(BaseController):\n"
        "    def act(self, obs):\n        return 1\n"
        "def get_controller():\n    return Controller()\n"
    )
    executor = ControllerExecutor(_client())
    result = executor.run(
        task_name="corridor",
        solution_path=_write_solution(tmp_path, body),
        output_path=str(tmp_path / "results.json"),
        lifecycle=Lifecycle.MULTI_INSTANCE,
        n_episodes=1,
        max_moves=10,
    )
    assert result.error is None
    assert result.aggregate["success_rate"] == 1.0


def test_executor_single_instance_runs_one_episode(tmp_path: Path) -> None:
    executor = ControllerExecutor(_client())
    result = executor.run(
        task_name="corridor",
        solution_path=_write_solution(tmp_path, _FORWARD_SOLUTION),
        output_path=str(tmp_path / "results.json"),
        lifecycle=Lifecycle.SINGLE_INSTANCE,
        n_episodes=5,  # ignored under single-instance
        max_moves=10,
    )
    assert result.aggregate["n_episodes"] == 1


def test_executor_catches_controller_exception(tmp_path: Path) -> None:
    executor = ControllerExecutor(_client())
    result = executor.run(
        task_name="corridor",
        solution_path=_write_solution(tmp_path, _BROKEN_SOLUTION),
        output_path=str(tmp_path / "results.json"),
        lifecycle=Lifecycle.MULTI_INSTANCE,
        n_episodes=1,
        max_moves=10,
    )
    assert result.aggregate["n_errors"] == 1
    assert result.episodes[0].error_category is ErrorCategory.AGENT_SOLUTION


def test_default_success_requires_positive_reward() -> None:
    """Done alone is NOT success (lava death / game over); a positive terminal reward is."""

    def success(reward: float, done: bool) -> bool:
        obs = Obs(frame=None, reward=reward, is_done=done)
        return bool(_default_episode_metrics(obs, steps=1)["success"])

    assert success(1.0, True)  # reached the goal
    assert not success(0.0, True)  # terminated without reward (lava / game over)
    assert not success(1.0, False)  # rewarded but not terminated


def test_executor_uses_injected_problem_metrics(tmp_path: Path) -> None:
    """The problem's metric functions decide the score, not the generic default."""

    def compute(final_obs: Obs, *, steps: int) -> dict[str, object]:
        return {"levels_completed": 7, "steps": steps}

    def aggregate(episodes: list[dict[str, object]]) -> dict[str, object]:
        n = len(episodes)
        return {"n_episodes": n, "mean_levels": sum(e["levels_completed"] for e in episodes) / n}

    executor = ControllerExecutor(_client(), compute_metrics=compute, aggregate_metrics=aggregate)
    result = executor.run(
        task_name="g",
        solution_path=_write_solution(tmp_path, _FORWARD_SOLUTION),
        output_path=str(tmp_path / "results.json"),
        lifecycle=Lifecycle.MULTI_INSTANCE,
        n_episodes=2,
        max_moves=10,
    )
    assert result.aggregate["mean_levels"] == 7.0
    assert result.aggregate["n_errors"] == 0
    assert result.episodes[0].metrics == {"levels_completed": 7, "steps": 3}


def test_executor_flags_missing_factory(tmp_path: Path) -> None:
    executor = ControllerExecutor(_client())
    result = executor.run(
        task_name="corridor",
        solution_path=_write_solution(tmp_path, _NO_FACTORY_SOLUTION),
        output_path=str(tmp_path / "results.json"),
        lifecycle=Lifecycle.MULTI_INSTANCE,
        n_episodes=1,
        max_moves=10,
    )
    assert result.error_category is ErrorCategory.AGENT_SOLUTION
    assert result.episodes == []


def test_shadow_replay_reproduces_solve_with_recorded_seed(tmp_path: Path) -> None:
    """Regression: the shadow replay must reset to the SAME seed the rollout recorded, so a
    controller that genuinely solves a seed-dependent (procedural) env is scored the same by
    the replay. Before the fix, the replay reset to a fresh random layout and scored real
    solves as 0 (this is exactly what zeroed working MiniGrid controllers).
    """
    from regact.controllers.executor import (
        replay_and_score,
        run_episodes_raw,
        score_episodes,
    )

    class SeedKeyedEnv:
        """The winning action IS the seed; a reset to a different seed makes it unwinnable."""

        def __init__(self) -> None:
            self._seed: int | None = None
            self._done = False

        def _obs(self) -> Obs:
            return Obs(
                frame=None,
                reward=1.0 if self._done else 0.0,
                is_done=self._done,
                available_actions=[self._seed],
                info={"win_action": self._seed},
            )

        def reset(self, *, seed: int | None = None) -> Obs:
            self._seed, self._done = seed, False
            return self._obs()

        def current(self) -> Obs:
            return self._obs()

        def step(self, action: object) -> Obs:
            self._done = action == self._seed
            return self._obs()

    solution = tmp_path / "solution.py"
    solution.write_text(
        "class Controller:\n"
        "    def act(self, obs):\n"
        "        return obs.info['win_action']\n"
        "def get_controller():\n"
        "    return Controller()\n"
    )

    def metrics(final_obs: Obs, *, steps: int) -> dict:
        reward = final_obs.reward or 0.0
        return {"success": bool(final_obs.is_done and reward > 0), "reward": reward, "steps": steps}

    def aggregate(eps: list[dict]) -> dict:
        n = len(eps) or 1
        return {
            "n_episodes": len(eps),
            "success_rate": sum(bool(e["success"]) for e in eps) / n,
            "mean_reward": sum(e["reward"] for e in eps) / n,
        }

    raw = run_episodes_raw(
        SeedKeyedEnv(),  # type: ignore[arg-type]
        str(solution),
        lifecycle=Lifecycle.MULTI_INSTANCE,
        n_episodes=3,
        max_moves=5,
        seed=None,
    )
    assert [e["seed"] for e in raw] == [0, 1, 2]  # concrete seeds recorded (seed=None -> base 0)

    direct = score_episodes(
        raw, task_name="t", compute_metrics=metrics, aggregate_metrics=aggregate, executor="x"
    )
    assert direct.aggregate["success_rate"] == 1.0  # the controller really solves each episode

    shadow = replay_and_score(
        SeedKeyedEnv(),  # type: ignore[arg-type]
        raw,
        task_name="t",
        seed=None,
        compute_metrics=metrics,
        aggregate_metrics=aggregate,
    )
    assert shadow.aggregate["success_rate"] == 1.0  # THE FIX: replay reproduces the solve

    # Negative control: replaying against a MISMATCHED seed misses -> proves the seed is used.
    mismatched = [{**e, "seed": e["seed"] + 100} for e in raw]
    bad = replay_and_score(
        SeedKeyedEnv(),  # type: ignore[arg-type]
        mismatched,
        task_name="t",
        seed=None,
        compute_metrics=metrics,
        aggregate_metrics=aggregate,
    )
    assert bad.aggregate["success_rate"] == 0.0
