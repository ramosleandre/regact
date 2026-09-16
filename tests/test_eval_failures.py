"""Regression coverage for zero-credit controller errors and incomplete evaluations."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from regact.config.schema import Lifecycle
from regact.controllers.executor import (
    ControllerExecutor,
    SandboxedExecutor,
    replay_and_score,
    run_episodes_raw,
    score_episodes,
)
from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.envclient.client import EnvClient
from regact.envclient.errors import InvalidActionError
from regact.envclient.obs import Obs
from regact.obs.errors import ErrorCategory
from regact.orchestration.env_transport import serve_env
from regact.orchestration.loop import _solved
from regact.problems.arc_agi.problem import ArcAgiProblem, _ArcGymShim
from regact.problems.minigrid.problem import MiniGridProblem, _ActionInfoShim
from regact.testing.fakes import FakeNativeEnv


class EpisodeEnv:
    def reset(self, *, seed=None):
        self.seed = seed
        self.done = False
        return self.current()

    def current(self):
        return Obs(
            frame=None,
            reward=float(self.done),
            is_done=self.done,
            info={
                "seed": self.seed,
                "state": "WIN" if self.done else "PLAY",
                "levels_completed": 3 if self.done else 2,
                "win_levels": 3,
            },
        )

    def step(self, action):
        self.done = True
        return self.current()


def solution(tmp_path, body):
    path = tmp_path / "solution.py"
    path.write_text(body)
    return str(path)


@pytest.mark.parametrize("problem", [MiniGridProblem(), ArcAgiProblem()])
@pytest.mark.parametrize("replay", [False, True])
def test_one_success_nine_controller_errors_count_as_ten(tmp_path, problem, replay):
    path = solution(
        tmp_path,
        """class C:
 def act(self, obs):
  if obs.info['seed']: raise ValueError('bad controller')
  return 1
def get_controller(): return C()
""",
    )
    env = EpisodeEnv()
    raw = run_episodes_raw(
        env, path, lifecycle=Lifecycle.MULTI_INSTANCE, n_episodes=10, max_moves=5
    )
    kwargs = {
        "task_name": "g",
        "compute_metrics": problem.compute_episode_metrics,
        "aggregate_metrics": problem.aggregate_episode_metrics,
        "failure_metrics": problem.failure_metrics,
        "expected_episodes": 10,
    }
    result = (
        replay_and_score(env, raw, seed=0, **kwargs)
        if replay
        else score_episodes(raw, executor="test", **kwargs)
    )
    assert result.aggregate["n_episodes"] == 10
    assert result.aggregate["n_errors"] == 9
    key = "win_rate" if isinstance(problem, ArcAgiProblem) else "success_rate"
    assert result.aggregate[key] == 0.1
    if isinstance(problem, ArcAgiProblem):
        assert result.aggregate["mean_levels_completion_rate"] == 0.1
        assert result.episodes[1].metrics["levels_completed"] == 0
    assert not _solved(
        SimpleNamespace(last_submission_results=result.to_json()), problem.is_perfect
    )


@pytest.mark.parametrize("fault", ["reset", "step"])
def test_environment_fault_is_incomplete_not_controller_zero(tmp_path, fault):
    class Flaky(EpisodeEnv):
        def reset(self, *, seed=None):
            if fault == "reset" and seed == 1:
                raise TimeoutError("server unavailable")
            return super().reset(seed=seed)

        def step(self, action):
            if fault == "step" and self.seed == 1:
                raise ValueError("internal engine bug")
            return super().step(action)

    problem = MiniGridProblem()
    executor = ControllerExecutor(
        Flaky(),
        compute_metrics=problem.compute_episode_metrics,
        aggregate_metrics=problem.aggregate_episode_metrics,
        failure_metrics=problem.failure_metrics,
    )
    result = executor.run(
        task_name="g",
        solution_path=solution(
            tmp_path, "class C:\n def act(self, obs): return 1\ndef get_controller(): return C()\n"
        ),
        output_path=str(tmp_path / "result.json"),
        lifecycle=Lifecycle.MULTI_INSTANCE,
        n_episodes=2,
    )
    assert result.episodes[1].error_category == ErrorCategory.ENV_RUNTIME
    assert result.episodes[1].metrics == {}
    assert result.aggregate["n_episodes"] == 1
    assert result.aggregate["evaluation_complete"] is False
    assert not _solved(
        SimpleNamespace(last_submission_results=result.to_json()), problem.is_perfect
    )


def test_missing_and_duplicate_episodes_cannot_be_perfect():
    problem = MiniGridProblem()
    raw = {"episode": 0, "final_obs": Obs(frame=None, reward=1, is_done=True).to_json(), "steps": 1}
    for episodes in ([raw], [raw, raw], []):
        result = score_episodes(
            episodes,
            task_name="g",
            executor="test",
            expected_episodes=2,
            compute_metrics=problem.compute_episode_metrics,
            aggregate_metrics=problem.aggregate_episode_metrics,
        )
        assert result.aggregate["evaluation_complete"] is False
        assert not _solved(
            SimpleNamespace(last_submission_results=result.to_json()), problem.is_perfect
        )


@pytest.mark.parametrize(
    "result",
    [
        {"aggregate": {"success_rate": 1}},  # no completeness evidence
        {"aggregate": {"success_rate": 1, "evaluation_complete": True, "n_errors": 1}},
        {"aggregate": {"success_rate": 1, "evaluation_complete": True}, "error": "failed"},
        {
            "aggregate": {"success_rate": 1, "evaluation_complete": True},
            "episodes": [{"error": "failed"}],
        },
    ],
)
def test_perfect_guard_rejects_errors_even_if_problem_says_perfect(result):
    assert not _solved(SimpleNamespace(last_submission_results=result), lambda _: True)


def test_arc_clean_win_can_be_perfect():
    problem = ArcAgiProblem()
    result = score_episodes(
        [
            {
                "episode": 0,
                "steps": 1,
                "final_obs": Obs(
                    frame=None,
                    is_done=True,
                    info={"state": "WIN", "levels_completed": 3, "win_levels": 3},
                ).to_json(),
            }
        ],
        task_name="g",
        executor="test",
        expected_episodes=1,
        compute_metrics=problem.compute_episode_metrics,
        aggregate_metrics=problem.aggregate_episode_metrics,
    )
    assert _solved(SimpleNamespace(last_submission_results=result.to_json()), problem.is_perfect)


def arc_decoder():
    shim = object.__new__(_ArcGymShim)
    shim._GameAction = SimpleNamespace(from_id=lambda value: value if 0 <= value <= 7 else None)
    return shim


@pytest.mark.parametrize(
    "action",
    [
        6,
        {"action": 6},
        {"action": 6, "data": {"x": 1}},
        {"action": 6, "data": {"x": 64, "y": 1}},
        {"action": 6, "data": {"x": "bad", "y": 1}},
        {},
        99,
        "6",
        None,
    ],
)
def test_arc_invalid_action_is_explicit(action):
    with pytest.raises(InvalidActionError):
        arc_decoder()._decode(action)


def test_arc_valid_click_and_directional_actions():
    assert arc_decoder()._decode({"action": 6, "data": {"x": 12, "y": 23}}) == (
        6,
        {"x": 12, "y": 23},
    )
    assert arc_decoder()._decode(1) == (1, None)


def test_minigrid_validation_does_not_misclassify_engine_errors():
    env = Mock()
    env.action_space.n = 7
    env.action_space.contains.side_effect = lambda a: type(a) is int and 0 <= a < 7
    shim = _ActionInfoShim(env)
    for action in (None, {}, 7, -1, True):
        with pytest.raises(InvalidActionError):
            shim.step(action)
    env.step.assert_not_called()
    env.step.side_effect = ValueError("engine broke")
    with pytest.raises(ValueError, match="engine broke") as exc:
        shim.step(1)
    assert not isinstance(exc.value, InvalidActionError)


def server_with_arc_validation():
    class Native(FakeNativeEnv):
        def step(self, action):
            arc_decoder()._decode(action)
            return super().step(action)

    server = EnvServer()
    server.register(
        "g",
        EnvSession(
            make_native=lambda: Native(goal=1),
            key="g",
            renderer=RawRenderer(),
            lifecycle=MultiInstancePolicy(),
        ),
    )
    return server


def test_invalid_action_survives_http_boundary():
    client = EnvClient(TestClient(server_with_arc_validation().app), "g")
    client.reset()
    with pytest.raises(InvalidActionError, match="ACTION6"):
        client.step(6)
    with pytest.raises(InvalidActionError, match="JSON"):
        client.step(object())


def test_plain_http_422_is_not_an_invalid_action():
    http = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(422, json={"detail": "env step failed: internal error"})
        ),
        base_url="http://test",
    )
    with pytest.raises(httpx.HTTPStatusError):
        EnvClient(http, "g").step(1)


@pytest.mark.parametrize("replay", [False, True])
async def test_subprocess_bad_click_scores_zero(tmp_path: Path, replay):
    async with serve_env(server_with_arc_validation(), "g", in_process=False) as conn:
        framework = tmp_path / "framework"
        framework.mkdir()
        (framework / "make_env.py").write_text(
            "from regact.envclient.client import EnvClient\n"
            f"def make_env(): return EnvClient.connect({conn.base_url!r}, 'g')\n"
        )
        path = solution(
            tmp_path, "class C:\n def act(self, obs): return 6\ndef get_controller(): return C()\n"
        )
        problem = ArcAgiProblem()
        executor = SandboxedExecutor(
            workdir=str(tmp_path),
            sandbox_wrap=lambda a: a,
            env_client=conn.client,
            shadow_replay=replay,
            compute_metrics=problem.compute_episode_metrics,
            aggregate_metrics=problem.aggregate_episode_metrics,
            failure_metrics=problem.failure_metrics,
        )
        result = executor.run(
            task_name="g",
            solution_path=path,
            output_path=str(tmp_path / "results.json"),
            lifecycle=Lifecycle.MULTI_INSTANCE,
            n_episodes=2,
        )
    assert result.aggregate["n_episodes"] == 2
    assert result.aggregate["n_errors"] == 2
    assert result.aggregate["mean_levels_completion_rate"] == 0
    assert all(e.error_category == ErrorCategory.AGENT_SOLUTION for e in result.episodes)


def test_shadow_replay_invalid_action_scores_zero():
    problem = MiniGridProblem()
    env = Mock()
    env.reset.return_value = Obs(frame=None)
    env.step.side_effect = InvalidActionError("bad action")
    result = replay_and_score(
        env,
        [{"episode": 0, "actions": [99]}],
        task_name="g",
        seed=0,
        compute_metrics=problem.compute_episode_metrics,
        aggregate_metrics=problem.aggregate_episode_metrics,
        failure_metrics=problem.failure_metrics,
    )
    assert result.aggregate["n_episodes"] == 1
    assert result.aggregate["success_rate"] == 0
    assert result.episodes[0].error_category == ErrorCategory.AGENT_SOLUTION


def test_factory_crash_scores_every_episode(tmp_path):
    result = ControllerExecutor(EpisodeEnv()).run(
        task_name="g",
        solution_path=solution(
            tmp_path, "def get_controller(): raise RuntimeError('factory broke')"
        ),
        output_path=str(tmp_path / "results.json"),
        lifecycle=Lifecycle.MULTI_INSTANCE,
        n_episodes=3,
    )
    assert result.aggregate["n_episodes"] == result.aggregate["n_errors"] == 3
    assert result.aggregate["success_rate"] == 0
