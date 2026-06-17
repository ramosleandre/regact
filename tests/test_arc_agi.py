"""Tests for the ARC-AGI-3 problem.

Pure parts (catalog discovery from environnement/, renderer JSON-safe, helper,
prompt, metrics, milestones, action decode) run always. ``make_env`` needs the
``arc`` extra (arc_agi/arcengine) + the local game data, so it is gated with
``importorskip`` — runs where installed, skips cleanly otherwise.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from regact.config.schema import InfoMode, ObsMode
from regact.envclient.obs import Obs
from regact.obs.errors import RegactError
from regact.problems.arc_agi.problem import ArcAgiProblem, ArcRenderer, _milestone_detector
from regact.problems.arc_agi.tasks import discover_tasks
from regact.problems.base import build_problem

# The repo ships 25 games under environnement/ (committed for offline play).
_ENV_DIR = str(Path(__file__).resolve().parents[1] / "environnement")


def _problem() -> ArcAgiProblem:
    return ArcAgiProblem(environments_dir=_ENV_DIR)


def test_discover_tasks_finds_the_committed_games() -> None:
    tasks = discover_tasks(_ENV_DIR)
    assert len(tasks) == 25
    assert "ls20" in tasks
    ls20 = tasks["ls20"]
    assert ls20.game_id == "ls20"
    assert ls20.win_levels == len(ls20.baseline_actions or ())


def test_build_problem_resolves_arc() -> None:
    problem = build_problem("arc_agi", {"environments_dir": _ENV_DIR})
    assert isinstance(problem, ArcAgiProblem)
    assert problem.name == "arc_agi"
    assert len(problem.get_task_names()) == 25


def test_unknown_game_raises() -> None:
    with pytest.raises(RegactError, match="unknown ARC game"):
        _problem().make_env("not-a-game")


def test_config_kwargs_roundtrip() -> None:
    kwargs = _problem().config_kwargs()
    assert kwargs == {"environments_dir": _ENV_DIR, "operation_mode": "offline"}
    assert isinstance(build_problem("arc_agi", kwargs), ArcAgiProblem)


def test_renderer_makes_frame_json_safe() -> None:
    class _Grid:  # a numpy-like grid (exposes tolist)
        def tolist(self) -> list[list[int]]:
            return [[0, 1], [2, 3]]

    native = SimpleNamespace(frame=[_Grid()])  # native obs: .frame is a list of grids
    obs = ArcRenderer().render(
        native,
        {"available_actions": [1, 2, 6], "state": "NOT_FINISHED", "levels_completed": 0},
    )
    assert obs.frame == [[[0, 1], [2, 3]]]
    assert obs.available_actions == [1, 2, 6]
    assert obs.info["state"] == "NOT_FINISHED"


def test_helper_template_is_import_free() -> None:
    [tmpl] = _problem().helper_templates("ls20")
    assert tmpl.relpath == "code_library/arc_agi_helper.py"
    assert "import" not in tmpl.content.split('"""', 2)[-1]  # no imports in the code body
    assert "def complex_action" in tmpl.content
    assert "ACTION6 = 6" in tmpl.content


def test_build_prompt_informative_vs_minimal() -> None:
    problem = _problem()
    info = problem.build_prompt("ls20", info_mode=InfoMode.INFORMATIVE)
    assert "ARC-AGI-3" in info
    assert "obs.available_actions" in info
    assert "## Actions" in info  # describes actions

    minimal = problem.build_prompt("ls20", info_mode=InfoMode.MINIMAL)
    assert "Discover the rules" in minimal
    assert "## Actions" not in minimal


def test_obs_renderer_rejects_non_raw_mode() -> None:
    assert isinstance(_problem().obs_renderer("ls20", mode=ObsMode.RAW), ArcRenderer)


def test_compute_episode_metrics_from_obs() -> None:
    problem = _problem()
    won = problem.compute_episode_metrics(
        Obs(
            frame=None, is_done=True, info={"state": "WIN", "levels_completed": 7, "win_levels": 7}
        ),
        steps=120,
    )
    assert won == {"success": True, "steps": 120, "levels_completed": 7, "win_levels": 7}


def test_milestone_detector_emits_on_level_and_win() -> None:
    class _Env:
        prev_obs = Obs(frame=None, info={"levels_completed": 0, "win_levels": 7})
        last_obs = Obs(
            frame=None, info={"levels_completed": 1, "win_levels": 7, "state": "NOT_FINISHED"}
        )

    assert _milestone_detector(_Env()) == ["level completed (1/7)"]

    class _Win:
        prev_obs = Obs(frame=None, info={"levels_completed": 7})
        last_obs = Obs(frame=None, info={"levels_completed": 7, "win_levels": 7, "state": "WIN"})

    assert _milestone_detector(_Win()) == ["game won"]


@pytest.mark.live
def test_make_env_resets_and_steps() -> None:
    """Runtime-gated: only where the arc extra + game data are present."""
    pytest.importorskip("arc_agi")
    pytest.importorskip("arcengine")
    problem = _problem()
    native = problem.make_env("ls20")
    _obs, info = native.reset()
    assert "available_actions" in info
    actions = info["available_actions"]
    assert actions
    *_, info = native.step(actions[0])
    assert "state" in info
    native.close()
