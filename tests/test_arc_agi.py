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


def test_discover_tasks_from_arcade_uses_live_listing() -> None:
    # Online/gateway mode has no local metadata: the catalog comes from the served
    # arcade's available_environments. The full game_id is preserved (online make()
    # needs it), keyed by the short id so --games filters match offline behavior.
    from regact.problems.arc_agi.tasks import discover_tasks_from_arcade

    fake_arcade = SimpleNamespace(
        available_environments=[
            SimpleNamespace(
                game_id="ls20-9607627b",
                title="LS20",
                baseline_actions=[19, 16, 34],
                level_tags=None,
            ),
            SimpleNamespace(
                game_id="vc33-abcd", title="VC33", baseline_actions=None, level_tags=("click",)
            ),
        ]
    )
    tasks = discover_tasks_from_arcade(fake_arcade)
    assert set(tasks) == {"ls20", "vc33"}
    assert tasks["ls20"].game_id == "ls20-9607627b"  # full served id for Arcade.make()
    assert tasks["ls20"].win_levels == 3
    assert tasks["vc33"].win_levels is None
    assert tasks["vc33"].tags == ("click",)


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


def test_renderer_last_frame_only_collapses_the_animation_stack() -> None:
    # A 3-grid intra-step stack: the default keeps only the final settled grid (1-element list).
    native = SimpleNamespace(frame=[[[1]], [[2]], [[3]]])
    assert ArcRenderer(last_frame_only=True).render(native, {}).frame == [[[3]]]
    # Opting out preserves the whole stack.
    assert ArcRenderer(last_frame_only=False).render(native, {}).frame == [[[1]], [[2]], [[3]]]


def test_validated_click_data_guards_action6() -> None:
    from regact.problems.arc_agi.problem import _validated_click_data

    assert _validated_click_data({"x": 3, "y": 5}) == {"x": 3, "y": 5}
    assert _validated_click_data({"x": "3", "y": "5"}) == {"x": 3, "y": 5}  # coerced to int
    for bad in (
        None,
        {},
        {"x": 1},
        {"y": 1},
        {"x": 1, "y": 99},
        {"x": -1, "y": 1},
        {"x": "a", "y": 1},
    ):
        with pytest.raises(ValueError, match="ACTION6"):
            _validated_click_data(bad)


def test_render_frame_colorizes_grid_for_video() -> None:
    img = _problem().render_frame(Obs(frame=[[[0, 1], [2, 3]]]))  # a 2x2 grid stack
    assert img is not None and img.shape == (16, 16, 3) and img.dtype.name == "uint8"
    assert tuple(img[0, 0]) == (255, 255, 255)  # value 0 -> white (ARC-AGI-3 palette)
    assert tuple(img[0, 8]) == (204, 204, 204)  # value 1 -> off-white (after x8 upscale)
    assert _problem().render_frame(Obs(frame=None)) is None


def test_helper_template_is_import_free() -> None:
    [tmpl] = _problem().helper_templates("ls20")  # default helper (to_png off)
    assert tmpl.relpath == "code_library/arc_agi_helper.py"
    assert "import" not in tmpl.content.split('"""', 2)[-1]  # no imports in the code body
    assert "def complex_action" in tmpl.content
    assert "ACTION6 = 6" in tmpl.content
    assert "def to_png" not in tmpl.content  # the renderer is opt-in


def test_helper_to_png_is_gated_and_game_library_free() -> None:
    from regact.config.schema import HelperConfig

    [on] = _problem().helper_templates("ls20", helper=HelperConfig(to_png=True))
    assert "def to_png" in on.content
    assert "from PIL import Image" in on.content  # lazy image import, not the game engine
    for secret in ("import arcengine", "import arc_agi"):
        assert secret not in on.content
    compile(on.content, "arc_agi_helper.py", "exec")  # the concatenated helper is valid Python


def test_build_prompt_informative_vs_minimal() -> None:
    problem = _problem()
    info = problem.build_prompt("ls20", info_mode=InfoMode.INFORMATIVE)
    assert "ARC-AGI-3" in info
    assert "obs.available_actions" in info  # the Observation section names the field
    assert "Levels to win" in info  # the one metadata line kept (game id + human baseline removed)
    assert "Game id" not in info and "Human baseline" not in info
    assert "{frame_desc}" not in info  # the obs-frame placeholder was substituted
    assert "multiple frames" in info  # default obs_mode (raw) describes the full animation stack
    # raw_last_frame_only swaps in the single-grid description instead
    single = problem.build_prompt(
        "ls20", info_mode=InfoMode.INFORMATIVE, obs_mode=ObsMode.RAW_LAST_FRAME_ONLY
    )
    assert "settled grid" in single and "multiple frames" not in single
    assert "How to read and solve" in info  # the ARC-approach advice block
    # actions are described live in the first observation (render_obs_text), not statically
    assert "## Actions" not in info

    minimal = problem.build_prompt("ls20", info_mode=InfoMode.MINIMAL)
    assert "Discover the rules" in minimal


def test_actions_for_ids_includes_only_available_actions() -> None:
    """Each action block appears only when its id is in the live available_actions."""
    from regact.problems.arc_agi.problem import _actions_for_ids

    assert "Directional actions" in _actions_for_ids([1, 2, 3, 4])
    assert "ACTION5" not in _actions_for_ids([1, 2, 3, 4])
    everything = _actions_for_ids([1, 5, 6, 7])
    assert "ACTION5" in everything and "ACTION7" in everything
    assert "click action ACTION6" in everything
    just_seven = _actions_for_ids([7])
    assert "ACTION7" in just_seven and "Directional actions" not in just_seven


def test_render_obs_text_compact_grid() -> None:
    """The text obs render is a header + hex grid + the live action descriptions."""
    from regact.envclient.obs import Obs

    obs = Obs(
        frame=[[[0, 1], [15, 2]]],  # a stack of one 2x2 grid
        available_actions=[1, 6],
        info={"state": "NOT_FINISHED", "levels_completed": 1, "win_levels": 8},
    )
    text = _problem().render_obs_text(obs)
    assert text is not None
    assert "available_actions=[1, 6]" in text
    assert "levels_completed=1/8" in text
    assert "01" in text and "f2" in text  # 15 -> "f", per-cell hex
    # actions loaded from the available ids (1 and 6, not 5/7)
    assert "Directional actions" in text and "click action ACTION6" in text
    assert "ACTION5" not in text and "ACTION7" not in text


def test_obs_renderer_maps_obs_mode_to_last_frame_only() -> None:
    p = _problem()
    assert p.obs_renderer("ls20", mode=ObsMode.RAW)._last_frame_only is False
    assert p.obs_renderer("ls20", mode=ObsMode.RAW_LAST_FRAME_ONLY)._last_frame_only is True


def test_compute_episode_metrics_from_obs() -> None:
    problem = _problem()
    won = problem.compute_episode_metrics(
        Obs(
            frame=None, is_done=True, info={"state": "WIN", "levels_completed": 7, "win_levels": 7}
        ),
        steps=120,
    )
    assert won == {
        "success": True,
        "steps": 120,
        "levels_completed": 7,
        "win_levels": 7,
        "level_completion_rate": 1.0,
    }


def test_mean_levels_completion_rate_is_graded() -> None:
    """The headline ARC metric is mean_levels_completion_rate - the mean PROPORTION of levels
    cleared (graded); win_rate keeps the binary full-win. A 3/6 partial (ft09's) scores 0.5, not 0.
    ARC has NO 'success_rate' key (it would be all-or-nothing, which is win_rate)."""
    problem = _problem()
    partial = problem.compute_episode_metrics(
        Obs(
            frame=None,
            is_done=True,
            info={"state": "NOT_FINISHED", "levels_completed": 3, "win_levels": 6},
        ),
        steps=25,
    )
    assert partial["level_completion_rate"] == 0.5 and partial["success"] is False
    agg = problem.aggregate_episode_metrics([partial])
    assert agg["mean_levels_completion_rate"] == 0.5  # graded, not 0
    assert "success_rate" not in agg  # renamed away from the misleading all-or-nothing name
    assert agg["win_rate"] == 0.0  # not a full win
    assert agg["mean_levels_completed"] == 3.0
    # win_levels=0 must not divide by zero:
    zero = problem.compute_episode_metrics(
        Obs(frame=None, is_done=True, info={"levels_completed": 0, "win_levels": 0}), steps=1
    )
    assert zero["level_completion_rate"] == 0.0


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
