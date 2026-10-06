"""Tests for the ARC-AGI-3 efficiency metrics RHAE and LRHAE-Uncapped."""

import json
from pathlib import Path

from regact.problems.arc_agi.scoring import (
    actions_per_level_from_milestones,
    actions_per_level_from_trace,
    rhae_from_results,
    rhae_score,
    summarize_run,
)


def test_actions_per_level_from_milestones() -> None:
    # Level 1 cleared at step 20, level 2 at step 45 -> 20 then 25 actions.
    ms = [
        {"step": 20, "description": "level completed (1/3)"},
        {"step": 45, "description": "level completed (2/3)"},
        {"step": 45, "description": "game won"},  # non-level milestone ignored
    ]
    assert actions_per_level_from_milestones(ms, total_steps=45, levels_completed=2) == [20, 25]
    # Never trust more completion marks than levels_completed.
    assert actions_per_level_from_milestones(ms, total_steps=45, levels_completed=1) == [20]


def test_surpassing_human_is_capped_by_completion() -> None:
    # 31 actions vs 40 baseline -> r>1, hits the 1.15 level cap; but the game is a single fully
    # completed level, so completion=1.0 caps RHAE at 1.0. LRHAE-Uncapped keeps r.
    r = rhae_score(
        baseline_actions=[40], actions_per_level=[31], levels_completed=1, total_levels=1
    )
    assert r.rhae == 1.0
    assert abs(r.lrhae_uncapped - 40 / 31) < 1e-9


def test_inefficiency_penalized_more_by_squaring() -> None:
    # 47 vs 35 baseline -> r=35/47~=0.745: RHAE squares it (~0.555), LRHAE-Uncapped does not.
    r = rhae_score(
        baseline_actions=[35], actions_per_level=[47], levels_completed=1, total_levels=1
    )
    assert 0.55 < r.rhae < 0.56
    assert 0.74 < r.lrhae_uncapped < 0.75


def test_uncapped_lets_one_level_pay_for_another() -> None:
    # Level 1 at r=2 (1.15 capped), level 2 at r=0.5 (0.25 squared).
    r = rhae_score(
        baseline_actions=[20, 20], actions_per_level=[10, 40], levels_completed=2, total_levels=2
    )
    assert abs(r.rhae - (1.15 * 1 + 0.25 * 2) / 3) < 1e-9
    assert abs(r.lrhae_uncapped - (2 * 1 + 0.5 * 2) / 3) < 1e-9


def test_zero_when_nothing_completed() -> None:
    r = rhae_score(
        baseline_actions=[35, 40], actions_per_level=[], levels_completed=0, total_levels=2
    )
    assert r.rhae == 0.0 and r.lrhae_uncapped == 0.0


def test_no_baseline_is_zero() -> None:
    r = rhae_score(baseline_actions=None, actions_per_level=[10], levels_completed=1)
    assert r.rhae == 0.0 and r.lrhae_uncapped == 0.0


def test_weights_later_levels_more() -> None:
    # Level 1 solved at human efficiency (r=1 -> S=1), level 2 at half efficiency (r=0.5).
    # Level 2 carries weight 2, so it drags the weighted average below the plain mean.
    r = rhae_score(
        baseline_actions=[20, 20], actions_per_level=[20, 40], levels_completed=2, total_levels=2
    )
    # RHAE: (1.0*1 + 0.25*2) / (1+2) = 0.5 ; LRHAE-Uncapped: (1.0*1 + 0.5*2) / 3 = 0.6667.
    assert r.rhae == 0.5
    assert abs(r.lrhae_uncapped - 2 / 3) < 1e-9


def test_partial_completion_bounds_the_score() -> None:
    # 1 of 2 levels cleared at human efficiency: weighted avg (1*1)/(1+2), the solved-share cap.
    r = rhae_score(
        baseline_actions=[10, 10], actions_per_level=[10], levels_completed=1, total_levels=2
    )
    assert abs(r.rhae - 1 / 3) < 1e-9


def test_rhae_from_results_end_to_end() -> None:
    results = {
        "task": "ls20",
        "episodes": [
            {
                "episode": 0,
                "milestones": [
                    {"step": 20, "description": "level completed (1/3)"},
                    {"step": 45, "description": "level completed (2/3)"},
                ],
                "metrics": {"levels_completed": 2, "win_levels": 3, "steps": 45},
            }
        ],
    }
    r = rhae_from_results(results, baseline_actions=[19, 16, 34])
    assert r is not None
    assert r.levels_completed == 2
    assert r.total_levels == 3
    assert r.actions_per_level == (20, 25)
    assert 0.0 < r.rhae < 1.0
    assert r.lrhae_uncapped > r.rhae  # both levels below human efficiency: linear scores higher


def test_summarize_run_reads_disk(tmp_path: Path) -> None:
    d = tmp_path / "smoke" / "ls20" / "workdir" / "submissions" / "final"
    d.mkdir(parents=True)
    (d / "results.json").write_text(
        json.dumps(
            {
                "task": "ls20",
                "episodes": [
                    {
                        "episode": 0,
                        "milestones": [{"step": 20, "description": "level completed (1/3)"}],
                        "metrics": {"levels_completed": 1, "win_levels": 3, "steps": 20},
                    }
                ],
            }
        )
    )
    out = summarize_run(
        str(tmp_path / "smoke"), ["ls20", "vc33"], {"ls20": [19, 16, 34], "vc33": None}
    )
    assert "ls20" in out and "RHAE/LRHAE-Uncapped" in out
    assert "1 games scored" in out  # ls20 scored, vc33 has no results


def test_summarize_run_handles_missing_results(tmp_path: Path) -> None:
    # No results on disk at all -> every game shows a dash, no crash.
    out = summarize_run(str(tmp_path / "nope"), ["ls20"], {"ls20": [19, 16]})
    assert "0 games scored" in out


def test_actions_per_level_from_trace_uses_first_completions() -> None:
    # Level 1 first done after 30 actions; a fresh environment then replays it, and level 2 is
    # first done after 100: the replay is charged to level 2. Falling back to 0 changes nothing.
    trace = [(10, 0), (30, 1), (31, 0), (60, 1), (100, 2), (101, 0)]
    assert actions_per_level_from_trace(trace) == [30, 70]
    assert actions_per_level_from_trace([]) == []


def test_beating_humans_on_one_level_does_not_pay_for_another() -> None:
    # Level 1 of 2 at 1.15 (capped): the weighted average 1.15/3 is above the weighted share of
    # solved levels, 1/3, which caps it (the arc library scorecard does the same).
    r = rhae_score(
        baseline_actions=[100, 100], actions_per_level=[10], levels_completed=1, total_levels=2
    )
    assert abs(r.rhae - 1 / 3) < 1e-9 and abs(r.lrhae_uncapped - 10 / 3) < 1e-9
