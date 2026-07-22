"""Tests for the ARC-AGI-3 RHAE score (the competition metric, computed offline)."""

import json
from pathlib import Path

from regact.problems.arc_agi.scoring import (
    actions_per_level_from_milestones,
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


def test_rhae_caps_at_surpassing_human() -> None:
    # 31 actions vs 40 baseline -> (40/31)^2*100 = 166 -> level cap 115, but the game cap
    # (one level, weight 1) is 100 -> min(115, 100) = 100.
    r = rhae_score(
        baseline_actions=[40], actions_per_level=[31], levels_completed=1, total_levels=1
    )
    assert r.score == 100.0


def test_rhae_penalizes_inefficiency() -> None:
    # 47 actions vs 35 baseline -> (35/47)^2*100 ~= 55.5 (worse than human => below 100).
    r = rhae_score(
        baseline_actions=[35], actions_per_level=[47], levels_completed=1, total_levels=1
    )
    assert 55.0 < r.score < 56.0


def test_rhae_zero_when_nothing_completed() -> None:
    r = rhae_score(
        baseline_actions=[35, 40], actions_per_level=[], levels_completed=0, total_levels=2
    )
    assert r.score == 0.0


def test_rhae_no_baseline_is_zero() -> None:
    r = rhae_score(baseline_actions=None, actions_per_level=[10], levels_completed=1)
    assert r.score == 0.0


def test_rhae_weights_later_levels_more() -> None:
    # Two levels cleared, second is weighted 2x. Same efficiency each -> weighted mean.
    r = rhae_score(
        baseline_actions=[20, 20], actions_per_level=[20, 20], levels_completed=2, total_levels=2
    )
    # Each level scores 100; weights 1 and 2; capped by fraction completed (all) => 100.
    assert r.score == 100.0


def test_rhae_from_results_end_to_end(tmp_path: Path) -> None:
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
    assert 0.0 < r.score < 100.0


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
        str(tmp_path), "smoke", ["ls20", "vc33"], {"ls20": [19, 16, 34], "vc33": None}
    )
    assert "ls20" in out and "RHAE proxy" in out
    assert "1 games scored" in out  # ls20 scored, vc33 has no results


def test_summarize_run_handles_missing_results(tmp_path: Path) -> None:
    # No results on disk at all -> every game shows a dash, no crash.
    out = summarize_run(str(tmp_path), "nope", ["ls20"], {"ls20": [19, 16]})
    assert "0 games scored" in out
