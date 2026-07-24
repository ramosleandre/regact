"""ARC-AGI-3 RHAE score — the competition's own metric, computed offline.

This mirrors ``arc_agi.scorecard.EnvironmentScoreCalculator`` (and the formula the
Milestone-1 winners compute locally): per level, efficiency versus the human action
baseline, squared, capped, and level-weighted. It lets us judge an *offline* run on
the 25 public games by the metric that actually scores the competition — a much better
signal than "won / did not win", because ARC-AGI-3 rewards *few actions*, not just wins.

Offline only estimates the score (the public games' baselines are known); the real
competition score is over the 110 hidden games whose baselines are private and OOD.

Inputs come from data we already record per episode:
- ``baseline_actions`` — the per-level human benchmark (from the game metadata).
- ``levels_completed`` — how many levels the controller cleared (from the final obs).
- ``actions_per_level`` — actions spent on each cleared level, derived from the
  "level completed" milestones (each carries the step at which a level was cleared).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Per-level efficiency is capped at 1.15x the human baseline (surpassing humans earns no
# extra), then scaled to a 0-115 points band.
_LEVEL_CAP = 115.0


@dataclass(frozen=True)
class RhaeResult:
    """One game's RHAE breakdown."""

    score: float  # 0-100 (can be < the naive mean; capped by fraction of levels scored)
    levels_completed: int
    total_levels: int
    actions_per_level: tuple[int, ...]  # actions on each COMPLETED level (index 0 = level 1)
    baseline_actions: tuple[int, ...]  # human baseline per level


def actions_per_level_from_milestones(
    milestones: Sequence[dict[str, object]], *, total_steps: int, levels_completed: int
) -> list[int]:
    """Actions spent on each completed level, from the level-completion milestones.

    Each "level completed" milestone carries the ``step`` at which that level was
    cleared; the actions for a level are the steps since the previous completion. The
    trailing (uncompleted) level, if any, is not counted — only cleared levels score.
    """
    completion_steps = sorted(
        int(str(m["step"]))
        for m in milestones
        if isinstance(m.get("description"), str) and "level completed" in str(m["description"])
    )
    # Guard: never trust more completion marks than levels_completed says were cleared.
    completion_steps = completion_steps[:levels_completed]
    out: list[int] = []
    prev = 0
    for step in completion_steps:
        out.append(max(0, step - prev))
        prev = step
    return out


def rhae_score(
    *,
    baseline_actions: Sequence[int] | None,
    actions_per_level: Sequence[int],
    levels_completed: int,
    total_levels: int | None = None,
) -> RhaeResult:
    """The ARC-AGI-3 score for one game.

    Per level ``l`` completed with ``a>0`` actions and human baseline ``b``:
    ``min(115, (b / a)^2 x 100)``; uncompleted levels score 0. Level weight is
    ``l`` (1-indexed). Final = ``sum (score_l x w_l) / sum  w_l``, capped at
    ``sum  w_l(scored>0) / sum  w_l x 100`` — so you cannot bank credit on levels you
    did not reach. Returns 0 when there is no baseline.
    """
    baselines = tuple(int(x) for x in (baseline_actions or ()))
    n_levels = total_levels if total_levels is not None else len(baselines)
    if not baselines or n_levels == 0:
        return RhaeResult(0.0, levels_completed, n_levels, tuple(actions_per_level), baselines)

    total_score = 0.0
    total_weights = 0
    max_weights = 0
    for level_idx in range(n_levels):
        weight = level_idx + 1
        total_weights += weight
        completed = level_idx < levels_completed
        actions = actions_per_level[level_idx] if level_idx < len(actions_per_level) else 0
        baseline = baselines[level_idx] if level_idx < len(baselines) else 0
        if completed and actions > 0 and baseline > 0:
            level_score = min(_LEVEL_CAP, (baseline / actions) ** 2 * 100)
        else:
            level_score = 0.0
        if level_score > 0:
            max_weights += weight
        total_score += level_score * weight

    if total_weights == 0:
        return RhaeResult(0.0, levels_completed, n_levels, tuple(actions_per_level), baselines)
    score = total_score / total_weights
    max_score = max_weights / total_weights * 100
    return RhaeResult(
        min(score, max_score),
        levels_completed,
        n_levels,
        tuple(actions_per_level),
        baselines,
    )


def _latest_results_json(game_output_dir: str) -> dict[str, object] | None:
    """The most recent ``results.json`` for a game (prefer ``final``, else the highest n)."""
    subs = os.path.join(game_output_dir, "workdir", "submissions")
    if not os.path.isdir(subs):
        return None
    final = os.path.join(subs, "final", "results.json")
    if os.path.isfile(final):
        path = final
    else:
        numbered = sorted(
            (d for d in os.listdir(subs) if d.isdigit()),
            key=int,
        )
        candidates = [
            os.path.join(subs, d, "results.json")
            for d in numbered
            if os.path.isfile(os.path.join(subs, d, "results.json"))
        ]
        if not candidates:
            return None
        path = candidates[-1]
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)  # type: ignore[no-any-return]
    except (OSError, json.JSONDecodeError):
        return None


def rhae_from_results(
    results: dict[str, object], *, baseline_actions: Sequence[int] | None
) -> RhaeResult | None:
    """Compute a game's RHAE from its serialized ``results.json`` + the human baseline.

    Reads the first non-errored episode's ``milestones`` (to derive per-level actions)
    and ``metrics`` (levels_completed / steps). Returns ``None`` if there is no usable
    episode.
    """
    episodes = results.get("episodes")
    if not isinstance(episodes, list):
        return None
    for ep in episodes:
        if not isinstance(ep, dict) or ep.get("error"):
            continue
        metrics = ep.get("metrics") if isinstance(ep.get("metrics"), dict) else {}
        raw_ms = ep.get("milestones")
        milestones: list[dict[str, object]] = raw_ms if isinstance(raw_ms, list) else []
        levels = int(metrics.get("levels_completed", 0) or 0)  # type: ignore[union-attr]
        total_levels = int(metrics.get("win_levels", 0) or 0) or (  # type: ignore[union-attr]
            len(baseline_actions) if baseline_actions else 0
        )
        steps = int(metrics.get("steps", 0) or 0)  # type: ignore[union-attr]
        apl = actions_per_level_from_milestones(
            milestones,
            total_steps=steps,
            levels_completed=levels,
        )
        return rhae_score(
            baseline_actions=baseline_actions,
            actions_per_level=apl,
            levels_completed=levels,
            total_levels=total_levels or None,
        )
    return None


def _game_status(game_dir: str, results: dict[str, object] | None) -> str:
    """Diagnose why a game has no score: distinguish 'agent never submitted' from
    'submitted but cleared no level' from 'ran but produced nothing' — so an empty
    recap tells us WHERE the pipeline stopped, not just that it's empty."""
    subs = os.path.join(game_dir, "workdir", "submissions")
    if not os.path.isdir(subs):
        return "no-run"  # the game workdir/submissions dir was never created
    if results is None:
        return "no-submit"  # ran, but no results.json → SubmitSolution never produced one
    episodes = results.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        return "empty"
    ep = next((e for e in episodes if isinstance(e, dict) and not e.get("error")), None)
    if ep is None:
        return "err"
    metrics = ep.get("metrics") if isinstance(ep.get("metrics"), dict) else {}
    levels = int(metrics.get("levels_completed", 0) or 0)  # type: ignore[union-attr]
    return "0-levels" if levels == 0 else f"{levels}-levels"


def summarize_run(
    output_root: str,
    experiment_name: str,
    tasks: Sequence[str],
    baselines: Mapping[str, Sequence[int] | None],
) -> str:
    """A human-readable RHAE recap of an offline run: one line per game + an aggregate.

    ``baselines`` maps game key -> per-level human baseline (from the task catalog). A game
    that did not score shows a STATUS (no-run / no-submit / 0-levels / no-baseline) instead
    of a bare dash, so the recap explains where the pipeline stopped. Returns the formatted
    block (the caller prints it) so this stays testable.
    """
    lines = [
        f"=== ARC-AGI-3 run summary ({len(tasks)} games) — RHAE proxy (offline) ===",
        f"  {'game':<8} {'levels':>8}  {'actions/baseline':>18}  {'RHAE':>6}  status",
    ]
    scored: list[float] = []
    wins = 0
    status_counts: dict[str, int] = {}
    for task in tasks:
        game_dir = os.path.join(output_root, experiment_name, task)
        results = _latest_results_json(game_dir)
        status = _game_status(game_dir, results)
        rhae = rhae_from_results(results, baseline_actions=baselines.get(task)) if results else None
        if rhae is None or (baselines.get(task) is None):
            if rhae is not None and baselines.get(task) is None:
                status = "no-baseline"
            status_counts[status] = status_counts.get(status, 0) + 1
            lines.append(f"  {task:<8} {'—':>8}  {'—':>18}  {'—':>6}  {status}")
            continue
        acts = sum(rhae.actions_per_level)
        base = sum(rhae.baseline_actions[: rhae.levels_completed]) if rhae.baseline_actions else 0
        lvl = f"{rhae.levels_completed}/{rhae.total_levels}"
        ab = f"{acts} / {base}" if base else f"{acts} / —"
        lines.append(f"  {task:<8} {lvl:>8}  {ab:>18}  {rhae.score:>6.1f}  {status}")
        scored.append(rhae.score)
        status_counts[status] = status_counts.get(status, 0) + 1
        if rhae.total_levels and rhae.levels_completed >= rhae.total_levels:
            wins += 1
    mean = sum(scored) / len(scored) if scored else 0.0
    lines.append("  " + "-" * 54)
    lines.append(
        f"  mean RHAE {mean:.1f}  |  {wins} wins/{len(tasks)}  |  {len(scored)} games scored"
    )
    if status_counts:
        breakdown = "  ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
        lines.append(f"  status breakdown: {breakdown}")
    return "\n".join(lines)
