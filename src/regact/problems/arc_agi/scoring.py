"""ARC-AGI-3 efficiency metrics (RHAE / LRHAE), computed offline from our own recorded data.

Implemented directly from the metric's definition (not derived from the arc library's scorecard),
so we own the formula and its variants:

Per level ``l`` (1-indexed) the agent completed with ``a>0`` actions, given the human baseline
``h`` actions, the ratio is ``r = h / a`` and the level score is::

    S_l = min(1.15, r**2)     # RHAE   - squared ratio (the competition's own choice)
    S_l = min(1.15, r)        # LRHAE  - the linear variant, EXACTLY RHAE without the squaring

An unsolved level scores 0. Levels are weighted by their number (level ``l`` has weight ``l``). The
game score is the level scores' weighted average, then capped by the fraction of levels solved so
credit cannot be banked on levels never reached::

    E = min( levels_completed / total_levels ,  sum_l(S_l * l) / sum_l(l) )

RHAE and LRHAE differ only in the per-level exponent, so both come from one pass. Scores are on
a 0-1 scale (the per-level cap 1.15 is only reachable when you beat the human badly; the completion
cap keeps the game score in [0, 1]). A dataset's score is the mean of its games' scores.

Inputs come from data we already record per episode:
- ``baseline_actions`` - the per-level human baseline (median human actions, from game metadata).
- ``levels_completed`` - how many levels the controller cleared (from the final obs).
- ``actions_per_level`` - actions on each cleared level, from the "level completed" milestones.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Per-level efficiency is capped at 1.15x the human baseline (surpassing humans earns no extra).
_LEVEL_CAP = 1.15


@dataclass(frozen=True)
class RhaeResult:
    """One game's efficiency breakdown (both metric variants)."""

    rhae: float  # 0-1, squared ratio (the competition metric)
    lrhae: float  # 0-1, linear ratio - RHAE without the squaring
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
    trailing (uncompleted) level, if any, is not counted - only cleared levels score.
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


def _efficiency_score(
    *,
    baselines: Sequence[int],
    actions_per_level: Sequence[int],
    levels_completed: int,
    n_levels: int,
    square: bool,
) -> float:
    """The game score ``E`` for one exponent (``square`` -> RHAE, else LRHAE)."""
    if not baselines or n_levels == 0:
        return 0.0
    total_weight = 0
    weighted_sum = 0.0
    for idx in range(n_levels):
        weight = idx + 1  # level l is 1-indexed and weighted by l
        total_weight += weight
        actions = actions_per_level[idx] if idx < len(actions_per_level) else 0
        baseline = baselines[idx] if idx < len(baselines) else 0
        if idx < levels_completed and actions > 0 and baseline > 0:
            ratio = baseline / actions
            level_score = min(_LEVEL_CAP, ratio * ratio if square else ratio)
        else:
            level_score = 0.0
        weighted_sum += level_score * weight
    weighted_avg = weighted_sum / total_weight if total_weight else 0.0
    completion = levels_completed / n_levels
    return min(completion, weighted_avg)


def rhae_score(
    *,
    baseline_actions: Sequence[int] | None,
    actions_per_level: Sequence[int],
    levels_completed: int,
    total_levels: int | None = None,
) -> RhaeResult:
    """RHAE and LRHAE for one game (both on a 0-1 scale). Zero when there is no baseline."""
    baselines = tuple(int(x) for x in (baseline_actions or ()))
    n_levels = total_levels if total_levels is not None else len(baselines)
    shared = {
        "baselines": baselines,
        "actions_per_level": actions_per_level,
        "levels_completed": levels_completed,
        "n_levels": n_levels,
    }
    return RhaeResult(
        rhae=_efficiency_score(**shared, square=True),
        lrhae=_efficiency_score(**shared, square=False),
        levels_completed=levels_completed,
        total_levels=n_levels,
        actions_per_level=tuple(actions_per_level),
        baseline_actions=baselines,
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
    """Compute a game's RHAE/LRHAE from its serialized ``results.json`` + the human baseline.

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
    'submitted but cleared no level' from 'ran but produced nothing' - so an empty
    recap tells us WHERE the pipeline stopped, not just that it's empty."""
    subs = os.path.join(game_dir, "workdir", "submissions")
    if not os.path.isdir(subs):
        return "no-run"  # the game workdir/submissions dir was never created
    if results is None:
        return "no-submit"  # ran, but no results.json -> SubmitSolution never produced one
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
    run_dir: str,
    tasks: Sequence[str],
    baselines: Mapping[str, Sequence[int] | None],
) -> str:
    """A human-readable RHAE/LRHAE recap of an offline run: one line per game + an aggregate.

    ``run_dir`` is the directory that run owns (each run is timestamped, so the path
    cannot be rebuilt from the config alone). ``baselines`` maps game key -> per-level
    human baseline (from the task catalog). A game that did not score shows a STATUS
    (no-run / no-submit / 0-levels / no-baseline) instead of a bare dash, so the recap
    explains where the pipeline stopped. Returns the formatted block (the caller prints
    it) so this stays testable.
    """
    lines = [
        f"=== ARC-AGI-3 run summary ({len(tasks)} games) - RHAE/LRHAE (offline) ===",
        f"  {'game':<8} {'levels':>8}  {'actions/baseline':>18}  {'RHAE':>6}  {'LRHAE':>6}  status",
    ]
    rhaes: list[float] = []
    lrhaes: list[float] = []
    wins = 0
    status_counts: dict[str, int] = {}
    for task in tasks:
        game_dir = os.path.join(run_dir, task)
        results = _latest_results_json(game_dir)
        status = _game_status(game_dir, results)
        rhae = rhae_from_results(results, baseline_actions=baselines.get(task)) if results else None
        if rhae is None or (baselines.get(task) is None):
            if rhae is not None and baselines.get(task) is None:
                status = "no-baseline"
            status_counts[status] = status_counts.get(status, 0) + 1
            lines.append(f"  {task:<8} {'-':>8}  {'-':>18}  {'-':>6}  {'-':>6}  {status}")
            continue
        acts = sum(rhae.actions_per_level)
        base = sum(rhae.baseline_actions[: rhae.levels_completed]) if rhae.baseline_actions else 0
        lvl = f"{rhae.levels_completed}/{rhae.total_levels}"
        ab = f"{acts} / {base}" if base else f"{acts} / -"
        lines.append(
            f"  {task:<8} {lvl:>8}  {ab:>18}  {rhae.rhae:>6.3f}  {rhae.lrhae:>6.3f}  {status}"
        )
        rhaes.append(rhae.rhae)
        lrhaes.append(rhae.lrhae)
        status_counts[status] = status_counts.get(status, 0) + 1
        if rhae.total_levels and rhae.levels_completed >= rhae.total_levels:
            wins += 1
    mean_rhae = sum(rhaes) / len(rhaes) if rhaes else 0.0
    mean_lrhae = sum(lrhaes) / len(lrhaes) if lrhaes else 0.0
    lines.append("  " + "-" * 60)
    lines.append(
        f"  mean RHAE {mean_rhae:.3f}  |  mean LRHAE {mean_lrhae:.3f}  |  "
        f"{wins} wins/{len(tasks)}  |  {len(rhaes)} games scored"
    )
    if status_counts:
        breakdown = "  ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
        lines.append(f"  status breakdown: {breakdown}")
    return "\n".join(lines)
