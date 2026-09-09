#!/usr/bin/env python3
"""Aggregate a regact benchmark tree into a model x task table.

Walks <bench_root>/<experiment>/<stamp>/<task>/ run dirs (the layout run_exp
writes; `bench_regact.sh` makes one experiment dir per task), reads each run's
config.json + experiment_state.json + final results, and prints a success-rate
pivot (rows = tasks, columns = models) plus a per-run detail table.

    python scripts/bench_aggregate.py experiments/bench_2026-08-08
    python scripts/bench_aggregate.py <root> --csv out.csv --json out.json

Stdlib only, read-only; safe to run on a live tree (partial runs show as n/a).
By default only the LATEST stamp of each experiment counts; --all-stamps keeps
every rerun.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

_ATTEMPT_RE = re.compile(r"attempt_\d+")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _count_lines(path: Path, *predicates: tuple[str, str]) -> dict[str, int]:
    """Count JSONL lines per label, where a (label, needle) predicate matches when
    the line's parsed dict has that needle as its ``type``/``event`` value."""
    counts = {label: 0 for label, _ in predicates}
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = record.get("type") or record.get("event")
        for label, needle in predicates:
            if kind == needle:
                counts[label] += 1
    return counts


def collect_runs(root: Path, *, all_stamps: bool) -> list[dict[str, Any]]:
    """One row per (experiment, stamp, task) run dir found under ``root``.

    A run dir is any directory that holds a ``config.json`` next to a ``logs/`` or
    ``workdir/``; they are found at any depth via rglob, so both a flat
    ``root/exp/stamp/task`` and a model-grouped ``root/model/exp/stamp/task``
    layout work. Without ``all_stamps``, only the latest stamp per
    (experiment, task) is kept - the newest rerun wins.
    """
    rows: list[dict[str, Any]] = []
    for config_path in sorted(root.rglob("config.json")):
        run_dir = config_path.parent
        if not ((run_dir / "logs").is_dir() or (run_dir / "workdir").is_dir()):
            continue  # a stray config.json, not a run dir
        config = _read_json(config_path)
        if config is None:
            continue
        # n_attempts_per_task>1 nests each run one level deeper, as <task>/attempt_N: the run dir
        # holds the artifacts, but the task name is its parent's.
        task_dir = run_dir.parent if _ATTEMPT_RE.fullmatch(run_dir.name) else run_dir
        attempt = int(run_dir.name.removeprefix("attempt_")) if task_dir is not run_dir else None
        stamp = task_dir.parent
        rows.append(
            _run_row(stamp.parent.name, stamp.name, task_dir.name, run_dir, config, attempt)
        )
    if all_stamps:
        return rows
    # Latest-stamp-wins collapses RERUNS of one cell. But a fan-out that launches one job per
    # (task, attempt) uses n_attempts_per_task=1, which writes no attempt_N dir - so its attempts are
    # distinguished ONLY by stamp and this would silently keep one and discard the rest. Nothing on
    # disk separates "a rerun" from "another attempt", so when there is no attempt marker the stamp
    # IS the identity: count them all. Over-counting a rerun is visible as a raised n; dropping
    # attempts is not visible at all.
    latest: dict[tuple[str, str, Any], dict[str, Any]] = {}
    for row in rows:
        identity = row["attempt"] if row["attempt"] is not None else row["stamp"]
        key = (row["experiment"], row["task"], identity)
        if key not in latest or row["stamp"] > latest[key]["stamp"]:
            latest[key] = row
    kept = list(latest.values())
    _warn_ambiguous_stamps(kept)
    return kept


def _warn_ambiguous_stamps(rows: list[dict[str, Any]]) -> None:
    """Name the cells whose repeats cannot be classified, instead of quietly picking a meaning."""
    seen: dict[tuple[str, str], int] = collections.Counter()
    for row in rows:
        if row["attempt"] is None:
            seen[(row["experiment"], row["task"])] += 1
    repeated = sorted(cell for cell, n in seen.items() if n > 1)
    if not repeated:
        return
    print(
        f"warning: {len(repeated)} cell(s) have several timestamped runs and no attempt_N marker, "
        "so a repeat cannot be told from an attempt; ALL are counted. If these are reruns rather "
        "than attempts, remove the superseded directories.",
        file=sys.stderr,
    )
    for experiment, task in repeated[:5]:
        print(f"  ambiguous: {experiment}/{task} x{seen[(experiment, task)]}", file=sys.stderr)


_RANK = {"stub": 0, "trivial": 1, "reasoned": 2}


def _classify_act(act: ast.FunctionDef) -> str:
    """``reasoned`` vs ``trivial`` for a found ``act`` body (the leaf judgement)."""
    body = [s for s in act.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    # Any control flow or bound state is reasoning about the situation.
    if any(isinstance(n, (ast.If, ast.For, ast.While, ast.Assign, ast.AugAssign)) for n in body):
        return "reasoned"
    # A single return: reasoned only if it reads the obs beyond the action list.
    reads = {
        n.attr
        for n in ast.walk(act)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "obs"
    }
    return "reasoned" if reads - {"available_actions"} else "trivial"


def _resolve_local_module(module: str | None, level: int, current_dir: Path, root: Path) -> Path | None:
    """The workdir file an import names, or None if it isn't a local module.

    Absolute ``code_library.smart_controller`` -> ``<root>/code_library/smart_controller.py``;
    relative ``.base_controller`` -> alongside the importing file.
    """
    if level:
        base = current_dir
        for _ in range(level - 1):
            base = base.parent
        parts = module.split(".") if module else []
        return base.joinpath(*parts).with_suffix(".py") if parts else None
    if not module:
        return None
    return root.joinpath(*module.split(".")).with_suffix(".py")


def _classify_source(source: str, current_dir: Path, root: Path, seen: set[Path], depth: int) -> str:
    """Classify the ``act`` a controller ultimately runs, following subclassing into
    agent-written local modules when ``solution.py`` is only a thin subclass."""
    if "raise NotImplementedError" in source:
        return "stub"
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "unparsable"
    act = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "act"), None
    )
    if act is not None:
        return _classify_act(act)
    if depth <= 0:
        return "unparsable"
    # No local ``act``: the real controller lives in an agent-written module that
    # solution.py either subclasses or instantiates in get_controller. code_library/
    # is seeded empty, so every locally-importable module is the agent's - follow
    # each into the workdir and take the strongest ``act`` found (scaffold files have
    # no ``act`` and contribute nothing).
    best: str | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        path = _resolve_local_module(node.module, node.level, current_dir, root)
        if path is None or path in seen or not path.is_file():
            continue
        seen.add(path)
        found = _classify_source(path.read_text(encoding="utf-8"), path.parent, root, seen, depth - 1)
        if best is None or _RANK.get(found, -1) > _RANK.get(best, -1):
            best = found
    return best if best is not None else "unparsable"


def _classify_controller(solution_path: Path) -> str:
    """Classify the submitted controller by what its ``act`` actually does.

    The benchmark's core question is whether the agent writes a *real* policy, so
    "not the stub" is not enough - a Phase-0 baseline that returns a constant or
    ``available_actions[0]`` is a genuine edit but reads nothing from the obs.

    - ``stub``: untouched scaffold (``raise NotImplementedError``);
    - ``trivial``: a single return that ignores the observation's content
      (a constant, or ``available_actions[i]`` / a random pick over them);
    - ``reasoned``: anything that inspects the obs (``obs.frame``/``obs.info``,
      branching, loops, kept state) to choose the action;
    - ``missing`` / ``unparsable`` when the file is absent or not valid Python.

    When ``solution.py`` is a thin subclass of an agent-written ``code_library/``
    module (a common pattern for the more sophisticated agents), the real ``act``
    lives in that module - so the base-class import is followed to classify it,
    rather than reporting ``unparsable``. The follow needs the sibling modules
    present (the full workdir); a lone solution.py falls back to ``unparsable``.
    """
    try:
        source = solution_path.read_text(encoding="utf-8")
    except OSError:
        return "missing"
    root = solution_path.parent
    return _classify_source(source, root, root, {solution_path}, depth=5)


_SOLVE_THRESHOLD = 0.6


def _episodes_shortfall(row: dict[str, Any], configured: int | None) -> int | None:
    """How many episodes the reported cell is short of what the run asked for, if any.

    A final evaluation can end early, and the count it actually managed is what its error bar
    is built from - a 1.00 over 3 episodes is a 12% coincidence when the true rate is 0.5, not
    a solve. Measured in this round: 158 finals ran the configured 10, 15 ran fewer, and 8 ran
    ZERO while still recording success_rate 0.0 - a false zero indistinguishable from failure.
    """
    ran = row.get("n_episodes")
    if ran is None or configured is None or ran >= configured:
        return None
    return configured - ran


def _primary_score(aggregate: dict[str, Any]) -> float | None:
    """A cell's headline score, problem-agnostic: MiniGrid's ``success_rate``, or - for ARC, which
    renamed that away - ``mean_levels_completion_rate`` (graded mean fraction of levels cleared), so
    the pivot and outcome classification stay meaningful across both problem families.

    An evaluation that ran ZERO episodes measured nothing, and the 0.0 it records is a default
    rather than a result - reporting it as a score puts a false zero in the table, where it reads
    exactly like a model that tried and failed."""
    if aggregate.get("n_episodes") == 0:
        return None
    score = aggregate.get("success_rate")
    return score if score is not None else aggregate.get("mean_levels_completion_rate")


def _classify_outcome(
    success_rate: float | None,
    exit_reason: str | None,
    controller_crashed: bool = False,
) -> str:
    """Whether a cell's score is a trustworthy capability signal.

    The 0.0s are not equal: a model that ended cleanly and genuinely failed is real
    data, but one the harness killed mid-run (the empty_response reasoning-only-turn
    bug -> exit ``agent_api``) or that ran out of walltime tells us nothing about
    capability. Separating them stops the reasoning-model bias (the over-thinkers are
    the ones killed) from being misread as incapacity.

    - ``solve``: scored at/above the solve threshold - reliable;
    - ``genuine-fail``: a clean ``agent_exit`` with a sub-threshold score - reliable;
    - ``harness-killed``: exited ``agent_api`` (the pre-nudge empty_response wall) - UNRELIABLE;
    - ``walltime``: hit the job walltime before finishing - UNRELIABLE;
    - ``no-final``: no scored result (still running, or killed before teardown);
    - ``controller-crashed``: EVERY episode raised, so the controller never ran. Distinct
      from a 0.0, which the table otherwise renders identically - one is a policy that
      loses, the other is code that does not execute, and only the first is about ability;
    - the raw exit reason for any other terminal state.
    """
    if controller_crashed:
        return "controller-crashed"
    if success_rate is not None and success_rate >= _SOLVE_THRESHOLD:
        return "solve"
    if exit_reason == "agent_api":
        return "harness-killed"
    if exit_reason == "walltime_limit":
        return "walltime"
    if exit_reason == "agent_exit":
        return "genuine-fail" if success_rate is not None else "no-final"
    if exit_reason:  # any other terminal state (loop_crash, interrupted, ...)
        return str(exit_reason)
    return "no-final"  # no exit reason recorded: still running / killed before teardown


def _run_row(
    experiment: str,
    stamp: str,
    task: str,
    task_dir: Path,
    config: dict[str, Any],
    attempt: int | None = None,
) -> dict[str, Any]:
    agent = config.get("agent", {})
    model = str(agent.get("model") or "?").removeprefix("openai/")
    state = _read_json(task_dir / "logs" / "experiment_state.json") or {}
    final = _read_json(task_dir / "workdir" / "submissions" / "final" / "results.json") or {}
    aggregate = final.get("aggregate", {})
    transcript = _count_lines(
        task_dir / "logs" / "transcript.jsonl",
        ("tool_calls", "ToolCall"),
        ("turns", "UserMessage"),
    )
    events = _count_lines(
        task_dir / "logs" / "events.jsonl",
        ("agent_errors", "agent_error"),
        ("error_retries", "agent_error_retry"),
    )
    return {
        "experiment": experiment,
        "stamp": stamp,
        "task": task,
        "attempt": attempt,
        "agent": agent.get("name", "?"),
        "model": model,
        "seed": (config.get("problem") or {}).get("seed"),
        "controller": _classify_controller(task_dir / "workdir" / "solution.py"),
        "outcome": _classify_outcome(
            _primary_score(aggregate),
            state.get("exit_reason"),
            controller_crashed=bool(aggregate.get("n_errors"))
            and not aggregate.get("n_episodes"),
        ),
        "success_rate": _primary_score(aggregate),  # MiniGrid success_rate or ARC completion rate
        "tail_mean": _tail_mean(task_dir)[0],  # the same controller's neighbourhood, for stability
        "episodes_asked": (config.get("controller") or {}).get("n_episodes"),
        "n_episodes": aggregate.get("n_episodes"),  # episodes SCORED; errored ones are excluded
        "n_errors": aggregate.get("n_errors"),
        "mean_levels_completed": aggregate.get("mean_levels_completed"),
        "exit_reason": state.get("exit_reason"),
        "last_error_category": state.get("last_error_category"),
        "submissions": state.get("submission_count"),
        "duration_s": state.get("duration_s"),
        "env_moves": state.get("env_moves"),
        "tool_calls": transcript["tool_calls"],
        "turns": transcript["turns"],
        "agent_errors": events["agent_errors"],
        "error_retries": events["error_retries"],
    }


_TAIL_SUBMISSIONS = 20


def _tail_mean(task_dir: Path, k: int = _TAIL_SUBMISSIONS) -> tuple[float | None, int]:
    """Mean score of the last ``k`` numbered submissions, and how many were averaged.

    The reported cell is one evaluation of one controller, and at ``n_episodes=10`` a
    success_rate carries a standard error near 0.15 - so a cell can land high or low by luck.
    Averaging the run's own recent submissions gives a second, independent reading of the same
    controller's neighbourhood; where the two agree the cell is trustworthy, and where they
    diverge the reader can see it rather than having to remember which cells were lucky.
    """
    subs = task_dir / "workdir" / "submissions"
    if not subs.is_dir():
        return None, 0
    scored: list[tuple[int, float]] = []
    for entry in subs.iterdir():
        if entry.name == "final" or not entry.is_dir():
            continue
        digits = "".join(ch for ch in entry.name if ch.isdigit())
        score = _primary_score((_read_json(entry / "results.json") or {}).get("aggregate", {}))
        if digits and score is not None:
            scored.append((int(digits), score))
    if not scored:
        return None, 0
    tail = [score for _, score in sorted(scored)[-k:]]
    return sum(tail) / len(tail), len(tail)


def _walltime_bucket(row: dict[str, Any]) -> str | None:
    """Which kind of walltime cut this run was, or ``None`` if it did not hit the walltime.

    One ``exit_reason`` covers two opposite outcomes. A run capped after genuinely iterating
    carries a score the model earned; one that never submitted was scored only by
    ``FinalizeControllerHook``, so the number is ours, not the model's - measured 0.003 mean
    against 0.042 for the iterating group. Reporting them in one column credits a model for a
    measurement we performed on its behalf. ``starved`` takes everything with no usable score,
    including the few that submitted something unscoreable, so the three buckets partition
    every walltime run.
    """
    if row.get("exit_reason") != "walltime_limit":
        return None
    if row.get("success_rate") is None:
        return "starved"
    return "capped" if (row.get("submissions") or 0) > 0 else "teardown"


def coverage_markdown(rows: list[dict[str, Any]]) -> str:
    """Per-model coverage, because incomplete columns here are NOT missing at random.

    A slow serve finishes the easy tasks and times out on the hard ones, so the cells a model does
    have are biased toward the ones every model passes. Averaging such a column reports the bias as
    capability. This table exists so that skew is visible next to the pivots rather than inferred.
    """
    tasks = {row["task"] for row in rows}
    models = sorted({row["model"] for row in rows})
    lines = [
        "| model | tasks | attempts/task | shape | solved | budget-capped "
        "| wt-capped | wt-teardown | wt-starved | missing |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in models:
        mine = [row for row in rows if row["model"] == model]
        covered = {row["task"] for row in mine}
        per_task = collections.Counter(row["task"] for row in mine)
        # Count every benchmark task, missing ones as 0: that is what exposes raggedness.
        spread = sorted(per_task.get(task, 0) for task in tasks)
        low, high = spread[0], spread[-1]
        # A uniform reduced n is a clean smaller sample; a ragged n is a BIASED subset, because the
        # cells that survive are the fast ones. Averaging them the same way hides that difference.
        shape = "uniform" if low == high else "RAGGED"
        attempts = str(low) if low == high else f"{low}-{high}"
        reasons = collections.Counter(row.get("exit_reason") for row in mine)
        walltime = collections.Counter(filter(None, (_walltime_bucket(row) for row in mine)))
        lines.append(
            f"| {model} | {len(covered)}/{len(tasks)} | {attempts} | {shape} "
            f"| {reasons.get('solved', 0)} | {reasons.get('tool_call_limit', 0)} "
            f"| {walltime['capped']} | {walltime['teardown']} | {walltime['starved']} "
            f"| {len(tasks) - len(covered)} |"
        )
    return "\n".join(lines)


def stability_markdown(rows: list[dict[str, Any]], sigmas: float = 2.0) -> str:
    """Cells whose reported score disagrees with the run's own recent submissions.

    Both numbers describe the same controller, so a large gap means the reported cell caught a
    lucky or unlucky evaluation rather than a real difference in ability. Measured example: a
    480B FourRooms cell read 1.00 - a solve - while its own last twenty submissions averaged
    0.51, which is 3.1 standard errors out and would have been reported as "solves FourRooms".
    """
    flagged = []
    for row in rows:
        final = row.get("success_rate")
        if final is None:
            continue
        short = _episodes_shortfall(row, row.get("episodes_asked"))
        tail = row.get("tail_mean")
        deviation = 0.0
        if tail is not None:
            spread = (tail * (1.0 - tail) / (row.get("n_episodes") or 10)) ** 0.5
            deviation = abs(final - tail) / spread if spread > 0 else 0.0
        if deviation >= sigmas or short:
            flagged.append((deviation, short, row, final, tail))
    if not flagged:
        return "All reported cells agree with their run's recent submissions.\n"
    lines = [
        "| model | task | reported | if errors counted | mean(last submissions) "
        "| sigmas out | episodes scored |",
        "|---|---|---|---|---|---|---|",
    ]
    for deviation, short, row, final, tail in sorted(flagged, key=lambda item: -item[0]):
        episodes = f"{row.get('n_episodes')} of {row.get('episodes_asked')}" if short else ""
        scored, errors = row.get("n_episodes") or 0, row.get("n_errors") or 0
        attempted = scored + errors
        honest = f"{final * scored / attempted:.2f}" if errors and attempted else ""
        lines.append(
            f"| {row['model']} | {row['task']} | {final:.2f} | {honest} "
            f"| {'-' if tail is None else f'{tail:.2f}'} "
            f"| {deviation:.1f} | {episodes} |"
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _cell(values: list[Any]) -> str:
    """One cell from every attempt of a (task, model): the point of running N attempts is to
    aggregate them, so numbers average and labels collapse to the majority (``value n/N`` when
    the attempts disagree)."""
    present = [v for v in values if v is not None]
    if not present:
        return "-"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        mean = sum(present) / len(present)
        return _fmt(mean) if len(present) == 1 else f"{mean:.2f} ({len(present)})"
    top, count = collections.Counter(str(v) for v in present).most_common(1)[0]
    return top if count == len(present) else f"{top} {count}/{len(present)}"


def _pivot(rows: list[dict[str, Any]], field: str) -> str:
    """One row per task, one column per model; each cell aggregates that pair's attempts."""
    models = sorted({row["model"] for row in rows})
    tasks = sorted({row["task"] for row in rows})
    grouped: dict[tuple[str, str], list[Any]] = {}
    for row in rows:
        grouped.setdefault((row["task"], row["model"]), []).append(row.get(field))
    lines = ["| task | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for task in tasks:
        cells = [_cell(grouped.get((task, m), [])) for m in models]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pivot_markdown(rows: list[dict[str, Any]]) -> str:
    """Success-rate pivot: one row per task, one column per model."""
    return _pivot(rows, "success_rate")


def controller_pivot_markdown(rows: list[dict[str, Any]]) -> str:
    """Controller-state pivot (stub/trivial/reasoned) - the behavioral signal that,
    unlike success, is not confounded by walltime."""
    return _pivot(rows, "controller")


def outcome_pivot_markdown(rows: list[dict[str, Any]]) -> str:
    """Outcome pivot: whether each cell's score is trustworthy (solve/genuine-fail)
    or must be discounted (harness-killed/walltime/no-final). Separates the
    empty_response harness bias from real incapacity - see :func:`_classify_outcome`."""
    return _pivot(rows, "outcome")


_DETAIL_COLUMNS = [
    "task", "model", "seed", "controller", "outcome", "success_rate", "exit_reason",
    "submissions", "tool_calls", "turns", "error_retries", "duration_s", "env_moves", "stamp",
]


def detail_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(_DETAIL_COLUMNS) + " |"]
    lines.append("|" + "---|" * len(_DETAIL_COLUMNS))
    for row in sorted(rows, key=lambda r: (r["model"], r["task"], str(r["seed"]))):
        lines.append("| " + " | ".join(_fmt(row[c]) for c in _DETAIL_COLUMNS) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bench_root", type=Path)
    parser.add_argument("--csv", type=Path, help="also write every run row as CSV")
    parser.add_argument("--json", type=Path, help="also write every run row as JSON")
    parser.add_argument("--all-stamps", action="store_true", help="keep reruns, not just the latest stamp")
    args = parser.parse_args(argv)

    if not args.bench_root.is_dir():
        print(f"not a directory: {args.bench_root}", file=sys.stderr)
        return 2
    rows = collect_runs(args.bench_root, all_stamps=args.all_stamps)
    if not rows:
        print(f"no runs found under {args.bench_root}", file=sys.stderr)
        return 1

    print(f"# Benchmark aggregate: {args.bench_root} ({len(rows)} runs)\n")
    counts = collections.Counter(row["controller"] for row in rows)
    print("Controller states: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + "\n")
    outcomes = collections.Counter(row["outcome"] for row in rows)
    print("Outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())) + "\n")
    print("## Coverage - which cells exist, and why some do not\n")
    print(
        "Incomplete columns are NOT missing at random: a slow serve completes the EASY tasks and "
        "times out on the hard ones, so a model's visible cells skew toward the tasks every model "
        "passes. Read per-task cells; never average a column into a per-model score.\n"
    )
    print(coverage_markdown(rows))
    print("\n## Stability - cells that disagree with their own run\n")
    print(
        "A cell is ONE evaluation of one controller; at n_episodes=10 its standard error is near "
        "0.15, so a cell can land high or low by luck. These rows report a score far from the mean "
        "of that same run's recent submissions - read them as uncertain, not as achievement.\n"
    )
    print(stability_markdown(rows))
    print("\n## Outcome - is the score trustworthy? (task x model)\n")
    print("`solve`/`genuine-fail` are reliable; `harness-killed` (empty_response) and "
          "`walltime` must be discounted / re-run.\n")
    print(outcome_pivot_markdown(rows))
    print("\n## Controller written (task x model)\n")
    print(controller_pivot_markdown(rows))
    print("\n## Final success rate (task x model)\n")
    print(pivot_markdown(rows))
    print("\n## Runs\n")
    print(detail_markdown(rows))

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\ncsv: {args.csv}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"json: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
