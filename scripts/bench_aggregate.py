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
import sys
from pathlib import Path
from typing import Any


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
        task_dir = config_path.parent
        if not ((task_dir / "logs").is_dir() or (task_dir / "workdir").is_dir()):
            continue  # a stray config.json, not a run dir
        config = _read_json(config_path)
        if config is None:
            continue
        stamp = task_dir.parent
        rows.append(_run_row(stamp.parent.name, stamp.name, task_dir, config))
    if all_stamps:
        return rows
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["experiment"], row["task"])
        if key not in latest or row["stamp"] > latest[key]["stamp"]:
            latest[key] = row
    return list(latest.values())


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
    # No local ``act``: follow each class's base into the local module that defines it.
    imports = {
        (alias.asname or alias.name): (node.module, node.level)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    best: str | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in (b for b in node.bases if isinstance(b, ast.Name)):
            info = imports.get(base.id)
            if info is None:
                continue
            path = _resolve_local_module(info[0], info[1], current_dir, root)
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


def _run_row(experiment: str, stamp: str, task_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
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
        "task": task_dir.name,
        "agent": agent.get("name", "?"),
        "model": model,
        "seed": (config.get("problem") or {}).get("seed"),
        "controller": _classify_controller(task_dir / "workdir" / "solution.py"),
        "success_rate": aggregate.get("success_rate"),
        "n_episodes": aggregate.get("n_episodes"),
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


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _pivot(rows: list[dict[str, Any]], field: str) -> str:
    """One row per task, one column per model, cells = ``field`` of that run."""
    models = sorted({row["model"] for row in rows})
    tasks = sorted({row["task"] for row in rows})
    by_key = {(row["task"], row["model"]): row for row in rows}
    lines = ["| task | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for task in tasks:
        cells = [_fmt(by_key[(task, m)][field]) if (task, m) in by_key else "-" for m in models]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pivot_markdown(rows: list[dict[str, Any]]) -> str:
    """Success-rate pivot: one row per task, one column per model."""
    return _pivot(rows, "success_rate")


def controller_pivot_markdown(rows: list[dict[str, Any]]) -> str:
    """Controller-state pivot (stub/trivial/reasoned) - the behavioral signal that,
    unlike success, is not confounded by walltime."""
    return _pivot(rows, "controller")


_DETAIL_COLUMNS = [
    "task", "model", "seed", "controller", "success_rate", "exit_reason", "submissions",
    "tool_calls", "turns", "error_retries", "duration_s", "env_moves", "stamp",
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
    print("## Controller written (task x model)\n")
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
