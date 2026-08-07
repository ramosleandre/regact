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
    """One row per (experiment, stamp, task) run dir found under ``root``."""
    rows: list[dict[str, Any]] = []
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        stamps = sorted(
            p for p in exp_dir.iterdir() if p.is_dir() and not p.is_symlink()
        )
        if not stamps:
            continue
        if not all_stamps:
            stamps = stamps[-1:]
        for stamp in stamps:
            for task_dir in sorted(p for p in stamp.iterdir() if p.is_dir()):
                config = _read_json(task_dir / "config.json")
                if config is None:
                    continue
                rows.append(_run_row(exp_dir.name, stamp.name, task_dir, config))
    return rows


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


def pivot_markdown(rows: list[dict[str, Any]]) -> str:
    """Success-rate pivot: one row per task, one column per model."""
    models = sorted({row["model"] for row in rows})
    tasks = sorted({row["task"] for row in rows})
    by_key = {(row["task"], row["model"]): row for row in rows}
    lines = ["| task | " + " | ".join(models) + " |"]
    lines.append("|---|" + "---|" * len(models))
    for task in tasks:
        cells = []
        for model in models:
            row = by_key.get((task, model))
            cells.append(_fmt(row["success_rate"]) if row else "-")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


_DETAIL_COLUMNS = [
    "task", "model", "seed", "success_rate", "exit_reason", "submissions",
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
    print("## Final success rate (task x model)\n")
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
