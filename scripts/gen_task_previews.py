"""Pre-render task preview PNGs into src/regact/viz/static/icons_tasks/ (OFFLINE, not at viz time).

The viz loads these static files directly, so it never builds an env while serving. Re-run when new
tasks appear:  PYTHONPATH=src python scripts/gen_task_previews.py

Filename = the task name sanitized to [A-Za-z0-9._-] (MiniGrid/ARC names already are), matching the
frontend's taskIconSrc.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "src" / "regact" / "viz" / "static" / "icons_tasks"


def _safe(task: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", task)


def _collect_tasks() -> list[str]:
    tasks: set[str] = set()
    # every task shown in the local experiments (its game path's last component)
    try:
        from regact.viz.reader import list_games

        exp = _ROOT / "experiments"
        if exp.is_dir():
            tasks.update(g.split("/")[-1] for g in list_games(str(exp)))
    except Exception as exc:  # best-effort
        print(f"warn: experiments scan failed: {exc}")
    # the full MiniGrid catalogue (so any suite is covered)
    try:
        from regact.problems.minigrid.tasks import ALL_MINIGRID_TASKS

        tasks.update(ALL_MINIGRID_TASKS)
    except Exception as exc:
        print(f"warn: minigrid catalogue unavailable: {exc}")
    # ARC games present locally (one dir per game id)
    envdir = _ROOT / "environnement"
    if envdir.is_dir():
        tasks.update(d.name for d in envdir.iterdir() if d.is_dir())
    return sorted(t for t in tasks if t)


def main() -> int:
    from regact.viz.task_preview import render_task_png

    _OUT.mkdir(parents=True, exist_ok=True)
    tasks = _collect_tasks()
    print(f"rendering {len(tasks)} task previews -> {_OUT.relative_to(_ROOT)}")
    ok = skipped = 0
    for task in tasks:
        try:
            (_OUT / (_safe(task) + ".png")).write_bytes(render_task_png(task))
            ok += 1
        except Exception as exc:
            skipped += 1
            print(f"  SKIP {task}: {type(exc).__name__}: {str(exc)[:80]}")
    print(f"done: {ok} rendered, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
