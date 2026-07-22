"""ARC-AGI-3 task catalog, discovered from the local game data.

Each game lives on disk as ``<environments_dir>/<game>/<hash>/metadata.json`` (+ a
sibling ``<game>.py``). We read the metadata directly — no arc library needed — so
the task list, titles, tags and level counts are available even where ``arc_agi``
is not installed. ``make_env`` (which DOES need the library) only runs at eval time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArcAgiTask:
    """One ARC-AGI-3 game."""

    key: str  # framework-side id (short, e.g. "ls20")
    game_id: str  # id passed to Arcade.make() (short form is accepted)
    title: str
    tags: tuple[str, ...] = ()  # "keyboard", "click", or both
    win_levels: int | None = None  # levels to complete for a WIN
    baseline_actions: tuple[int, ...] | None = None  # human benchmark per level


def discover_tasks(environments_dir: str) -> dict[str, ArcAgiTask]:
    """Build the catalog by scanning ``<environments_dir>/*/*/metadata.json``."""
    root = Path(environments_dir)
    tasks: dict[str, ArcAgiTask] = {}
    for meta_path in sorted(root.glob("*/*/metadata.json")):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        full_id = str(data.get("game_id") or meta_path.parent.parent.name)
        key = full_id.split("-", 1)[0]
        baseline = data.get("baseline_actions") or None
        tasks[key] = ArcAgiTask(
            key=key,
            game_id=key,
            title=str(data.get("title", key)),
            tags=tuple(data.get("tags", ())),
            win_levels=len(baseline) if baseline else None,
            baseline_actions=tuple(baseline) if baseline else None,
        )
    return tasks


def discover_tasks_from_arcade(arcade: object) -> dict[str, ArcAgiTask]:
    """Build the catalog from a live ``Arcade.available_environments`` listing.

    The online/gateway path has no local metadata files: the games are served and
    enumerated by the arcade itself. Each ``env_info`` exposes ``game_id`` (full,
    e.g. ``"ls20-9607627b"``), ``title``, and ``baseline_actions`` (the per-level
    human benchmark). Keyed by the short id so framework-side ids match the offline
    catalog and ``--games ls20`` filters work identically in both modes.
    """
    tasks: dict[str, ArcAgiTask] = {}
    for info in getattr(arcade, "available_environments", ()):
        full_id = str(getattr(info, "game_id", "") or "")
        if not full_id:
            continue
        key = full_id.split("-", 1)[0]
        baseline = getattr(info, "baseline_actions", None) or None
        tasks[key] = ArcAgiTask(
            key=key,
            game_id=full_id,  # online: pass the full served id to Arcade.make()
            title=str(getattr(info, "title", key) or key),
            tags=tuple(getattr(info, "level_tags", None) or ()),
            win_levels=len(baseline) if baseline else None,
            baseline_actions=tuple(baseline) if baseline else None,
        )
    return tasks
