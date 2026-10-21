"""Build ``problems/grid_games/data/sokoban_packs.json`` from a Boxoban checkout.

Usage:
    python scripts/grid_games/build_sokoban_packs.py <boxoban-levels dir> [--packs 8] [--seed 0]

Each pack holds levels of increasing difficulty: 2 from ``unfiltered``, 3 from ``medium``,
2 from ``hard``, ordered by optimal length. Every level is solved move-optimally (A* with an
admissible box-to-goal matching heuristic) and its plan is replayed through regact's engine
before it is kept; unsolved levels within the time limit are skipped.
Boxoban levels: https://github.com/google-deepmind/boxoban-levels (Apache-2.0).
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from regact.problems.grid_games.base import GridGameEnv
from regact.problems.grid_games.sokoban import MOVES, SokobanGame, parse_level

OUT = Path(__file__).resolve().parents[2] / "src/regact/problems/grid_games/data/sokoban_packs.json"
MIX = (("unfiltered", 2), ("medium", 3), ("hard", 2))
TIME_LIMIT = 20.0


def read_levels(path: Path) -> list[tuple[str, ...]]:
    levels, current = [], []
    for line in path.read_text().splitlines():
        if line.startswith(";"):
            if current:
                levels.append(tuple(current))
            current = []
        elif line.strip():
            current.append(line)
    if current:
        levels.append(tuple(current))
    return levels


def solve(rows: tuple[str, ...], time_limit: float) -> list[int] | None:
    """A move-optimal plan, or None when the time limit is reached."""
    level = parse_level(rows)
    cells, queue = {level["player"]}, deque([level["player"]])
    while queue:
        r, c = queue.popleft()
        for dr, dc in MOVES.values():
            n = (r + dr, c + dc)
            if n not in level["walls"] and n not in cells:
                cells.add(n)
                queue.append(n)
    goals = sorted(level["goals"])
    pushes = {}  # goal -> {cell: pushes needed to bring a box from cell to goal}
    for g in goals:
        dist, queue = {g: 0}, deque([g])
        while queue:
            b = queue.popleft()
            for dr, dc in MOVES.values():
                before, player = (b[0] - dr, b[1] - dc), (b[0] - 2 * dr, b[1] - 2 * dc)
                if before in cells and player in cells and before not in dist:
                    dist[before] = dist[b] + 1
                    queue.append(before)
        pushes[g] = dist
    live = {c for c in cells if any(c in pushes[g] for g in goals)}
    inf = 10**6

    def h(boxes: tuple) -> int:
        best = inf
        for order in itertools.permutations(goals):
            total = 0
            for box, goal in zip(boxes, order, strict=True):
                total += pushes[goal].get(box, inf)
                if total >= best:
                    break
            best = min(best, total)
        return best

    start = (level["player"], tuple(sorted(level["boxes"])))
    target = tuple(goals)
    cost, parent = {start: 0}, {start: None}
    frontier = [(h(start[1]), 0, start)]
    t0, expanded = time.time(), 0
    while frontier:
        f, g, state = heapq.heappop(frontier)
        if g > cost.get(state, inf):
            continue
        player, boxes = state
        if boxes == target:
            plan = []
            while parent[state] is not None:
                state, action = parent[state]
                plan.append(action)
            return plan[::-1]
        expanded += 1
        if expanded % 20000 == 0 and time.time() - t0 > time_limit:
            return None
        box_set = set(boxes)
        for action, (dr, dc) in MOVES.items():
            n = (player[0] + dr, player[1] + dc)
            if n not in cells:
                continue
            if n in box_set:
                beyond = (n[0] + dr, n[1] + dc)
                if beyond not in live or beyond in box_set:
                    continue
                new_boxes = tuple(sorted((box_set - {n}) | {beyond}))
            else:
                new_boxes = boxes
            nxt = (n, new_boxes)
            if g + 1 < cost.get(nxt, inf):
                cost[nxt] = g + 1
                parent[nxt] = (state, action)
                estimate = h(new_boxes) if new_boxes is not boxes else f - g
                if estimate < inf:
                    heapq.heappush(frontier, (g + 1 + estimate, g + 1, nxt))
    return None


def solves(rows: tuple[str, ...], plan: list[int]) -> bool:
    env = GridGameEnv(SokobanGame((rows,)), n_levels=1, actions=(1, 2, 3, 4))
    env.reset()
    done = False
    for i, action in enumerate(plan):
        _, _, done, _, _ = env.step(action)
        if done and i != len(plan) - 1:
            return False
    return done


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("boxoban")
    args.add_argument("--packs", type=int, default=8)
    args.add_argument("--seed", type=int, default=0)
    opts = args.parse_args()
    root, rng = Path(opts.boxoban), random.Random(opts.seed)
    files = {
        name: sorted(p for p in (root / name).rglob("*.txt") if "test" not in p.parts)
        for name, _ in MIX
    }
    used: set[tuple[str, ...]] = set()
    packs = []
    for p in range(opts.packs):
        levels = []
        for name, count in MIX:
            found = 0
            while found < count:
                path = rng.choice(files[name])
                index = rng.randrange(len(read_levels(path)))
                rows = read_levels(path)[index]
                if rows in used:
                    continue
                plan = solve(rows, TIME_LIMIT)
                if plan is None or not solves(rows, plan):
                    continue
                used.add(rows)
                found += 1
                source = f"{path.relative_to(root)}#{index}"
                levels.append(
                    {"rows": list(rows), "optimal": len(plan), "plan": plan, "source": source}
                )
                print(f"pack {p + 1}: {source} optimal {len(plan)}", flush=True)
        levels.sort(key=lambda level: level["optimal"])
        packs.append({"key": f"sokoban_p{p + 1:02d}", "levels": levels})
    OUT.write_text(
        json.dumps(
            {
                "source": "https://github.com/google-deepmind/boxoban-levels (Apache-2.0)",
                "built_by": "scripts/grid_games/build_sokoban_packs.py",
                "seed": opts.seed,
                "packs": packs,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
