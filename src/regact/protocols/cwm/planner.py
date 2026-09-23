"""Bounded trusted BFS over predictions; imagined states never enter real experience."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any

from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.store import canonical, digest
from regact.protocols.cwm.validation import check_observation
from regact.protocols.cwm.worker import Worker, WorkerError


def plan(
    worker: Worker,
    initial: dict[str, Any],
    known: set[str],
    config: CwmConfig,
    actions_for: Callable[[dict[str, Any]], Iterable[Any]],
    *,
    goal_worker: Worker | None = None,
) -> dict[str, Any]:
    cfg = config.planner
    goal_worker = goal_worker or worker
    started = time.monotonic()
    state = worker.call("parse", obs=initial)
    nodes = [
        {"state": state, "obs": initial, "parent": None, "action": None, "depth": 0, "novel": False}
    ]
    visited = {(canonical(state), False)}
    stored_bytes = len(canonical(nodes[0]).encode())
    cursor = 0
    calls = 0
    best: tuple[float, int, int, bool] | None = None
    reason = "frontier_exhausted"
    depth_limited = False
    halted = False
    try:
        while cursor < len(nodes) and not halted:
            index = cursor
            node = nodes[cursor]
            cursor += 1
            if node["obs"]["is_done"]:
                continue
            if node["depth"] >= cfg.max_depth_per_planner_call:
                depth_limited = True
                continue
            for action in actions_for(node["obs"]):
                if time.monotonic() >= worker.deadline:
                    reason = "max_seconds_per_planner_call"
                    halted = True
                    break
                if calls >= cfg.max_cwm_calls_per_planner_call:
                    reason = "max_cwm_calls_per_planner_call"
                    halted = True
                    break
                if len(nodes) >= cfg.max_nodes_per_planner_call:
                    reason = "max_nodes_per_planner_call"
                    halted = True
                    break
                next_state = worker.call("step", state=node["state"], action=action)
                calls += 1
                obs = check_observation(worker.call("render", state=next_state))
                novelty = node["novel"] or digest(obs) not in known
                key = (canonical(next_state), novelty)
                if key in visited:
                    continue
                child = {
                    "state": next_state,
                    "obs": obs,
                    "parent": index,
                    "action": action,
                    "depth": node["depth"] + 1,
                    "novel": novelty,
                }
                stored_bytes += len(canonical(child).encode())
                # Bound trusted queue payload too, not just the submitted-code worker.
                if stored_bytes > config.execution.max_memory_mb * 1024 * 1024 // 4:
                    reason = "search_memory_limit"
                    halted = True
                    break
                visited.add(key)
                nodes.append(child)
                goal = goal_worker.call("goal", state=next_state)
                rank = (goal["utility"], -child["depth"])
                if novelty and (best is None or rank > best[:2]):
                    best = (rank[0], rank[1], len(nodes) - 1, goal["achieved"])
                    if rank[0] == 1:
                        # BFS has reached the utility upper bound at the shortest eligible depth.
                        reason = "utility_upper_bound"
                        halted = True
                        break
    except WorkerError as exc:
        if exc.kind != "operation_timeout":
            raise
        reason = "max_seconds_per_planner_call"
        halted = True
    if not halted and depth_limited:
        reason = "max_depth_per_planner_call"
    sequence: list[Any] = []
    predictions: list[str] = []
    if best is not None:
        index = best[2]
        while nodes[index]["parent"] is not None:
            sequence.append(nodes[index]["action"])
            predictions.append(digest(nodes[index]["obs"]))
            index = nodes[index]["parent"]
        sequence.reverse()
        predictions.reverse()
    return {
        "candidate_found": best is not None,
        "actions": sequence,
        "predicted_observation_hashes": predictions,
        "achieved": best[3] if best else False,
        "utility": best[0] if best else None,
        "optimality_proven": reason in ("utility_upper_bound", "frontier_exhausted")
        and best is not None,
        "search_stop_reason": reason,
        "cwm_calls": calls,
        "nodes": len(nodes),
        "elapsed_seconds": time.monotonic() - started,
        "real_actions": 0,
    }
