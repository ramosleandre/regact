"""The Replay tab's timeline: frames and conversation items placed on the three x-axes."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from regact.problems.base import build_problem
from regact.protocols.cwm.store import ExperienceStore
from regact.viz import reader, replay


def _iso(at: float) -> str:
    return datetime.fromtimestamp(at, UTC).isoformat()


def _obs(grid: list[list[int]], levels: int = 0) -> dict:
    return {
        "frame": grid,
        "info": {"levels_completed": levels},
        "available_actions": [0, 1, 2, 3, 4],
        "reward": 0.0,
        "is_done": False,
    }


def _run(tmp_path: Path) -> Path:
    """A sokoban run: two steps and an uncounted RESET in a first controller call (stopped by a
    mismatch), one step completing a level in a second; a model update between them."""
    run = tmp_path / "exp" / "2026-10-23_10-00-00" / "sokoban_p01"
    (run / "logs").mkdir(parents=True)
    (run / "cwm").mkdir()
    (run / "config.json").write_text(
        json.dumps({"problem": {"name": "sokoban"}, "protocol": {"name": "cwm"}})
    )
    (run / "logs/experiment_state.json").write_text(json.dumps({"task_name": "sokoban_p01"}))
    store = ExperienceStore(run / "cwm" / "experience.sqlite3")
    grids = [
        [[1, 2], [3, 4]],
        [[2, 1], [3, 4]],
        [[2, 2], [3, 4]],
        [[1, 2], [3, 4]],
        [[4, 4], [4, 4]],
    ]
    episode, _ = store.start_episode(_obs(grids[0]), "initial_collection", {})
    store.record_step(episode, _obs(grids[0]), 1, _obs(grids[1]))
    store.record_step(episode, _obs(grids[1]), 2, _obs(grids[2]))
    store.record_step(episode, _obs(grids[2]), 0, _obs(grids[3]))  # RESET: never scored
    store.record_step(episode, _obs(grids[3]), 3, _obs(grids[4], levels=1))
    for kind, result in (
        ("dataset_prepared", {}),
        (
            "RunController",
            {
                "transition_sequence": [1, 2, 3],
                "counterexample": {"kind": "prediction_mismatch", "diagnostic_id": 7},
            },
        ),
        ("UpdateCodeWorldModel", {"accepted": True}),
        ("RunController", {"transition_sequence": [4]}),
    ):
        store.event(kind, "x", {"result": result})
    store.db.close()
    with sqlite3.connect(run / "cwm" / "experience.sqlite3") as db:
        for event_id, at in ((1, 100.0), (2, 110.0), (3, 120.0), (4, 130.0)):
            db.execute("UPDATE events SET timestamp=? WHERE id=?", (at, event_id))
    lines = [
        {
            "type": "ToolCall",
            "id": "a",
            "name": "Bash",
            "input": {"command": "python framework/commands.py RunController"},
            "ts": _iso(104),
        },
        {"type": "ToolResult", "id": "a", "output": "{}", "ts": _iso(111)},
        {"type": "TextDelta", "text": "The model was wrong.", "ts": _iso(115)},
        {
            "type": "ToolCall",
            "id": "b",
            "name": "Bash",
            "input": {"command": "python framework/commands.py RunController"},
            "ts": _iso(125),
        },
        {"type": "ToolResult", "id": "b", "output": "{}", "ts": _iso(131)},
        {"type": "TextDelta", "text": "Level done.", "ts": _iso(140)},
    ]
    (run / "logs/transcript.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    (run / "logs/events.jsonl").write_text(
        json.dumps({"event": "session_start", "ts": _iso(100)}) + "\n"
    )
    return run


def test_frames_and_items_share_the_three_axes(tmp_path):
    run = _run(tmp_path)
    root = tmp_path / "exp"
    view = reader.load_game(str(root), str(run.relative_to(root)))
    tl = replay.timeline(run, view, build_problem("sokoban", {}))

    frames = tl["frames"]
    assert [f["env_actions"] for f in frames] == [0, 1, 2, 2, 3]  # the RESET is not counted
    assert [f["tool_calls"] for f in frames] == [0, 1, 1, 1, 2]
    # A call's steps are spread over its tool call, from its start (104) to its event (110).
    assert [f["seconds"] for f in frames[1:4]] == [6.0, 8.0, 10.0]
    assert frames[3]["diagnostic"] == 7  # the step that stopped the call on a mismatch
    assert "action" not in frames[0]  # the start of the run
    # Each item: [turn, item, env actions, tool calls, seconds]. A call sits where it started.
    assert [row[2:] for row in tl["items"]] == [
        [0, 1, 4.0],
        [2, 1, 15.0],
        [2, 2, 25.0],
        [3, 2, 40.0],
    ]
    kinds = [(m["kind"], m["env_actions"]) for m in tl["markers"]]
    assert ("mismatch", 2) in kinds and ("model", 2) in kinds and ("progress", 3) in kinds
    assert tl["axes"] == ["env_actions", "tool_calls", "seconds"]
    assert tl["views"] == ["agent"]  # sokoban draws one picture for the agent and for people
    assert (
        replay.frame_png(run, build_problem("sokoban", {}), frames[-1]["obs"], "agent")[:4]
        == b"\x89PNG"
    )


def test_a_run_without_a_store_replays_its_conversation(tmp_path):
    run = _run(tmp_path)
    (run / "cwm" / "experience.sqlite3").unlink()
    root = tmp_path / "exp"
    view = reader.load_game(str(root), str(run.relative_to(root)))
    tl = replay.timeline(run, view, build_problem("sokoban", {}))
    assert tl["frames"] == [] and tl["views"] == []
    assert tl["axes"] == ["tool_calls", "seconds"]
    assert [row[3] for row in tl["items"]] == [1, 1, 2, 2]


def test_viewer_pictures_are_offered_only_where_they_differ():
    assert replay.has_viewer_renderer(build_problem("puzzlescript", {}))
    assert not replay.has_viewer_renderer(build_problem("sokoban", {}))
