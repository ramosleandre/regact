"""Unit tests for the viz reader + metrics (no server).

Builds a tiny fake experiment dir (our layout) and checks the transcript is folded
into turns, submissions are read, and the proxy metrics are computed.
"""

import json
from pathlib import Path

from regact.viz.metrics import game_metrics
from regact.viz.reader import list_artifacts, list_games, load_game, load_logs

_TRANSCRIPT = [
    {"type": "ThinkingDelta", "text": "let me probe"},
    {"type": "TextDelta", "text": "Exploring."},
    {"type": "ToolCall", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
    {"type": "ToolResult", "id": "t1", "output": "files", "is_error": False},
    {"type": "ToolCall", "id": "t2", "name": "SubmitSolution", "input": {}},
    {"type": "ToolResult", "id": "t2", "output": "scored", "is_error": False},
    {"type": "TurnComplete", "final_text": "", "usage": {"output_tokens": 12, "input_tokens": 100}},
    {"type": "ToolCall", "id": "t3", "name": "Bash", "input": {"command": "echo"}},
    {"type": "ToolResult", "id": "t3", "output": "ok", "is_error": False},
    {"type": "TurnComplete", "usage": {"output_tokens": 5}},
]


def _make_experiment(tmp_path: Path) -> str:
    game = tmp_path / "exp" / "ls20"
    (game / "logs").mkdir(parents=True)
    (game / "workdir" / "submissions" / "0").mkdir(parents=True)
    (game / "logs" / "experiment_state.json").write_text(
        json.dumps({"problem_name": "arc_agi", "task_name": "ls20", "exit_requested": True})
    )
    (game / "logs" / "transcript.jsonl").write_text("\n".join(json.dumps(e) for e in _TRANSCRIPT))
    (game / "workdir" / "submissions" / "0" / "results.json").write_text(
        json.dumps(
            {"aggregate": {"success_rate": 0.0, "mean_levels_completed": 2, "mean_steps": 30}}
        )
    )
    return str(tmp_path / "exp")


def test_list_games(tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    assert list_games(exp) == ["ls20"]


def test_transcript_folds_into_turns(tmp_path: Path) -> None:
    game = load_game(_make_experiment(tmp_path), "ls20")
    assert len(game.turns) == 2
    first = game.turns[0]
    # Items are kept in chronological order (thinking → text → tool → tool).
    assert [i.kind for i in first.items] == ["thinking", "text", "tool", "tool"]
    assert first.items[0].text == "let me probe"
    assert first.items[2].tool.name == "Bash"
    # ToolResult paired to its ToolCall by id.
    assert first.items[2].tool.result == "files"
    assert first.usage == {"output_tokens": 12, "input_tokens": 100}
    # Convenience properties still work (used by metrics).
    assert first.thinkings == ["let me probe"] and first.texts == ["Exploring."]


def test_metrics_proxies(tmp_path: Path) -> None:
    m = game_metrics(load_game(_make_experiment(tmp_path), "ls20"))
    assert m["n_turns"] == 2
    assert m["n_tool_calls"] == 3
    assert m["tool_histogram"] == {"Bash": 2, "SubmitSolution": 1}
    assert m["n_submissions"] == 1
    assert m["tokens"]["output"] == 17
    assert m["best_levels"] == 2
    assert m["submission_trajectory"][0]["submission"] == 0


def test_artifacts_lists_workdir_python(tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    (Path(exp) / "ls20" / "workdir" / "solution.py").write_text(
        "def get_controller():\n    return None\n"
    )
    files = list_artifacts(exp, "ls20")
    rel = {f.relpath for f in files}
    assert "solution.py" in rel
    sol = next(f for f in files if f.relpath == "solution.py")
    assert "get_controller" in sol.content


def test_logs_reads_output_and_events(tmp_path: Path) -> None:
    exp = _make_experiment(tmp_path)
    logs = Path(exp) / "ls20" / "logs"
    logs.joinpath("output.log").write_text("hello run\n")
    logs.joinpath("events.jsonl").write_text(
        json.dumps({"component": "loop", "level": "ERROR", "event": "turn_crash"}) + "\n"
    )
    out = load_logs(exp, "ls20")
    assert "hello run" in out["output"]
    assert out["events"][0]["event"] == "turn_crash"
