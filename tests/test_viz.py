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


def test_result_pairs_across_a_turn_boundary(tmp_path: Path) -> None:
    # Per-completion pattern (the alan adapter closes every completion with a
    # TurnComplete): a ToolResult arrives AFTER its call's turn was flushed, and must
    # still pair to that call (persistent id map). Each completion is its own turn.
    transcript = [
        {"type": "ToolCall", "id": "a", "name": "Bash", "input": {"command": "ls"}},
        {"type": "TurnComplete", "usage": {"output_tokens": 3}},
        {"type": "ToolResult", "id": "a", "output": "files", "is_error": False},
        {"type": "ToolCall", "id": "b", "name": "Bash", "input": {"command": "cat x"}},
        {"type": "TurnComplete", "usage": {"output_tokens": 4}},
        {"type": "ToolResult", "id": "b", "output": "boom", "is_error": True},
    ]
    game = tmp_path / "exp" / "g"
    (game / "logs").mkdir(parents=True)
    (game / "logs" / "experiment_state.json").write_text("{}")
    (game / "logs" / "transcript.jsonl").write_text("\n".join(json.dumps(e) for e in transcript))
    view = load_game(str(tmp_path / "exp"), "g")
    assert len(view.turns) == 2  # one turn per completion, not one giant turn
    assert view.turns[0].tools[0].result == "files"  # paired across the flush boundary
    assert view.turns[1].tools[0].result == "boom" and view.turns[1].tools[0].is_error


def test_metrics_proxies(tmp_path: Path) -> None:
    m = game_metrics(load_game(_make_experiment(tmp_path), "ls20"))
    assert m["n_turns"] == 2
    assert m["n_tool_calls"] == 3
    assert m["tool_histogram"] == {"Bash": 2, "SubmitSolution": 1}
    assert m["n_submissions"] == 1
    assert m["tokens"]["output"] == 17
    # Game-agnostic: the whole aggregate is passed through opaquely (no named "levels" key).
    assert m["final_aggregate"]["mean_levels_completed"] == 2
    traj0 = m["submission_trajectory"][0]
    assert traj0["submission"] == 0
    assert traj0["metrics"]["mean_levels_completed"] == 2


def test_submit_call_tagged_win_when_a_level_clears(tmp_path: Path) -> None:
    """The SubmitSolution whose submission cleared levels (mean_levels_completed=2) is a win."""
    game = load_game(_make_experiment(tmp_path), "ls20")
    by_name = {c.name: c for turn in game.turns for c in turn.tools}
    assert by_name["SubmitSolution"].tag == "submit_win"  # green
    assert by_name["Bash"].tag is None  # a plain shell call


def test_grep_mentioning_submitsolution_is_not_counted_as_a_submit() -> None:
    """A grep/rg that merely MENTIONS the strings must not be a submit (it would shift the
    submit->submission alignment and mis-color the wins)."""
    from regact.viz.reader import ToolCallView, _is_submit_call

    real = ToolCallView("1", "shell", {"command": "python framework/control.py SubmitSolution"})
    grep = ToolCallView("2", "shell", {"command": 'rg -n "control/.*tool|SubmitSolution" src'})
    assert _is_submit_call(real) is True
    assert _is_submit_call(grep) is False


def test_cheat_call_is_tagged(tmp_path: Path) -> None:
    """A tool call reaching for a forbidden path is tagged 'cheat' (red), like the loop flags it."""
    game_dir = tmp_path / "exp" / "g"
    (game_dir / "logs").mkdir(parents=True)
    (game_dir / "logs" / "experiment_state.json").write_text(
        json.dumps({"problem_name": "p", "task_name": "g"})
    )
    cheat = {"command": "cat ../environnement/x"}  # references a forbidden path
    events = [
        {"type": "ToolCall", "id": "c1", "name": "Bash", "input": cheat},
        {"type": "ToolResult", "id": "c1", "output": "x", "is_error": False},
        {"type": "TurnComplete", "usage": {}},
    ]
    (game_dir / "logs" / "transcript.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    game = load_game(str(tmp_path / "exp"), "g")
    assert game.turns[0].tools[0].tag == "cheat"


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


def test_final_score_falls_back_past_errored_final_and_surfaces_both() -> None:
    """The Score KPI must show the last REAL submission when 'final' errored (e.g. a teardown
    ReadTimeout -> empty aggregate), instead of rendering blank; and it surfaces both the
    verified (shadow-replay) score and the direct one."""
    from regact.viz.metrics import game_metrics
    from regact.viz.reader import GameView, SubmissionView

    submissions = [
        SubmissionView(
            "0",
            {"n_episodes": 5, "success_rate": 1.0, "mean_reward": 0.98},
            [],
            None,
            [],
            aggregate_unverified={"n_episodes": 5, "success_rate": 1.0, "mean_reward": 0.98},
        ),
        SubmissionView("final", {}, [], "ReadTimeout: timed out", []),  # errored -> empty
    ]
    game = GameView(name="g", state={}, turns=[], submissions=submissions, config={})
    m = game_metrics(game)

    assert m["final_aggregate"]["success_rate"] == 1.0  # fell back past the blank final
    assert m["success_rate"] == 1.0
    assert m["final_aggregate_unverified"]["success_rate"] == 1.0  # both scores surfaced
