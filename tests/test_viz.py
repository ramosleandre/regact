"""Unit tests for the viz reader + metrics (no server).

Builds a tiny fake experiment dir (our layout) and checks the transcript is folded
into turns, submissions are read, and the proxy metrics are computed.
"""

import json
from pathlib import Path

import pytest

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


def test_list_games_recurses_a_sweep_and_dedupes_latest_symlink(tmp_path: Path) -> None:
    """One viz over a whole sweep: nested run dirs are listed by their relative path, a run
    is not descended into, and a `latest` symlink resolving to a timestamp dir is not listed
    twice."""
    import os

    root = tmp_path / "sweep"
    runs = ["modelA_seed0/2026-08-15_a/DoorKey", "modelB_seed0/2026-08-15_b/DoorKey"]
    for r in runs:
        (root / r / "logs").mkdir(parents=True)
        (root / r / "logs" / "experiment_state.json").write_text("{}")  # the real-game marker
    os.symlink(root / "modelA_seed0" / "2026-08-15_a", root / "modelA_seed0" / "latest")

    games = list_games(str(root))
    assert games == sorted(runs)  # nested relpaths, and the `latest` symlink deduped away
    # a run dir is not descended into: a stray logs/ nested INSIDE a run is not a second game
    (root / runs[0] / "workdir" / "logs").mkdir(parents=True)
    assert list_games(str(root)) == sorted(runs)


def test_list_games_skips_bare_logs_dirs(tmp_path: Path) -> None:
    """A dir with a ``logs/`` but no ``experiment_state.json`` is NOT a game - Slurm job-log folders
    (sbatch/simplelm output) litter the experiments root and must not masquerade as runs."""
    root = tmp_path / "root"
    (root / "slurm_job" / "logs").mkdir(parents=True)
    (root / "slurm_job" / "logs" / "sbatch.123.out").write_text("...")  # no run state -> not a game
    (root / "real/2026-08-15_a/DoorKey" / "logs").mkdir(parents=True)
    (root / "real/2026-08-15_a/DoorKey" / "logs" / "experiment_state.json").write_text("{}")
    assert list_games(str(root)) == ["real/2026-08-15_a/DoorKey"]


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


def test_games_endpoint_tags_experiment_and_task(tmp_path: Path) -> None:
    """/api/games tags each run with its experiment (config.experiment_name) and task
    (state.task_name) so the graphs panel groups + aggregates across runs - including a task
    repeated across timestamps within one experiment (the aggregation case)."""
    from fastapi.testclient import TestClient

    from regact.viz.app import build_app

    root = tmp_path / "bench"
    runs = {
        "claude-fo/2026-01-01_a/DoorKey": ("bench/claude-fo", "DoorKey"),
        "claude-fo/2026-01-01_a/Empty": ("bench/claude-fo", "Empty"),
        "claude-fo/2026-01-02_b/Empty": ("bench/claude-fo", "Empty"),  # a rerun of Empty
        "codex-fo/2026-01-01_a/DoorKey": ("bench/codex-fo", "DoorKey"),
    }
    for rel, (exp, task) in runs.items():
        d = root / rel
        (d / "logs").mkdir(parents=True)
        (d / "logs" / "experiment_state.json").write_text(json.dumps({"task_name": task}))
        (d / "logs" / "transcript.jsonl").write_text("")
        (d / "config.json").write_text(json.dumps({"experiment_name": exp}))

    games = TestClient(build_app(str(root))).get("/api/games").json()["games"]
    assert {g["name"]: (g["experiment"], g["task"]) for g in games} == runs
    empties = [g for g in games if g["task"] == "Empty" and g["experiment"] == "bench/claude-fo"]
    assert len(empties) == 2  # both timestamps surface, to be aggregated per (experiment, task)


def test_task_preview_renders_png() -> None:
    """render_task_png builds a small PNG of a task's env, offline. It runs from
    scripts/gen_task_previews.py to pre-populate static/icons_tasks/ (never at viz time)."""
    from regact.viz.task_preview import render_task_png

    png = render_task_png("MiniGrid-Empty-5x5-v0")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic


def test_settings_persist_per_interface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """/api/settings saves + reloads a per-interface JSON (under REGACT_VIZ_SETTINGS_DIR), so the
    graphs panel's colors/order/toggles persist across sessions, scoped by the graph `under`."""
    from fastapi.testclient import TestClient

    from regact.viz.app import build_app

    monkeypatch.setenv("REGACT_VIZ_SETTINGS_DIR", str(tmp_path / "settings"))
    client = TestClient(build_app(str(tmp_path)))
    scope = "2026-08-29-bench-minigrid-01"
    assert client.get("/api/settings", params={"scope": scope}).json() == {}  # none yet
    blob = {"version": 1, "colors": {"alan-MiniMax-M2.7-Q8-fo": "#865e3c"}, "order": ["a", "b"]}
    assert client.put("/api/settings", params={"scope": scope}, json=blob).status_code == 200
    assert client.get("/api/settings", params={"scope": scope}).json() == blob  # round-trips
    assert (
        client.get("/api/settings", params={"scope": "other-bench"}).json() == {}
    )  # isolated per interface


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


def test_build_tree_names_each_rung_of_the_hierarchy(tmp_path: Path) -> None:
    """build_tree nests folders benchmark -> experiment -> run -> task from a cheap walk - no metric
    parsing - so a many-experiment root browses instantly, and each rung is labelled so the landing
    can separate benchmark folders from bare experiments."""
    from regact.viz.reader import build_tree

    s1, s2 = "2026-08-26_17-56-06", "2026-08-26_18-00-00"
    paths = [
        f"bench/claude_arc/{s1}/ft09",  # bench holds two experiments -> a benchmark
        f"bench/claude_arc/{s1}/ls20",
        f"bench/codex_arc/{s2}/tn36",
        f"solo/{s1}/mg01",  # solo holds runs directly -> a bare experiment
    ]
    for p in paths:
        (tmp_path / p / "logs").mkdir(parents=True)
        (tmp_path / p / "logs" / "experiment_state.json").write_text("{}")
    tree = build_tree(str(tmp_path))
    by_name = {n["name"]: n for n in tree}
    assert set(by_name) == {"bench", "solo"}
    assert by_name["bench"]["kind"] == "benchmark"
    assert by_name["bench"]["n_children"] == 2 and by_name["bench"]["n_tasks"] == 3
    assert by_name["solo"]["kind"] == "experiment" and by_name["solo"]["n_children"] == 1
    claude = by_name["bench"]["children"][0]  # bench/claude_arc
    assert claude["kind"] == "experiment"
    run = claude["children"][0]  # bench/claude_arc/<stamp>
    assert run["kind"] == "run" and run["n_children"] == 2
    assert {c["name"] for c in run["children"]} == {"ft09", "ls20"}
    assert all(c["kind"] == "task" for c in run["children"])


def test_build_tree_survives_repeated_task_attempts(tmp_path: Path) -> None:
    """A repeated task nests an extra <task>/attempt_N level. The timestamped run must still read as
    a run (and its parent as an experiment) - detection is by the stamp name, not by child depth,
    so n_attempts_per_task cannot shift every folder up a rung."""
    from regact.viz.reader import build_tree

    stamp = "2026-08-26_17-56-06"
    for p in [f"e/{stamp}/DoorKey/attempt_0", f"e/{stamp}/DoorKey/attempt_1", f"e/{stamp}/Empty"]:
        (tmp_path / p / "logs").mkdir(parents=True)
        (tmp_path / p / "logs" / "experiment_state.json").write_text("{}")
    tree = build_tree(str(tmp_path))
    assert [n["kind"] for n in tree] == ["experiment"]  # not benchmark: the attempts nest below run
    run = tree[0]["children"][0]
    assert run["kind"] == "run" and run["n_children"] == 2  # DoorKey + Empty task dirs


def test_api_games_scopes_to_a_subtree(tmp_path: Path) -> None:
    """/api/games?under=<run> parses only that subtree, so the browser never parses the whole
    root at once."""
    from fastapi.testclient import TestClient

    from regact.viz.app import build_app

    for p in ["bench/claude_arc/2026_10/ft09", "bench/codex_arc/2026_11/ls20"]:
        (tmp_path / p / "logs").mkdir(parents=True)
        (tmp_path / p / "logs" / "experiment_state.json").write_text('{"task_name": "x"}')
        (tmp_path / p / "config.json").write_text("{}")
    client = TestClient(build_app(str(tmp_path)))
    assert client.get("/api/tree").json()["tree"][0]["name"] == "bench"
    assert len(client.get("/api/games").json()["games"]) == 2  # no scope = every game
    scoped = client.get("/api/games", params={"under": "bench/claude_arc/2026_10"}).json()["games"]
    assert len(scoped) == 1 and scoped[0]["name"] == "bench/claude_arc/2026_10/ft09"
