import shlex

import pytest

from regact.viz.reader import (
    ToolCallView,
    _cwm_command,
    _cwm_feedback,
    _group_turns,
    _tag_tool_calls,
)


def test_truncated_feedback_keeps_only_complete_top_level_metadata():
    # A real Claude CWM run used `RunController | head -50`. Its status and
    # playback IDs preceded the truncated counterexample; these are still known.
    text = '''{
  "status": "Completed",
  "exploration_id": 9,
  "current_observation_id": 32,
  "counterexample": {"differences": [
'''
    call = ToolCallView("head", "Bash", {"command": "python framework/control.py RunController | head -50"}, result=text)
    from types import SimpleNamespace
    _tag_tool_calls([SimpleNamespace(tools=[call])], [])
    assert call.succeeded is True and call.controller_playback_ids == [9]
    assert "counterexample" not in _cwm_feedback(text)
    assert _cwm_feedback('{"nested": {"status":"Accepted"}, "x": ') == {}
    assert _cwm_feedback('{"status":"Accep') == {}
    assert "current_observation_id" not in _cwm_feedback('{"status":"Completed","current_observation_id":12')


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("script_name", ["control.py", "commands.py"])
def test_command_after_inline_script(wrapped, script_name):
    # Claude emits the script directly; Codex wraps the whole script, including
    # heredoc newlines, in one quoted shell argument (as in the Luna FT09 run).
    script = f"python - <<'PY'\nprint('local checks')\nPY\npython framework/{script_name} UpdateCodeWorldModel"
    command = "/bin/sh -lc " + shlex.quote(script) if wrapped else script
    call = ToolCallView("1", "shell" if wrapped else "Bash", {"command": command})
    assert _cwm_command(call) == "UpdateCodeWorldModel"
    # Text inside a file being written must never become an execution marker.
    script = f"cat > notes.txt <<'EOF'\npython framework/{script_name} UpdateCodeWorldModel\nEOF"
    call.input["command"] = "/bin/sh -lc " + shlex.quote(script) if wrapped else script
    assert _cwm_command(call) is None


def test_many_calls_in_one_turn_keep_their_results_and_flags():
    events = [
        {"type": "ToolCall", "id": "update", "name": "shell", "input": {"command": '/bin/sh -lc "python framework/control.py UpdateCodeWorldModel"'}},
        {"type": "ToolCall", "id": "run", "name": "Bash", "input": {"command": "python framework/control.py RunController"}},
        {"type": "ToolCall", "id": "flag", "name": "Bash", "input": {"command": 'python -c "import arcengine"'}},
        {"type": "ToolResult", "id": "run", "output": '{"status":"Incomplete"}'},
        {"type": "ToolResult", "id": "update", "output": '{"status":"Accepted"}'},
        {"type": "IterationComplete"},
    ]
    turns = _group_turns(events)
    _tag_tool_calls(turns, [])
    assert len(turns) == 1
    update, run, flag = turns[0].tools
    assert update.framework_tool == "UpdateCodeWorldModel" and update.succeeded
    assert run.framework_tool == "RunController" and not run.succeeded
    assert flag.tag == "cheat" and flag.flags


def test_codex_shell_wrappers_and_quoted_mentions():
    for command in (
        '/bin/sh -lc "python framework/control.py RunController"',
        '/bin/bash -lc "python framework/control.py UpdateCodeWorldModel"',
    ):
        assert _cwm_command(ToolCallView("1", "shell", {"command": command})) is not None
    assert (
        _cwm_command(
            ToolCallView(
                "1", "shell", {"command": 'echo "python framework/control.py RunController"'}
            )
        )
        is None
    )
    assert (
        _cwm_command(
            ToolCallView(
                "1",
                "shell",
                {"command": "/bin/sh -lc 'echo \"python framework/control.py RunController\"'"},
            )
        )
        is None
    )


def test_historical_submit_still_recognized():
    assert (
        _cwm_command(
            ToolCallView(
                "1", "Bash", {"command": "python framework/control.py SubmitExplorationController"}
            )
        )
        == "SubmitExplorationController"
    )


def test_framework_badge_does_not_hide_flag():
    from types import SimpleNamespace

    call = ToolCallView(
        "1",
        "Bash",
        {"command": 'python framework/control.py RunController; python -c "import arcengine"'},
        result='{"status":"Completed"}',
    )
    _tag_tool_calls([SimpleNamespace(tools=[call])], [])
    assert call.framework_tool == "RunController"
    assert call.flags


def test_framework_calls_take_their_outcome_and_playback_from_store_events(tmp_path):
    """Sonnet piped RunController through grep, which drops status and exploration_id from the
    output; the store's command events still give each call its outcome and playback."""
    import json
    import sqlite3

    store = tmp_path / "experience.sqlite3"
    with sqlite3.connect(store) as db:
        db.execute("CREATE TABLE events(id INTEGER PRIMARY KEY, timestamp REAL, kind TEXT, phase TEXT, payload TEXT)")
        for at, kind, result in (
            (100.5, "RunController", {"exploration_id": 7, "observation_sequence": [1, 2]}),
            (200.5, "UpdateCodeWorldModel", {"accepted": True}),
            (201.0, "UpdateCodeWorldModel", {"accepted": False}),
            (201.5, "RunController", {"exploration_id": 8, "observation_sequence": [2]}),
            (300.5, "ResetLevel", {"stop_reason": "reset_level"}),
        ):
            db.execute(
                "INSERT INTO events(timestamp, kind, phase, payload) VALUES(?,?,?,?)",
                (at, kind, "x", json.dumps({"result": result})),
            )

    def call(command, start, end, output=""):
        c = ToolCallView(str(start), "Bash", {"command": command}, result=output)
        c.started, c.ended = start, end
        return c

    piped = call("python framework/commands.py RunController | grep levels", 100, 101, '"levels_completed": 1')
    chained = call("python framework/commands.py UpdateCodeWorldModel; python framework/commands.py RunController", 200, 202)
    reset = call("python framework/commands.py ResetLevel | grep current", 300, 301, '"current_observation_id": 4')
    untimed = ToolCallView("old", "Bash", {"command": "python framework/commands.py RunController"})
    from types import SimpleNamespace

    from regact.viz.reader import _link_store_records

    turns = [SimpleNamespace(tools=[piped, chained, reset, untimed])]
    _tag_tool_calls(turns, [])
    _link_store_records(turns, store)
    assert (piped.succeeded, piped.controller_playback_ids) == (True, [7])
    # Two validations in one call, one refused: not all of its own kind succeeded.
    assert (chained.succeeded, chained.controller_playback_ids) == (False, [8])
    assert reset.succeeded is True and reset.controller_playback_ids == []
    assert untimed.succeeded is None  # an old transcript without timestamps keeps the output parse
