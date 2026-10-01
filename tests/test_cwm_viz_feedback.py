
from regact.viz.reader import ToolCallView, TurnItem, TurnView, _tag_tool_calls, list_artifacts


def test_cwm_jump_markers_use_invocations_and_actual_outcomes():
    calls = [
        ToolCallView(id="a", name="Bash", input={"command": "python framework/commands.py UpdateCodeWorldModel"}, result='{"status":"Accepted"}'),
        ToolCallView(id="b", name="PlanInCWM", input={}, result='{"status":"Plan found","goal_achieved":false}'),
        ToolCallView(id="c", name="Bash", input={"command": "python framework/commands.py RunController"}, result='Running in Code World Model... Failure. Stopping here.\n{"status":"Refused"}'),
        ToolCallView(id="d", name="Bash", input={"command": 'rg "control.py PlanInCWM" .'}, result=""),
        ToolCallView(id="e", name="Bash", input={"command": 'echo "python framework/commands.py PlanInCWM"'}, result=""),
        ToolCallView(id="f", name="Bash", input={"command": "cat > demo.py <<'EOF'\npython framework/commands.py PlanInCWM\nEOF\n"}, result=""),
        ToolCallView(id="g", name="Bash", input={"command": "python framework/commands.py PlanInCWM"}, result=None),
    ]
    turn = TurnView(items=[TurnItem(kind="tool", tool=call) for call in calls])
    _tag_tool_calls([turn], [])
    assert [call.framework_tool for call in calls[:3]] == ["UpdateCodeWorldModel", "PlanInCWM", "RunController"]
    assert [call.succeeded for call in calls[:3]] == [True, True, False]
    assert all(call.framework_tool is None for call in calls[3:6])
    assert calls[6].succeeded is None


def test_artifacts_include_markdown_and_skip_outside_symlinks(tmp_path):
    workdir = tmp_path / "task/workdir"
    workdir.mkdir(parents=True)
    (workdir / "CWM_INTERFACE.md").write_text("# Interface")
    (workdir / "controller.py").write_text("pass")
    outside = tmp_path / "outside.md"
    outside.write_text("not an artifact")
    (workdir / "link.md").symlink_to(outside)
    (workdir / "image.png").write_bytes(b"binary")
    assert [item.relpath for item in list_artifacts(str(tmp_path), "task")] == ["CWM_INTERFACE.md", "controller.py"]
