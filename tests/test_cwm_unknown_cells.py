"""protocol.allow_unknown_cells: render may leave parts of the frame unpredicted."""

import pytest

from regact.protocols.cwm.prompting import _render
from regact.protocols.cwm.unknown import UNKNOWN, UNKNOWN_FRAMES, UnknownShare, resolve
from regact.protocols.cwm.validation import add_state_size
from test_cwm_hidden_state import hidden_model, hidden_rig  # noqa: F401
from test_cwm_protocol import exploration

pytestmark = pytest.mark.integration


def obs(frame):
    return {"frame": frame, "reward": 0.0, "is_done": False, "available_actions": [1], "info": {}}


def test_unknown_parts_are_filled_from_the_observation():
    real = obs([[1, 2], [3, 4]])
    filled, unknown = resolve(obs([[1, UNKNOWN], UNKNOWN]), real, True)
    assert filled == real
    assert unknown == [("frame[0][1]", 2), ("frame[1]", [3, 4])]
    wrong, _ = resolve(obs([[9, UNKNOWN], UNKNOWN]), real, True)
    assert wrong["frame"] == [[9, 2], [3, 4]]  # the known cell still differs


def test_unknown_is_read_only_in_the_frame_and_only_when_allowed():
    real = obs([1, 2])
    rendered = {**obs([1, 2]), "reward": UNKNOWN}
    assert resolve(rendered, real, True) == (rendered, [])
    masked = obs([UNKNOWN, 2])
    assert resolve(masked, real, False) == (masked, [])


def test_unknown_frames_stand_for_any_number_of_leading_frames():
    last = [[5, 5], [5, 5]]
    rendered = obs([UNKNOWN_FRAMES, last])
    for frames in ([last], [[[0, 0], [0, 0]], [[1, 1], [1, 1]], last]):
        filled, unknown = resolve(rendered, obs(frames), True)
        assert filled == obs(frames) and len(unknown) == len(frames) - 1
    # Two known frames cannot match one observed frame.
    filled, _ = resolve(obs([UNKNOWN_FRAMES, last, last]), obs([last]), True)
    assert filled != obs([last])


def test_only_cells_that_change_are_charged():
    share = UnknownShare()
    frames = [[0, 7, n] for n in range(4)]  # only the last cell changes
    for frame in frames:
        share.add([("frame[1]", frame[1])])
    assert share.report(frames)["share_of_changing_cells"] == 0.0
    share.add([("frame[2]", 0)])
    assert share.report(frames)["share_of_changing_cells"] == 0.25


def test_unknown_bytes_earn_no_state_size():
    sizes = {}
    add_state_size(sizes, 1, {"n": 1}, obs([1] * 50), 0.5, None, unpredicted_bytes=40)
    full = {}
    add_state_size(full, 1, {"n": 1}, obs([1] * 50), 0.5, None)
    assert full["observation_bytes"] - sizes["observation_bytes"] == 40


def render(frame):
    return (
        f'def render(s): return {{"frame":{frame}, "reward":float(s.n>=20),"is_done":s.n>=20,'
        '"available_actions":[1],"info":{"milestones":[]}}\n'
    )


def test_a_cwm_with_unknown_cells_is_accepted_and_runs(hidden_rig):  # noqa: F811
    c = hidden_rig(target=3, allow_unknown_cells=True)
    hidden_model(c.workdir)
    (c.workdir / "world_model" / "model_render.py").write_text(
        render(f'[s.n//2]*20 + ["{UNKNOWN}"]*10')
    )
    accepted = c.tool("UpdateCodeWorldModel", {})
    assert accepted["accepted"], accepted
    assert accepted["unknown_cells"]["share_of_changing_cells"] == pytest.approx(1 / 3, abs=1e-3)
    exploration(c, (1, 1))
    ran = c.tool("RunController", {})
    assert ran["real_actions"] == 2 and "error" not in ran, ran


def test_unknown_cells_are_refused_when_not_allowed(hidden_rig):  # noqa: F811
    c = hidden_rig(target=3)
    hidden_model(c.workdir)
    (c.workdir / "world_model" / "model_render.py").write_text(
        render(f'[s.n//2]*20 + ["{UNKNOWN}"]*10')
    )
    refused = c.tool("UpdateCodeWorldModel", {})
    assert not refused["accepted"] and "reconstruction_mismatch" in refused["failures"]


def test_a_wrong_known_cell_is_still_refused(hidden_rig):  # noqa: F811
    c = hidden_rig(target=3, allow_unknown_cells=True)
    hidden_model(c.workdir, step="State(s.n+2)")
    (c.workdir / "world_model" / "model_render.py").write_text(
        render(f'[s.n//2]*20 + ["{UNKNOWN}"]*10')
    )
    refused = c.tool("UpdateCodeWorldModel", {})
    assert refused["failures"] == {"prediction_mismatch": 1}
    [example] = refused["counterexamples"]
    assert "frame" in example["differences"]


def test_too_many_unknown_cells_are_refused(hidden_rig):  # noqa: F811
    c = hidden_rig(target=3, allow_unknown_cells=True)
    hidden_model(c.workdir)
    (c.workdir / "world_model" / "model_render.py").write_text(render(f'["{UNKNOWN}"]*30'))
    refused = c.tool("UpdateCodeWorldModel", {})
    assert not refused["accepted"] and refused["failures"] == {"unknown_cells": 1}, refused
    assert refused["unknown_cells"]["share_of_changing_cells"] == 1.0


def test_the_docs_describe_unknown_cells_only_when_allowed(hidden_rig):  # noqa: F811
    for allowed in (False, True):
        c = hidden_rig(target=3, allow_unknown_cells=allowed)
        text = (c.workdir / "docs" / "modeling_phase.md").read_text()
        assert (UNKNOWN_FRAMES in text) is allowed and "__UNKNOWN" not in text
        assert ("`unknown_cells`:" in text) is allowed
        assert ("**Unknown cells.**" in _render("workflow.md", c.options)) is allowed
