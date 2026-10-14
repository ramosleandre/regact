"""The predicted-versus-observed difference summary agents read (docs/reading_differences.md)."""

import json

import pytest

from regact.protocols.cwm.config import FeedbackConfig
from regact.protocols.cwm.differences import summarize
from test_cwm_protocol import exploration, model
from test_managed_protocol import make_rig  # noqa: F401  (fixture)


def grid(rows=8, cols=8, value=0):
    return [[value] * cols for _ in range(rows)]


def paint(g, cells, value):
    g = [row[:] for row in g]
    for y, x in cells:
        g[y][x] = value
    return g


def frame_regions(predicted, observed, **config):
    result = summarize({"frame": [predicted]}, {"frame": [observed]}, FeedbackConfig(**config))
    (entry,) = result["frame"]
    return entry


def test_rectangles_lines_and_cells_use_slice_notation():
    base = grid()
    rect = paint(base, [(y, x) for y in range(1, 3) for x in range(2, 5)], 7)
    line = paint(base, [(6, x) for x in range(0, 8)], 3)
    cell = paint(base, [(4, 4)], 9)
    assert frame_regions(base, rect)["regions"] == [
        {"where": "frame[0][1:3, 2:5]", "cells": 6, "changes": {"0 -> 7": 6}}
    ]
    assert frame_regions(base, line)["regions"][0]["where"] == "frame[0][6, 0:8]"
    assert frame_regions(base, cell)["regions"][0]["where"] == "frame[0][4, 4]"


def test_diagonal_cells_form_one_region_and_up_to_three_rectangles_are_exact():
    base = grid()
    diagonal = paint(base, [(0, 0), (1, 1)], 5)
    assert len(frame_regions(base, diagonal)["regions"]) == 1
    ell = paint(base, [(y, 0) for y in range(4)] + [(3, x) for x in range(1, 4)], 5)
    (region,) = frame_regions(base, ell)["regions"]
    assert region["where"] == "frame[0][0:4, 0] + frame[0][3, 1:4]"
    assert region["changes"] == {"0 -> 5": 7}


def test_mixed_or_irregular_small_regions_get_a_crop_instead_of_changes():
    before = paint(grid(), [(2, 2), (2, 3)], 12)
    after = paint(grid(), [(2, 4), (2, 5)], 12)
    (region,) = frame_regions(before, after)["regions"]
    assert region["where"] == "frame[0][2, 2:6]" and "changes" not in region
    assert region["predicted"] == ["12 12  0  0"]
    assert region["observed"] == [" 0  0 12 12"]
    stair = paint(grid(), [(i, j) for i in range(5) for j in range(i + 1)], 4)
    (region,) = frame_regions(grid(), stair)["regions"]
    assert region["where"] == "within frame[0][0:5, 0:5]" and region["crop"] == "frame[0][0:5, 0:5]"
    assert len(region["predicted"]) == 5


def test_large_mixed_regions_list_their_most_frequent_changes():
    base = grid(20, 20)
    changed = paint(base, [(y, x) for y in range(20) for x in range(20)], 1)
    changed = paint(changed, [(0, x) for x in range(20)], 2)
    changed = paint(changed, [(1, 0)], 3)
    (region,) = frame_regions(base, changed, max_diff_changes=2)["regions"]
    assert "predicted" not in region
    assert region["changes"] == {"0 -> 1": 379, "0 -> 2": 20, "other changes": 1}


def test_regions_beyond_the_budget_are_counted():
    scattered = paint(grid(), [(0, 0), (0, 4), (4, 0), (4, 4), (7, 7)], 1)
    entry = frame_regions(grid(), scattered, max_diff_regions=2)
    assert entry["cells_differing"] == 5 and len(entry["regions"]) == 2
    assert entry["other_regions"] == "3 more regions, 3 cells"


def test_every_field_has_its_own_budget():
    predicted = {
        "frame": [grid()],
        "info": {"state": "NOT_FINISHED", "milestones": []},
        "is_done": False,
    }
    observed = {
        "frame": [paint(grid(), [(y, y) for y in range(0, 8, 2)], 1)],
        "info": {"state": "GAME_OVER", "milestones": ["game over"]},
        "is_done": True,
    }
    result = summarize(predicted, observed, FeedbackConfig(max_diff_items=1, max_diff_regions=1))
    assert set(result) == {"frame", "info", "is_done"}
    assert result["is_done"] == [{"where": "is_done", "predicted": False, "observed": True}]
    assert result["info"][0] == {
        "where": "info.milestones",
        "predicted": [],
        "observed": ["game over"],
    }
    assert result["info"][1] == {"other differences": 1}


def test_shape_changes_layer_counts_rgb_cells_and_missing_fields():
    ragged = [row[:] for row in grid(4, 4)]
    ragged[1] = []
    assert frame_regions(ragged, grid(4, 4)) == {
        "where": "frame[0]",
        "shape": {"predicted": "4 rows of 0 to 4", "observed": "4 rows of 4"},
    }
    assert frame_regions(grid(3, 4), grid(4, 4))["shape"] == {
        "predicted": [3, 4],
        "observed": [4, 4],
    }
    layers = summarize({"frame": [grid(), grid()]}, {"frame": [grid()]}, FeedbackConfig())
    assert layers["frame"] == [{"where": "len(frame)", "predicted": 2, "observed": 1}]
    image = [[[0, 0, 0] for _ in range(5)] for _ in range(5)]
    moved = json.loads(json.dumps(image))
    moved[1][1] = [255, 0, 0]
    (entry,) = summarize({"image": image}, {"image": moved}, FeedbackConfig())["image"]
    assert entry["regions"][0]["changes"] == {"[0, 0, 0] -> [255, 0, 0]": 1}
    missing = summarize({"reward": 0}, {"reward": 0, "is_done": True}, FeedbackConfig())
    assert missing == {
        "is_done": [{"where": "is_done", "predicted": "<missing>", "observed": True}]
    }


def test_null_budgets_list_everything():
    predicted = {"info": dict.fromkeys("abcdefgh", 0)}
    observed = {"info": dict.fromkeys("abcdefgh", 1)}
    config = FeedbackConfig(max_diff_items=None)
    assert len(summarize(predicted, observed, config)["info"]) == 8
    scattered = paint(grid(), [(0, 0), (0, 4), (4, 0), (4, 4), (7, 7)], 1)
    assert len(frame_regions(grid(), scattered, max_diff_regions=None)["regions"]) == 5


def _with_progress(c):
    c.problem.progress_label = "Levels completed"
    c.problem.progress_value = lambda info: 0


@pytest.mark.integration
def test_a_contradiction_names_its_step_and_how_to_load_it(make_rig):  # noqa: F811
    c = make_rig("cwm")
    _with_progress(c)
    model(c.workdir, bad=True)
    assert c.tool("UpdateCodeWorldModel", {}).get("accepted")
    exploration(c, (1, 1))
    result = c.tool("RunController", {})
    example = result["counterexample"]
    assert example["action"] == 1 and example["levels_completed"] == 0
    assert (
        example["from_observation_id"]
        == c.store.transition(example["transition_id"])["before_obs_id"]
    )
    assert example["full_data"] == f"data_api.load_diagnostic({result['diagnostic_id']})"
    assert result["differences"]["frame"][0]["where"] == "frame"
    refused = c.tool("UpdateCodeWorldModel", {})
    (failed,) = refused["counterexamples"]
    assert failed["transition_id"] == example["transition_id"]
    assert failed["explained_by_previous_version"] is False  # new evidence, never predicted
    assert failed["full_data"].startswith("data_api.load_diagnostic(")


@pytest.mark.integration
def test_a_refusal_says_when_the_edit_broke_an_explained_step(make_rig):  # noqa: F811
    c = make_rig("cwm")
    exploration(c, (1, 1))
    assert c.tool("RunController", {})["real_actions"] == 2  # predicted by the accepted model
    model(c.workdir, bad=True)
    refused = c.tool("UpdateCodeWorldModel", {})
    assert refused["counterexamples"][0]["explained_by_previous_version"] is True


def test_the_reading_guide_is_in_the_agent_docs_with_the_run_budgets():
    from regact.protocols.cwm.config import CwmConfig
    from regact.protocols.cwm.prompting import workspace_docs

    options = CwmConfig.from_mapping({"feedback": {"max_diff_regions": 3, "max_diff_items": None}})
    docs = {f.relpath: f.content for f in workspace_docs(options, vision=False)}
    guide = docs["docs/reading_differences.md"]
    assert "__" not in guide and "At most 3 regions" in guide
    assert "docs/reading_differences.md" in docs["docs/CWM_modeling_phase.md"]
