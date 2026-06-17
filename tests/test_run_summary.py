"""Tests for the controller-rollout summary."""

from regact.controllers.summary import ControllerRun, ControllerSummary, MilestoneEvent
from regact.envclient.obs import Obs


def test_milestones_flatten_depth_first() -> None:
    inner = ControllerRun(name="sub", events=[MilestoneEvent(step=3, description="x")])
    root = ControllerRun(
        name="root",
        events=[
            MilestoneEvent(step=1, description="a"),
            inner,
            MilestoneEvent(step=5, description="b"),
        ],
    )
    summary = ControllerSummary(
        stop_kind="env_done",
        stop_reason="win",
        total_steps=5,
        history=root,
        final_obs=Obs(frame=None),
    )
    assert [m.step for m in summary.milestones] == [1, 3, 5]


def test_repr_is_compact() -> None:
    summary = ControllerSummary(
        stop_kind="max_steps",
        stop_reason="cap",
        total_steps=10,
        history=ControllerRun(name="r"),
        final_obs=Obs(frame=None),
    )
    text = repr(summary)
    assert "max_steps" in text
    assert "total_steps=10" in text
    assert "milestones=0" in text
