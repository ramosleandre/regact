"""Tests for the server-side observation renderer."""

from regact.env.renderer import RawRenderer


def test_raw_renderer_passes_frame_and_actions() -> None:
    obs = RawRenderer().render({"pos": 1}, {"available_actions": [0, 1]})
    assert obs.frame == {"pos": 1}
    assert obs.available_actions == [0, 1]
    assert obs.info == {"available_actions": [0, 1]}


def test_raw_renderer_handles_none_info() -> None:
    obs = RawRenderer().render([1, 2, 3], None)
    assert obs.frame == [1, 2, 3]
    assert obs.available_actions == []
    assert obs.info == {}
