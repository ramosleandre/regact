"""Tests for the observation DTO."""

from regact.envclient.obs import Obs


def test_obs_roundtrip() -> None:
    obs = Obs(
        frame=[[1, 2], [3, 4]],
        reward=1.5,
        is_done=True,
        available_actions=[0, 1, 2],
        info={"state": "PLAYING"},
    )
    assert Obs.from_json(obs.to_json()) == obs


def test_obs_defaults() -> None:
    obs = Obs(frame=None)
    assert obs.reward is None
    assert obs.is_done is False
    assert obs.available_actions == []
    assert obs.info == {}


def test_obs_from_partial_payload() -> None:
    obs = Obs.from_json({"frame": [[0]]})
    assert obs.frame == [[0]]
    assert obs.available_actions == []
    assert obs.is_done is False
