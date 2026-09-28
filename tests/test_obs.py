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
    assert obs.reward == 0.0
    assert obs.is_done is False
    assert obs.available_actions == []
    assert obs.info == {}


def test_obs_from_partial_payload() -> None:
    obs = Obs.from_json({"frame": [[0]]})
    assert obs.frame == [[0]]
    assert obs.reward == 0.0
    assert obs.available_actions == []
    assert obs.is_done is False


def test_explicit_legacy_null_reward_roundtrips():
    payload = Obs(frame=[[0]], reward=None).to_json()
    assert Obs.from_json(payload).to_json() == payload
