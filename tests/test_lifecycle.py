"""Tests for env lifecycle policies."""

import pytest

from regact.env.lifecycle import MultiInstancePolicy, SingleInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.wrapped_env import WrappedEnv
from regact.obs.errors import ErrorCategory, RegactError
from regact.testing.fakes import FakeNativeEnv


def _build() -> WrappedEnv:
    return WrappedEnv(FakeNativeEnv(), task_name="g", renderer=RawRenderer())


def test_multi_instance_builds_fresh() -> None:
    policy = MultiInstancePolicy()
    a = policy.acquire(_build, key="g")
    b = policy.acquire(_build, key="g")
    assert a is not b
    policy.assert_can_make("g")  # never raises


def test_single_instance_reuses_handle() -> None:
    policy = SingleInstancePolicy()
    a = policy.acquire(_build, key="g")
    b = policy.acquire(_build, key="g")
    assert a is b


def test_single_instance_one_make_guard() -> None:
    policy = SingleInstancePolicy()
    policy.acquire(_build, key="g")
    with pytest.raises(RegactError) as exc:
        policy.assert_can_make("g")
    assert exc.value.category is ErrorCategory.ENV_RUNTIME
    policy.assert_can_make("other")  # a different game is fine
