"""The env-wrapper seam: delegation, application order, per-instance wrapping."""

from typing import Any

from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.session import EnvSession
from regact.env.wrapper import EnvWrapper
from regact.features.base import Feature, FeatureContext, Hook, RunDeps, TemplateFile
from regact.testing.fakes import FakeNativeEnv
from regact.tools.base import Tool


class _Tag(EnvWrapper):
    """Logs its label on each step, then delegates."""

    def __init__(self, inner: Any, label: str, log: list[str]) -> None:
        super().__init__(inner)
        self._label = label
        self._log = log

    def step(self, action: Any) -> Any:
        self._log.append(self._label)
        return super().step(action)


def _session(wrappers: list[Any]) -> EnvSession:
    return EnvSession(
        make_native=lambda: FakeNativeEnv(goal=2),
        key="corridor",
        renderer=RawRenderer(),
        lifecycle=MultiInstancePolicy(),
        wrappers=wrappers,
    )


def test_wrapper_delegates_the_wrapped_env_surface() -> None:
    session = _session([lambda env: EnvWrapper(env)])
    env = session.make()
    env.reset()
    assert env.action_count == 0
    obs = env.step(1)
    assert env.action_count == 1  # attribute read through __getattr__
    assert env.last_obs is obs  # same object the inner env produced
    assert obs.frame == {"pos": 1, "grid": [0, 1, 0]}


def test_wrappers_apply_in_list_order_first_listed_innermost() -> None:
    log: list[str] = []
    session = _session([lambda env: _Tag(env, "first", log), lambda env: _Tag(env, "second", log)])
    env = session.make()
    env.reset()
    env.step(1)
    # second(first(env)): the outermost (last-listed) wrapper runs first.
    assert log == ["second", "first"]


def test_every_multi_instance_build_is_wrapped() -> None:
    calls: list[str] = []

    def factory(env: Any) -> Any:
        calls.append("wrap")
        return EnvWrapper(env)

    session = _session([factory])
    session.make()
    session.make()
    assert calls == ["wrap", "wrap"]


class _Bare(Feature):
    """A feature that overrides nothing optional - to assert the base defaults."""

    name = "bare"

    def templates(self, ctx: FeatureContext) -> list[TemplateFile]:
        return []

    def prompt_fragment(self, ctx: FeatureContext) -> str | None:
        return None

    def tools(self, deps: RunDeps) -> list[Tool]:
        return []

    def hooks(self, deps: RunDeps) -> list[Hook]:
        return []


def test_feature_env_wrapper_defaults_to_none() -> None:
    ctx = FeatureContext(problem_name="p", task_name="t", workdir="w")
    assert _Bare().env_wrapper(ctx) is None
