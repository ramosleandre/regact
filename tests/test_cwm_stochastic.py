"""CWM-stochastic: named random choices in step, explanations of recorded steps, and their cost."""

import math
import random
import sys

import pytest

from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.validation import Loss
from regact.protocols.cwm.worker_entry import _Trace
from regact.protocols.cwm.workspace_helpers import CWM_STOCHASTIC

MODEL = {
    "model_state.py": "from dataclasses import dataclass\n\n@dataclass(frozen=True)\nclass State:\n    pos: int\n    coin: str\n",
    "model_initial_state.py": "from world_model.model_state import State\n\ndef get_initial_state(obs):\n    return State(obs['pos'], obs['coin'])\n",
    "model_render.py": "def render(state):\n    return {'pos': state.pos, 'coin': state.coin}\n",
    "model_transition.py": (
        "from world_model.model_state import State\n\n"
        "def step(state, action, choose):\n"
        "    slip = choose('slip', {'no': 0.75, 'yes': 0.25})\n"
        "    coin = choose('coin@a', ['heads', 'tails'])\n"
        "    return State(state.pos + (0 if slip == 'yes' else action), coin)\n"
    ),
}


@pytest.fixture
def helper(tmp_path, monkeypatch):
    package = tmp_path / "world_model"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name, source in MODEL.items():
        (package / name).write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m.startswith("world_model")]:
        monkeypatch.delitem(sys.modules, name)
    namespace: dict = {}
    exec(compile(CWM_STOCHASTIC, "cwm_stochastic", "exec"), namespace)
    return namespace


def test_an_explanation_is_the_cheapest_values_that_reproduce_the_observation(helper):
    from world_model.model_state import State

    start = State(0, "heads")
    likely = helper["explain_step"](start, 1, {"pos": 1, "coin": "heads"})
    assert likely["explained"] and likely["runs"] == 1
    assert likely["bits"] == pytest.approx(-math.log2(0.75) + 1)
    rare = helper["explain_step"](start, 1, {"pos": 0, "coin": "tails"})
    assert rare["explained"] and rare["bits"] == pytest.approx(2 + 1)  # searched, not given
    assert {name: value for name, value, _, _ in rare["choices"]} == {"slip": "yes", "coin@a": "tails"}
    impossible = helper["explain_step"](start, 1, {"pos": 5, "coin": "heads"})
    assert not impossible["explained"] and impossible["bits"] is None
    state, choices = helper["sample_step"](start, 1, random.Random(0))
    assert state.pos in (0, 1) and len(choices) == 2


def test_choose_checks_its_distribution_and_pins():
    trace = _Trace({"a": "x"}, [])
    assert trace.choose("a", {"x": 0.5, "y": 0.5}) == "x" and trace.possible
    assert trace.choose("b", ["p", "q"]) == "p" and trace.bits() == pytest.approx(2)
    with pytest.raises(ValueError):
        trace.choose("a", ["again"])
    with pytest.raises(ValueError):
        _Trace({}, []).choose("c", {"x": 0.5, "y": 0.2})
    pinned_out = _Trace({"a": "z"}, [])
    pinned_out.choose("a", {"x": 1.0})
    assert not pinned_out.possible and pinned_out.bits() == math.inf


def test_the_loss_report_groups_choices_and_prices_references():
    loss = Loss()
    before, after = {"frame": [0, 1]}, {"frame": [0, 2]}
    for value, p in (("stay", 0.5), ("stay", 0.5), ("left", 0.25), ("stay", 0.5)):
        ranked = [["stay", 0.5], ["left", 0.25], ["right", 0.25]]
        result = {"bits": -math.log2(p), "choices": [["cow@1,2", value, p, False, ranked]]}
        loss.add(result, {"episode_id": 1, "step": 1}, before, after)
    report = loss.report()
    cow = report["by_choice"][0]
    assert report["steps"] == 4 and report["total"] == pytest.approx(5)
    assert cow["choice"] == "cow" and cow["uses"] == 4
    assert cow["observed"] == {'"stay"': 0.75, '"left"': 0.25}
    assert cow["declared"] == {'"stay"': 0.5, '"left"': 0.25}
    assert cow["excess_bits"] == pytest.approx(5 - 4 * 0.811, abs=0.06)  # rounded to 0.1
    assert report["references_per_step"]["no_model"] > 0
    half = next(b for b in report["calibration"] if b["declared_up_to"] == 0.5)
    assert half == {"declared_up_to": 0.5, "options": 12, "expected_to_happen": 4.0, "happened": 4}


def test_stochastic_options_are_validated():
    config = CwmConfig.from_mapping({"stochastic": True, "max_surprise_bits_per_choice": 6.5})
    assert config.stochastic and config.max_surprise_bits_per_choice == 6.5
    with pytest.raises(ValueError):
        CwmConfig.from_mapping({"stochastic": "yes"})
