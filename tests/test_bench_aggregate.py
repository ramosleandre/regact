"""The benchmark aggregator's load-bearing piece: controller classification.

``scripts/`` is not importable as a package, so load the module by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_aggregate.py"
_spec = importlib.util.spec_from_file_location("bench_aggregate", _SCRIPT)
assert _spec and _spec.loader
bench_aggregate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_aggregate)


def _ctrl(body: str) -> str:
    return f"class C:\n    def act(self, obs):\n        {body}\n"


_STUB = "class C:\n    def act(self, obs):\n        raise NotImplementedError\n"
_TRIVIAL_FIRST = _ctrl("return obs.available_actions[0]")
_TRIVIAL_CONST = _ctrl("return 1")
_TRIVIAL_RANDOM = "import random\n" + _ctrl("return random.choice(obs.available_actions)")
_REASONED_FRAME = _ctrl("return 2 if obs.frame[0][0] == 5 else 1")
_REASONED_STATE = _ctrl(
    "self.t = getattr(self, 't', 0) + 1\n        return obs.available_actions[self.t % 2]"
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (_STUB, "stub"),
        (_TRIVIAL_FIRST, "trivial"),
        (_TRIVIAL_CONST, "trivial"),
        (_TRIVIAL_RANDOM, "trivial"),
        (_REASONED_FRAME, "reasoned"),
        (_REASONED_STATE, "reasoned"),
    ],
)
def test_classify_controller(tmp_path: Path, source: str, expected: str) -> None:
    path = tmp_path / "solution.py"
    path.write_text(source)
    assert bench_aggregate._classify_controller(path) == expected


def test_classify_controller_missing_and_unparsable(tmp_path: Path) -> None:
    assert bench_aggregate._classify_controller(tmp_path / "nope.py") == "missing"
    bad = tmp_path / "bad.py"
    bad.write_text("class C:\n    def act(self, obs)\n        return 1\n")  # syntax error
    assert bench_aggregate._classify_controller(bad) == "unparsable"
    no_act = tmp_path / "no_act.py"
    no_act.write_text("x = 1\n")
    assert bench_aggregate._classify_controller(no_act) == "unparsable"
