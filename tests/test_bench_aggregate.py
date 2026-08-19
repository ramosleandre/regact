"""The benchmark aggregator's load-bearing piece: controller classification.

``scripts/`` is not importable as a package, so load the module by path.
"""

from __future__ import annotations

import importlib.util
import json
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


def test_classify_controller_follows_factored_code_library(tmp_path: Path) -> None:
    """A thin ``solution.py`` subclass of an agent-written ``code_library`` controller is
    classified by the real ``act`` in that module - not reported ``unparsable``."""
    lib = tmp_path / "code_library"
    lib.mkdir()
    (lib / "__init__.py").write_text("")
    (lib / "smart_controller.py").write_text(
        "class SmartController:\n"
        "    def act(self, obs):\n"
        "        for a in obs.available_actions:\n"  # a loop over the obs = reasoned
        "            return a\n"
    )
    sol = tmp_path / "solution.py"
    sol.write_text(
        "from code_library.smart_controller import SmartController\n"
        "class Controller(SmartController):\n    pass\n"
        "def get_controller():\n    return Controller()\n"
    )
    assert bench_aggregate._classify_controller(sol) == "reasoned"


def test_classify_controller_follows_transitive_base_and_relative_import(tmp_path: Path) -> None:
    """The follow chains through a middle module and handles a relative import."""
    lib = tmp_path / "code_library"
    lib.mkdir()
    (lib / "__init__.py").write_text("")
    (lib / "base_controller.py").write_text(
        "class BaseController:\n    def act(self, obs):\n        return 1\n"  # constant = trivial
    )
    (lib / "smart_controller.py").write_text(
        "from .base_controller import BaseController\n"
        "class SmartController(BaseController):\n    pass\n"
    )
    sol = tmp_path / "solution.py"
    sol.write_text(
        "from code_library.smart_controller import SmartController\n"
        "class Controller(SmartController):\n    pass\n"
    )
    assert bench_aggregate._classify_controller(sol) == "trivial"


def test_classify_controller_thin_subclass_without_modules_is_unparsable(tmp_path: Path) -> None:
    """No local ``act`` and the imported module isn't present -> graceful ``unparsable``."""
    sol = tmp_path / "solution.py"
    sol.write_text(
        "from code_library.smart_controller import SmartController\n"
        "class Controller(SmartController):\n    pass\n"
    )
    assert bench_aggregate._classify_controller(sol) == "unparsable"


def _mk_run(task_dir: Path, *, model: str, success: float) -> None:
    """A minimal on-disk run dir: config.json + logs/ + a submitted solution + final result."""
    task_dir.mkdir(parents=True)
    (task_dir / "config.json").write_text(
        json.dumps({"agent": {"name": "alan", "model": f"openai/{model}"}, "problem": {"seed": 0}})
    )
    (task_dir / "logs").mkdir()
    wd = task_dir / "workdir"
    wd.mkdir()
    (wd / "solution.py").write_text(_TRIVIAL_CONST)
    final = wd / "submissions" / "final"
    final.mkdir(parents=True)
    (final / "results.json").write_text(json.dumps({"aggregate": {"success_rate": success}}))


def test_collect_runs_flat_and_model_grouped_layouts(tmp_path: Path) -> None:
    """Runs are discovered at any depth: a flat ``exp/stamp/task`` and a
    model-grouped ``model/exp/stamp/task`` tree both yield one row per run."""
    # Flat: root/exp/stamp/task
    _mk_run(tmp_path / "exp_A" / "2026-01-01_00-00-00" / "TaskX", model="Flat-7B", success=1.0)
    # Model-grouped: root/model/exp/stamp/task (one level deeper)
    _mk_run(
        tmp_path / "ModelM" / "exp_B" / "2026-01-01_00-00-00" / "TaskY",
        model="Grouped-70B",
        success=0.0,
    )
    rows = bench_aggregate.collect_runs(tmp_path, all_stamps=False)
    by_model = {r["model"]: r for r in rows}
    assert set(by_model) == {"Flat-7B", "Grouped-70B"}
    assert by_model["Flat-7B"]["task"] == "TaskX"
    assert by_model["Flat-7B"]["success_rate"] == 1.0
    assert by_model["Grouped-70B"]["task"] == "TaskY"


def test_collect_runs_keeps_latest_stamp_unless_all(tmp_path: Path) -> None:
    """Two reruns of the same (experiment, task): default keeps the newest stamp only."""
    exp = tmp_path / "exp_A"
    _mk_run(exp / "2026-01-01_00-00-00" / "TaskX", model="M", success=0.0)  # older
    _mk_run(exp / "2026-01-02_00-00-00" / "TaskX", model="M", success=1.0)  # newer

    latest = bench_aggregate.collect_runs(tmp_path, all_stamps=False)
    assert len(latest) == 1
    assert latest[0]["stamp"] == "2026-01-02_00-00-00"
    assert latest[0]["success_rate"] == 1.0

    both = bench_aggregate.collect_runs(tmp_path, all_stamps=True)
    assert len(both) == 2
